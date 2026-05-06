"""
scheduler.py
------------
WorkflowScheduler: priority-based request scheduler for multi-agent LLM pipelines.

Core idea (from Pythia, arxiv 2604.25899):
  - Requests are NOT treated as independent — they belong to a workflow
  - Priority = inverse of remaining_work: requests close to completion are served first
    (Shortest Remaining Processing Time, adapted for workflow graphs)
  - This minimises average end-to-end latency and reduces head-of-line blocking

Priority levels (coarse-grained for compatibility with external queuing systems):
  CRITICAL  — exit-node agents (one hop from done)
  HIGH      — penultimate agents (2 hops from done)
  NORMAL    — mid-workflow
  LOW       — entry agents (most remaining work ahead)
  SPECULATIVE — prefill-only, can be dropped under load
"""

from __future__ import annotations

import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

from .workflow import WorkflowGraph


class Priority(IntEnum):
    CRITICAL    = 0   # Lowest number = highest priority in min-heap
    HIGH        = 1
    NORMAL      = 2
    LOW         = 3
    SPECULATIVE = 4


@dataclass
class ScheduledRequest:
    """A single LLM call wrapped with scheduling metadata."""

    request_id: str
    workflow_id: str
    agent_name: str
    prompt: str
    priority: Priority = Priority.NORMAL
    remaining_work_ms: float = 0.0
    enqueue_time: float = field(default_factory=time.monotonic)
    callback: Optional[Callable[[str, str], None]] = field(default=None, repr=False)

    # Set by scheduler after completion
    result: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def wait_time_ms(self) -> float:
        if self.start_time is None:
            return (time.monotonic() - self.enqueue_time) * 1000
        return (self.start_time - self.enqueue_time) * 1000

    @property
    def execution_time_ms(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    # Comparison for heapq (min-heap on priority, then remaining_work_ms descending)
    def __lt__(self, other: "ScheduledRequest") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        # Within same priority: prefer requests with LESS remaining work
        return self.remaining_work_ms < other.remaining_work_ms


@dataclass
class _HeapItem:
    priority: int
    remaining_ms: float
    seq: int
    request: ScheduledRequest

    def __lt__(self, other: "_HeapItem") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.remaining_ms != other.remaining_ms:
            return self.remaining_ms < other.remaining_ms
        return self.seq < other.seq


class WorkflowScheduler:
    """
    Workflow-aware scheduler for multi-agent LLM requests.

    Usage:
        scheduler = WorkflowScheduler(graph, backend)
        scheduler.start()

        req = scheduler.submit(
            agent_name="retriever",
            prompt="Find docs about transformers",
            workflow_id="wf-001",
        )
        result = scheduler.wait(req.request_id)
        scheduler.stop()

    The scheduler runs a background worker thread that dequeues requests
    in priority order and dispatches them to the LLM backend.
    """

    def __init__(
        self,
        graph: WorkflowGraph,
        backend,
        max_concurrent: int = 4,
        enable_speculative_prefill: bool = True,
    ):
        self.graph = graph
        self.backend = backend
        self.max_concurrent = max_concurrent
        self.enable_speculative_prefill = enable_speculative_prefill

        self._heap: List[_HeapItem] = []
        self._heap_lock = threading.Lock()
        self._seq = 0

        self._pending: Dict[str, ScheduledRequest] = {}
        self._completed: Dict[str, ScheduledRequest] = {}
        self._results_events: Dict[str, threading.Event] = {}

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._semaphore = threading.Semaphore(max_concurrent)

        # Stats
        self.total_submitted = 0
        self.total_completed = 0
        self.total_speculative_hits = 0
        self.total_speculative_misses = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "WorkflowScheduler":
        self._running = True
        self._worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._worker_thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        agent_name: str,
        prompt: str,
        workflow_id: Optional[str] = None,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> ScheduledRequest:
        """
        Enqueue a request for the given agent.
        Returns immediately; use wait() or callback for the result.
        """
        node = self.graph.get_node(agent_name)
        if node is None:
            raise ValueError(f"Agent '{agent_name}' not found in workflow graph.")

        remaining = self.graph.remaining_work_ms(agent_name)
        priority = self._compute_priority(agent_name)

        req = ScheduledRequest(
            request_id=str(uuid.uuid4()),
            workflow_id=workflow_id or str(uuid.uuid4()),
            agent_name=agent_name,
            prompt=prompt,
            priority=priority,
            remaining_work_ms=remaining,
            callback=callback,
        )

        event = threading.Event()
        with self._heap_lock:
            self._seq += 1
            item = _HeapItem(
                priority=int(priority),
                remaining_ms=remaining,
                seq=self._seq,
                request=req,
            )
            heapq.heappush(self._heap, item)
            self._pending[req.request_id] = req
            self._results_events[req.request_id] = event

        self.total_submitted += 1
        return req

    def wait(self, request_id: str, timeout: float = 30.0) -> Optional[str]:
        """Block until the request completes and return its result."""
        event = self._results_events.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request_id: {request_id}")
        event.wait(timeout=timeout)
        completed = self._completed.get(request_id)
        return completed.result if completed else None

    def submit_and_wait(
        self,
        agent_name: str,
        prompt: str,
        workflow_id: Optional[str] = None,
    ) -> Tuple[str, ScheduledRequest]:
        """Convenience: submit and block until result is available."""
        req = self.submit(agent_name, prompt, workflow_id)
        result = self.wait(req.request_id)
        return result, req

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_priority(self, agent_name: str) -> Priority:
        depth = self.graph.depth_from_exit(agent_name)
        if depth == 0:
            return Priority.CRITICAL
        elif depth == 1:
            return Priority.HIGH
        elif depth <= 3:
            return Priority.NORMAL
        else:
            return Priority.LOW

    def _dispatch_loop(self) -> None:
        while self._running:
            item = self._pop_next()
            if item is None:
                time.sleep(0.005)
                continue
            self._semaphore.acquire()
            t = threading.Thread(target=self._execute, args=(item,), daemon=True)
            t.start()

    def _pop_next(self) -> Optional[_HeapItem]:
        with self._heap_lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    def _execute(self, item: _HeapItem) -> None:
        req = item.request
        try:
            req.start_time = time.monotonic()

            # Optionally prime speculative prefill for next agents
            if self.enable_speculative_prefill:
                next_agents = self.graph.predict_next_agents(req.agent_name)
                for next_agent in next_agents:
                    next_node = self.graph.get_node(next_agent)
                    if next_node and next_node.system_prompt_prefix:
                        self.backend.warm_prefix_cache(next_node.system_prompt_prefix)

            result = self.backend.call(
                agent_name=req.agent_name,
                prompt=req.prompt,
                system_prompt=self._get_system_prompt(req.agent_name),
            )
            req.result = result
            req.end_time = time.monotonic()

            # Update node latency stats
            node = self.graph.get_node(req.agent_name)
            if node:
                latency_ms = req.execution_time_ms
                token_estimate = len(result.split())  # rough token count
                node.update_stats(latency_ms, token_estimate)

        except Exception as e:
            req.result = f"[ERROR] {e}"
            req.end_time = time.monotonic()
        finally:
            self._semaphore.release()
            with self._heap_lock:
                self._pending.pop(req.request_id, None)
                self._completed[req.request_id] = req

            event = self._results_events.get(req.request_id)
            if event:
                event.set()

            if req.callback:
                req.callback(req.request_id, req.result or "")

            self.total_completed += 1

    def _get_system_prompt(self, agent_name: str) -> str:
        node = self.graph.get_node(agent_name)
        if node:
            return node.system_prompt_prefix
        return ""

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def queue_depth(self) -> int:
        with self._heap_lock:
            return len(self._heap)

    def queue_snapshot(self) -> List[Dict]:
        """Return a snapshot of the current queue for monitoring."""
        with self._heap_lock:
            return [
                {
                    "request_id": item.request.request_id,
                    "agent": item.request.agent_name,
                    "priority": Priority(item.priority).name,
                    "remaining_ms": item.remaining_ms,
                    "wait_ms": item.request.wait_time_ms,
                }
                for item in sorted(self._heap)
            ]
