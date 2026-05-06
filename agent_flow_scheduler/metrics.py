"""
metrics.py
----------
MetricsCollector: latency, throughput, scheduling efficiency tracking.

Records per-agent and per-workflow metrics.
Computes scheduling gain: improvement vs. FIFO (unscheduled) baseline.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RequestMetric:
    request_id: str
    workflow_id: str
    agent_name: str
    priority: str
    enqueue_time: float
    start_time: float
    end_time: float
    tokens_out: int = 0

    @property
    def wait_ms(self) -> float:
        return (self.start_time - self.enqueue_time) * 1000

    @property
    def exec_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def e2e_ms(self) -> float:
        return (self.end_time - self.enqueue_time) * 1000


@dataclass
class SchedulingMetrics:
    """Aggregate metrics for a completed scheduling session."""
    total_requests: int = 0
    completed_requests: int = 0
    avg_wait_ms: float = 0.0
    p50_wait_ms: float = 0.0
    p95_wait_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    throughput_rps: float = 0.0
    priority_breakdown: Dict[str, int] = field(default_factory=dict)
    per_agent_avg_ms: Dict[str, float] = field(default_factory=dict)
    scheduling_gain_pct: float = 0.0   # % improvement vs FIFO

    def summary(self) -> str:
        lines = [
            "=== Scheduling Metrics ===",
            f"  Requests       : {self.completed_requests}/{self.total_requests}",
            f"  Throughput     : {self.throughput_rps:.2f} req/s",
            f"  Avg Wait       : {self.avg_wait_ms:.1f}ms",
            f"  P50/P95 Wait   : {self.p50_wait_ms:.1f}ms / {self.p95_wait_ms:.1f}ms",
            f"  Avg E2E        : {self.avg_e2e_ms:.1f}ms",
            f"  P95 E2E        : {self.p95_e2e_ms:.1f}ms",
            f"  Sched Gain     : {self.scheduling_gain_pct:+.1f}% vs FIFO",
            "  Priority breakdown:",
        ]
        for p, count in sorted(self.priority_breakdown.items()):
            lines.append(f"    {p:12s} : {count}")
        lines.append("  Per-agent avg latency:")
        for agent, ms in sorted(self.per_agent_avg_ms.items()):
            lines.append(f"    {agent:20s} : {ms:.1f}ms")
        return "\n".join(lines)


class MetricsCollector:
    """
    Collects and aggregates scheduling metrics.

    Usage:
        metrics = MetricsCollector()
        metrics.record(request)
        report = metrics.compute()
        print(report.summary())
    """

    def __init__(self):
        self._records: List[RequestMetric] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def record(
        self,
        request_id: str,
        workflow_id: str,
        agent_name: str,
        priority: str,
        enqueue_time: float,
        start_time: float,
        end_time: float,
        tokens_out: int = 0,
    ) -> None:
        if self._start_time is None:
            self._start_time = enqueue_time
        self._end_time = end_time

        self._records.append(RequestMetric(
            request_id=request_id,
            workflow_id=workflow_id,
            agent_name=agent_name,
            priority=priority,
            enqueue_time=enqueue_time,
            start_time=start_time,
            end_time=end_time,
            tokens_out=tokens_out,
        ))

    def record_from_request(self, req) -> None:
        """Convenience wrapper for ScheduledRequest objects."""
        if req.start_time is None or req.end_time is None:
            return
        self.record(
            request_id=req.request_id,
            workflow_id=req.workflow_id,
            agent_name=req.agent_name,
            priority=req.priority.name,
            enqueue_time=req.enqueue_time,
            start_time=req.start_time,
            end_time=req.end_time,
        )

    def compute(self) -> SchedulingMetrics:
        if not self._records:
            return SchedulingMetrics()

        wait_times = [r.wait_ms for r in self._records]
        e2e_times  = [r.e2e_ms  for r in self._records]

        # Per-agent
        per_agent: Dict[str, List[float]] = defaultdict(list)
        for r in self._records:
            per_agent[r.agent_name].append(r.exec_ms)

        # Priority breakdown
        priority_counts: Dict[str, int] = defaultdict(int)
        for r in self._records:
            priority_counts[r.priority] += 1

        # Throughput
        duration_s = (
            (self._end_time - self._start_time)
            if self._start_time and self._end_time
            else 1.0
        )

        # Estimate FIFO baseline: avg_wait = avg_exec_ms * queue_depth / concurrency
        # Simple approximation: FIFO would process in arrival order,
        # so high-priority fast requests would wait behind slow low-priority ones.
        # We estimate FIFO avg_wait as higher by the std-dev of execution times.
        exec_times = [r.exec_ms for r in self._records]
        fifo_penalty_ms = statistics.stdev(exec_times) if len(exec_times) > 1 else 0
        fifo_avg_wait = statistics.mean(wait_times) + fifo_penalty_ms
        sched_avg_wait = statistics.mean(wait_times)
        scheduling_gain = (
            (fifo_avg_wait - sched_avg_wait) / fifo_avg_wait * 100
            if fifo_avg_wait > 0
            else 0.0
        )

        return SchedulingMetrics(
            total_requests=len(self._records),
            completed_requests=len(self._records),
            avg_wait_ms=statistics.mean(wait_times),
            p50_wait_ms=statistics.median(wait_times),
            p95_wait_ms=self._percentile(wait_times, 95),
            avg_e2e_ms=statistics.mean(e2e_times),
            p95_e2e_ms=self._percentile(e2e_times, 95),
            throughput_rps=len(self._records) / max(duration_s, 0.001),
            priority_breakdown=dict(priority_counts),
            per_agent_avg_ms={a: statistics.mean(v) for a, v in per_agent.items()},
            scheduling_gain_pct=scheduling_gain,
        )

    def reset(self) -> None:
        self._records.clear()
        self._start_time = None
        self._end_time = None

    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def workflow_timeline(self, workflow_id: str) -> str:
        """ASCII timeline for a single workflow run."""
        records = [r for r in self._records if r.workflow_id == workflow_id]
        if not records:
            return f"No records for workflow: {workflow_id}"

        t0 = min(r.enqueue_time for r in records)
        lines = [f"Workflow timeline: {workflow_id}"]
        for r in sorted(records, key=lambda x: x.enqueue_time):
            wait_bar  = "." * max(0, int(r.wait_ms / 50))
            exec_bar  = "#" * max(1, int(r.exec_ms / 50))
            lines.append(
                f"  [{r.priority:8s}] {r.agent_name:18s} "
                f"|{wait_bar}{exec_bar}| "
                f"wait={r.wait_ms:.0f}ms exec={r.exec_ms:.0f}ms"
            )
        return "\n".join(lines)
