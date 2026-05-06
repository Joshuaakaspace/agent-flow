"""
workflow.py
-----------
Defines the WorkflowGraph: a directed acyclic graph (DAG) of agent nodes.

Each node represents an agent role in a multi-agent pipeline.
Edges represent "calls-next" relationships between agents.

The graph is used by the scheduler to:
  1. Estimate remaining work for any in-flight request
  2. Assign scheduling priorities (requests closer to completion get higher priority)
  3. Predict next-step agents for speculative prefill
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class WorkflowNode:
    """Represents a single agent in the workflow DAG."""

    name: str
    role: str                          # e.g. "retriever", "reasoner", "summarizer"
    estimated_tokens_out: int = 512    # Expected output token count (used to predict cost)
    estimated_latency_ms: float = 500  # Baseline latency in ms (updated at runtime)
    system_prompt_prefix: str = ""     # Shared system-prompt prefix for prefix-cache warm-up

    # Runtime statistics (updated as requests flow through)
    _observed_latencies: List[float] = field(default_factory=list, repr=False)
    _observed_token_counts: List[int] = field(default_factory=list, repr=False)

    def update_stats(self, latency_ms: float, tokens_out: int) -> None:
        """EMA update for latency and token count estimates."""
        alpha = 0.2
        self._observed_latencies.append(latency_ms)
        self._observed_token_counts.append(tokens_out)
        # Exponential moving average
        self.estimated_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.estimated_latency_ms
        )
        self.estimated_tokens_out = int(
            alpha * tokens_out + (1 - alpha) * self.estimated_tokens_out
        )

    @property
    def avg_latency_ms(self) -> float:
        if not self._observed_latencies:
            return self.estimated_latency_ms
        return sum(self._observed_latencies[-20:]) / len(self._observed_latencies[-20:])


class WorkflowGraph:
    """
    Directed Acyclic Graph of agent nodes.

    Usage:
        graph = WorkflowGraph()
        graph.add_node(WorkflowNode("retriever", "retrieval", estimated_latency_ms=200))
        graph.add_node(WorkflowNode("reasoner", "reasoning", estimated_latency_ms=800))
        graph.add_node(WorkflowNode("writer",   "synthesis", estimated_latency_ms=600))
        graph.add_edge("retriever", "reasoner")
        graph.add_edge("reasoner",  "writer")
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._nodes: Dict[str, WorkflowNode] = {}
        self._edges: Dict[str, List[str]] = {}      # parent -> [children]
        self._reverse: Dict[str, List[str]] = {}    # child  -> [parents]
        self._entry_nodes: Set[str] = set()
        self._exit_nodes: Set[str] = set()
        self._topo_cache: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: WorkflowNode) -> "WorkflowGraph":
        self._nodes[node.name] = node
        if node.name not in self._edges:
            self._edges[node.name] = []
        if node.name not in self._reverse:
            self._reverse[node.name] = []
        self._topo_cache = None
        return self

    def add_edge(self, from_node: str, to_node: str) -> "WorkflowGraph":
        if from_node not in self._nodes:
            raise ValueError(f"Node '{from_node}' not found in graph.")
        if to_node not in self._nodes:
            raise ValueError(f"Node '{to_node}' not found in graph.")
        self._edges[from_node].append(to_node)
        self._reverse[to_node].append(from_node)
        self._topo_cache = None
        return self

    def finalize(self) -> "WorkflowGraph":
        """Compute entry/exit nodes and validate the graph is a DAG."""
        self._entry_nodes = {n for n in self._nodes if not self._reverse[n]}
        self._exit_nodes  = {n for n in self._nodes if not self._edges[n]}
        if self._detect_cycle():
            raise ValueError("WorkflowGraph contains a cycle — only DAGs are supported.")
        return self

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def remaining_work_ms(self, current_node: str) -> float:
        """
        Estimated remaining latency (ms) from current_node to the end of the workflow.
        Uses the longest path (critical path) through the remaining graph.
        """
        node = self._nodes.get(current_node)
        if node is None:
            return 0.0
        return node.estimated_latency_ms + self._critical_path_after(current_node)

    def _critical_path_after(self, node_name: str) -> float:
        """Recursively compute the critical path length after node_name."""
        successors = self._edges.get(node_name, [])
        if not successors:
            return 0.0
        return max(
            self._nodes[s].estimated_latency_ms + self._critical_path_after(s)
            for s in successors
        )

    def depth_from_exit(self, node_name: str) -> int:
        """Number of hops from this node to the nearest exit node."""
        if node_name in self._exit_nodes:
            return 0
        successors = self._edges.get(node_name, [])
        if not successors:
            return 0
        return 1 + min(self.depth_from_exit(s) for s in successors)

    def predict_next_agents(self, current_node: str) -> List[str]:
        """Return direct successors of current_node (candidates for speculative prefill)."""
        return list(self._edges.get(current_node, []))

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (Kahn's algorithm)."""
        if self._topo_cache is not None:
            return self._topo_cache
        in_degree = {n: len(self._reverse[n]) for n in self._nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self._edges[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        self._topo_cache = order
        return order

    def _detect_cycle(self) -> bool:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        return any(dfs(n) for n in self._nodes if n not in visited)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> Optional[WorkflowNode]:
        return self._nodes.get(name)

    def all_nodes(self) -> List[WorkflowNode]:
        return list(self._nodes.values())

    def all_edges(self) -> List[Tuple[str, str]]:
        return [
            (src, dst)
            for src, dsts in self._edges.items()
            for dst in dsts
        ]

    def summary(self) -> str:
        lines = [f"WorkflowGraph: {self.name}"]
        for name in self.topological_order():
            node = self._nodes[name]
            succs = self._edges[name]
            arrow = " -> " + ", ".join(succs) if succs else " [EXIT]"
            lines.append(
                f"  [{node.role:12s}] {name:20s} ~{node.estimated_latency_ms:5.0f}ms{arrow}"
            )
        return "\n".join(lines)
