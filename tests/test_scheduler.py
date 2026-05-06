"""
Tests for agent-flow-scheduler core components.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent_flow_scheduler import (
    WorkflowGraph, WorkflowNode,
    WorkflowScheduler, MockLLMBackend,
    Priority, ScheduledRequest,
    SpeculativePrefillCache,
    MetricsCollector,
)


# -----------------------------------------------------------------------
# WorkflowGraph Tests
# -----------------------------------------------------------------------

def make_linear_graph(n: int = 3) -> WorkflowGraph:
    names = ["agent_" + str(i) for i in range(n)]
    g = WorkflowGraph("test-linear")
    for name in names:
        g.add_node(WorkflowNode(name=name, role="generic", estimated_latency_ms=100))
    for i in range(n - 1):
        g.add_edge(names[i], names[i + 1])
    g.finalize()
    return g


class TestWorkflowGraph:

    def test_add_nodes_and_edges(self):
        g = make_linear_graph(3)
        assert len(g.all_nodes()) == 3
        assert len(g.all_edges()) == 2

    def test_topological_order(self):
        g = make_linear_graph(4)
        order = g.topological_order()
        assert order == ["agent_0", "agent_1", "agent_2", "agent_3"]

    def test_depth_from_exit(self):
        g = make_linear_graph(4)
        assert g.depth_from_exit("agent_3") == 0  # exit
        assert g.depth_from_exit("agent_2") == 1
        assert g.depth_from_exit("agent_1") == 2
        assert g.depth_from_exit("agent_0") == 3

    def test_remaining_work(self):
        g = make_linear_graph(3)  # all 100ms each
        # agent_0: 100ms own + (100 + 100) downstream = 300ms
        assert g.remaining_work_ms("agent_0") == pytest.approx(300.0)
        assert g.remaining_work_ms("agent_2") == pytest.approx(100.0)

    def test_predict_next_agents(self):
        g = make_linear_graph(3)
        assert g.predict_next_agents("agent_0") == ["agent_1"]
        assert g.predict_next_agents("agent_2") == []  # exit node

    def test_cycle_detection(self):
        g = WorkflowGraph("cyclic")
        g.add_node(WorkflowNode("a", "generic"))
        g.add_node(WorkflowNode("b", "generic"))
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            g.finalize()

    def test_unknown_node_edge(self):
        g = WorkflowGraph("test")
        g.add_node(WorkflowNode("a", "generic"))
        with pytest.raises(ValueError):
            g.add_edge("a", "nonexistent")

    def test_branching_graph(self):
        g = WorkflowGraph("branching")
        for name in ["root", "left", "right", "merge"]:
            g.add_node(WorkflowNode(name=name, role="generic", estimated_latency_ms=100))
        g.add_edge("root", "left")
        g.add_edge("root", "right")
        g.add_edge("left",  "merge")
        g.add_edge("right", "merge")
        g.finalize()

        next_agents = g.predict_next_agents("root")
        assert set(next_agents) == {"left", "right"}


# -----------------------------------------------------------------------
# Priority Computation Tests
# -----------------------------------------------------------------------

class TestPriorityComputation:

    def test_exit_node_is_critical(self):
        g = make_linear_graph(3)
        backend = MockLLMBackend(add_jitter=False)
        sched = WorkflowScheduler(g, backend)
        assert sched._compute_priority("agent_2") == Priority.CRITICAL

    def test_penultimate_is_high(self):
        g = make_linear_graph(4)
        backend = MockLLMBackend(add_jitter=False)
        sched = WorkflowScheduler(g, backend)
        assert sched._compute_priority("agent_2") == Priority.HIGH

    def test_entry_node_is_low(self):
        g = make_linear_graph(5)
        backend = MockLLMBackend(add_jitter=False)
        sched = WorkflowScheduler(g, backend)
        assert sched._compute_priority("agent_0") == Priority.LOW


# -----------------------------------------------------------------------
# Scheduler Functional Tests
# -----------------------------------------------------------------------

class TestWorkflowScheduler:

    def setup_method(self):
        self.graph = make_linear_graph(3)
        self.backend = MockLLMBackend(add_jitter=False)
        self.scheduler = WorkflowScheduler(
            self.graph, self.backend,
            max_concurrent=2,
            enable_speculative_prefill=False,
        )
        self.scheduler.start()

    def teardown_method(self):
        self.scheduler.stop()

    def test_submit_and_wait_returns_result(self):
        result, req = self.scheduler.submit_and_wait(
            agent_name="agent_0",
            prompt="Hello world",
            workflow_id="test-wf-001",
        )
        assert result is not None
        assert len(result) > 0

    def test_request_has_correct_priority(self):
        req = self.scheduler.submit(
            agent_name="agent_2",  # exit node
            prompt="Test",
        )
        assert req.priority == Priority.CRITICAL

    def test_callback_is_called(self):
        received = []
        event = threading.Event()

        def cb(request_id, result):
            received.append(result)
            event.set()

        self.scheduler.submit(
            agent_name="agent_0",
            prompt="Callback test",
            callback=cb,
        )
        event.wait(timeout=5.0)
        assert len(received) == 1
        assert received[0] is not None

    def test_concurrent_requests(self):
        results = []
        lock = threading.Lock()

        def run(i):
            r, req = self.scheduler.submit_and_wait(
                agent_name="agent_1",
                prompt=f"Request {i}",
                workflow_id=f"concurrent-{i}",
            )
            with lock:
                results.append(r)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(results) == 6

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError):
            self.scheduler.submit(agent_name="nonexistent", prompt="test")

    def test_queue_depth(self):
        # Queue should be 0 initially (no pending requests)
        assert self.scheduler.queue_depth() == 0


# -----------------------------------------------------------------------
# SpeculativePrefillCache Tests
# -----------------------------------------------------------------------

class TestSpeculativePrefillCache:

    def test_warm_and_hit(self):
        cache = SpeculativePrefillCache()
        cache.warm("You are a helpful assistant.", estimated_token_count=10)
        hit, entry = cache.check_hit("You are a helpful assistant.")
        assert hit is True
        assert entry is not None
        assert entry.hit_count == 1

    def test_miss_on_unknown_prefix(self):
        cache = SpeculativePrefillCache()
        hit, entry = cache.check_hit("Unknown prefix that was never warmed")
        assert hit is False
        assert entry is None

    def test_ttl_expiry(self):
        cache = SpeculativePrefillCache(ttl_seconds=0.05)  # 50ms TTL
        cache.warm("Short-lived prefix.")
        time.sleep(0.1)  # Wait for TTL to expire
        hit, _ = cache.check_hit("Short-lived prefix.")
        assert hit is False

    def test_hit_rate_calculation(self):
        cache = SpeculativePrefillCache()
        cache.warm("prefix-a", 50)
        cache.warm("prefix-b", 30)

        cache.check_hit("prefix-a")        # hit
        cache.check_hit("prefix-a")        # hit
        cache.check_hit("prefix-unknown")  # miss

        assert abs(cache.hit_rate - 2/3) < 0.01

    def test_eviction_at_capacity(self):
        cache = SpeculativePrefillCache(max_entries=3)
        for i in range(4):
            cache.warm(f"prefix-{i}", 10)
        assert len(cache.active_entries()) <= 3

    def test_tokens_saved_tracking(self):
        cache = SpeculativePrefillCache()
        cache.warm("big system prompt here", estimated_token_count=100)
        cache.check_hit("big system prompt here")
        assert cache.total_tokens_saved == 100


# -----------------------------------------------------------------------
# MetricsCollector Tests
# -----------------------------------------------------------------------

class TestMetricsCollector:

    def test_compute_with_records(self):
        mc = MetricsCollector()
        t0 = time.monotonic()
        for i in range(5):
            mc.record(
                request_id=f"req-{i}",
                workflow_id="wf-1",
                agent_name="agent_0",
                priority="NORMAL",
                enqueue_time=t0 + i * 0.1,
                start_time=t0 + i * 0.1 + 0.05,
                end_time=t0 + i * 0.1 + 0.55,
                tokens_out=100,
            )
        metrics = mc.compute()
        assert metrics.total_requests == 5
        assert metrics.avg_wait_ms > 0
        assert metrics.avg_e2e_ms > 0
        assert metrics.throughput_rps > 0

    def test_empty_metrics(self):
        mc = MetricsCollector()
        m = mc.compute()
        assert m.total_requests == 0

    def test_reset(self):
        mc = MetricsCollector()
        mc.record("r1", "wf1", "a", "HIGH", 0, 0.1, 0.5)
        mc.reset()
        m = mc.compute()
        assert m.total_requests == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
