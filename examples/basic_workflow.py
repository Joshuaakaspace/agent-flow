"""
Basic 3-agent RAG pipeline demonstrating workflow-aware scheduling.

Workflow:
  retriever --> reasoner --> summarizer

Priority assignment:
  retriever  = LOW       (most remaining work: 3 hops to exit)
  reasoner   = NORMAL    (2 hops)
  summarizer = CRITICAL  (exit node)

When multiple concurrent requests exist, summarizer requests
are served first — minimizing average E2E latency.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_flow_scheduler import (
    WorkflowGraph, WorkflowNode,
    WorkflowScheduler, MockLLMBackend,
    Agent, AgentRole, Pipeline,
    MetricsCollector,
)


def build_rag_workflow() -> WorkflowGraph:
    graph = WorkflowGraph(name="rag-pipeline")

    graph.add_node(WorkflowNode(
        name="retriever",
        role="retriever",
        estimated_latency_ms=150,
        system_prompt_prefix="You are a document retrieval specialist. Find the most relevant context.",
    ))
    graph.add_node(WorkflowNode(
        name="reasoner",
        role="reasoner",
        estimated_latency_ms=800,
        system_prompt_prefix="You are a logical reasoner. Analyze the retrieved context carefully.",
    ))
    graph.add_node(WorkflowNode(
        name="summarizer",
        role="summarizer",
        estimated_latency_ms=350,
        system_prompt_prefix="You are a concise summarizer. Produce a clear, brief summary.",
    ))

    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner",  "summarizer")
    graph.finalize()
    return graph


def run_pipeline(query: str, graph: WorkflowGraph, backend: MockLLMBackend) -> None:
    scheduler = WorkflowScheduler(graph, backend, max_concurrent=2)
    scheduler.start()

    metrics = MetricsCollector()

    retriever  = Agent("retriever",  scheduler, graph, role=AgentRole.RETRIEVER)
    reasoner   = Agent("reasoner",   scheduler, graph, role=AgentRole.REASONER)
    summarizer = Agent("summarizer", scheduler, graph, role=AgentRole.SUMMARIZER)

    pipeline = Pipeline([retriever, reasoner, summarizer], name="rag-pipeline")

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    print("\nWorkflow graph:")
    print(graph.summary())

    print("\nRunning pipeline...")
    results = pipeline.run(query, workflow_id="demo-001")

    print("\n" + pipeline.summary(results))

    print("\nAgent outputs:")
    for r in results:
        print(f"\n[{r.agent_name.upper()} | {r.priority} priority | {r.latency_ms:.0f}ms]")
        print(f"  {r.output[:200]}...")

    for r in results:
        metrics.record_from_request(
            type("R", (), {
                "request_id": r.request_id,
                "workflow_id": r.workflow_id,
                "agent_name": r.agent_name,
                "priority": type("P", (), {"name": r.priority})(),
                "enqueue_time": 0.0,
                "start_time": 0.001,
                "end_time": r.latency_ms / 1000.0,
            })()
        )

    scheduler.stop()


def main():
    graph   = build_rag_workflow()
    backend = MockLLMBackend(add_jitter=True)

    queries = [
        "Explain how speculative decoding works in transformer inference",
        "What are the tradeoffs between different KV cache compression strategies?",
    ]

    for query in queries:
        run_pipeline(query, graph, backend)

    print(f"\nTotal LLM calls made: {backend.call_count}")
    print("\nDone! The scheduler assigned priorities:")
    print("  retriever  -> LOW      (3 hops to exit)")
    print("  reasoner   -> NORMAL   (2 hops to exit)")
    print("  summarizer -> CRITICAL (0 hops, exit node)")
    print("\nUnder concurrent load, CRITICAL requests are served first,")
    print("minimising average end-to-end latency across all workflows.")


if __name__ == "__main__":
    main()
