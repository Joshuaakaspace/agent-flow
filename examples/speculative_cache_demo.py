"""
Demonstrates speculative prefix-cache warming.

When agent A completes, the scheduler proactively warms the
KV-cache for agent B's system prompt (the predicted next step).

This demo shows:
  - Cache warm calls triggered per completed request
  - Hit rate climbing as the same workflow pattern repeats
  - Estimated token savings from cache hits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from agent_flow_scheduler import (
    WorkflowGraph, WorkflowNode,
    WorkflowScheduler, MockLLMBackend,
    SpeculativePrefillCache,
)


SYSTEM_PROMPTS = {
    "planner":    "You are a careful planning agent. Break tasks into clear steps.",
    "retriever":  "You are a retrieval agent. Find the most relevant documents.",
    "reasoner":   "You are a logical reasoning agent. Analyze evidence carefully.",
    "validator":  "You are a validation agent. Check outputs for correctness.",
    "synthesizer":"You are a synthesis agent. Combine findings into a final report.",
}


def build_graph() -> WorkflowGraph:
    g = WorkflowGraph("speculative-cache-demo")
    for name, prompt in SYSTEM_PROMPTS.items():
        g.add_node(WorkflowNode(
            name=name,
            role=name,
            estimated_latency_ms={"planner": 400, "retriever": 150, "reasoner": 800,
                                   "validator": 250, "synthesizer": 500}[name],
            system_prompt_prefix=prompt,
        ))
    g.add_edge("planner",   "retriever")
    g.add_edge("retriever", "reasoner")
    g.add_edge("reasoner",  "validator")
    g.add_edge("validator", "synthesizer")
    g.finalize()
    return g


def main():
    graph   = build_graph()
    backend = MockLLMBackend(add_jitter=False)
    cache   = SpeculativePrefillCache(max_entries=32, ttl_seconds=60.0)

    # Wire the cache into the backend's warm_prefix_cache method
    original_warm = backend.warm_prefix_cache
    def instrumented_warm(prefix_text: str):
        cache.warm(prefix_text)
        original_warm(prefix_text)
    backend.warm_prefix_cache = instrumented_warm

    # Also intercept backend.call to simulate cache hit detection
    original_call = backend.call
    def instrumented_call(agent_name, prompt, system_prompt="", **kwargs):
        hit, entry = cache.check_hit(system_prompt)
        if hit:
            print(f"    [CACHE HIT]  {agent_name} prefix ({entry.token_count} tokens saved)")
        else:
            print(f"    [cache miss] {agent_name} prefix (cold)")
        return original_call(agent_name, prompt, system_prompt, **kwargs)
    backend.call = instrumented_call

    scheduler = WorkflowScheduler(
        graph, backend,
        max_concurrent=2,
        enable_speculative_prefill=True,
    )
    scheduler.start()

    print("=" * 60)
    print("Speculative Prefill Cache Demo")
    print("=" * 60)
    print("\nRunning 5 identical workflows to demonstrate cache warm-up:\n")

    for run in range(5):
        print(f"--- Run {run+1} ---")
        wf_id = f"wf-{run:03d}"
        agents = ["planner", "retriever", "reasoner", "validator", "synthesizer"]
        for agent in agents:
            result, req = scheduler.submit_and_wait(
                agent_name=agent,
                prompt=f"Process task {run+1}",
                workflow_id=wf_id,
            )
        print(f"  Cache stats: {cache.report()}")
        print()

    scheduler.stop()

    print("\n" + "=" * 60)
    print("Final Cache Statistics:")
    stats = cache.stats()
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")
    print(f"\nSpeculative prefill warmed next-agent caches, achieving a")
    print(f"{stats['hit_rate']} hit rate and saving ~{stats['tokens_saved']} tokens")
    print("in prefix re-computation across all runs.")


if __name__ == "__main__":
    main()
