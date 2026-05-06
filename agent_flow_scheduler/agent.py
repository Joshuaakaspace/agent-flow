"""
agent.py
--------
Agent abstraction: wraps a WorkflowScheduler + WorkflowGraph node.

An Agent knows its role in the workflow and how to invoke itself
through the scheduler. Agents can be chained to build pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .scheduler import WorkflowScheduler
from .workflow import WorkflowGraph, WorkflowNode


class AgentRole(str, Enum):
    RETRIEVER   = "retriever"
    REASONER    = "reasoner"
    CRITIC      = "critic"
    SUMMARIZER  = "summarizer"
    PLANNER     = "planner"
    EXECUTOR    = "executor"
    VALIDATOR   = "validator"
    SYNTHESIZER = "synthesizer"
    GENERIC     = "generic"


@dataclass
class AgentRunResult:
    agent_name: str
    input_prompt: str
    output: str
    latency_ms: float
    request_id: str
    workflow_id: str
    priority: str


class Agent:
    """
    A named agent in a multi-agent workflow.

    Usage:
        retriever = Agent("retriever", scheduler, graph, role=AgentRole.RETRIEVER)
        result = retriever.run("Find papers about speculative decoding", workflow_id="wf-1")
    """

    def __init__(
        self,
        name: str,
        scheduler: WorkflowScheduler,
        graph: WorkflowGraph,
        role: AgentRole = AgentRole.GENERIC,
        prompt_template: Optional[str] = None,
    ):
        self.name = name
        self.scheduler = scheduler
        self.graph = graph
        self.role = role
        self.prompt_template = prompt_template or "{input}"

        # Ensure this agent is registered in the workflow graph
        if not graph.get_node(name):
            graph.add_node(WorkflowNode(name=name, role=role.value))

    def run(self, input_text: str, workflow_id: Optional[str] = None) -> AgentRunResult:
        """Run this agent synchronously and return the result."""
        prompt = self.prompt_template.format(input=input_text)
        start = time.monotonic()

        result_text, req = self.scheduler.submit_and_wait(
            agent_name=self.name,
            prompt=prompt,
            workflow_id=workflow_id,
        )

        latency_ms = (time.monotonic() - start) * 1000
        return AgentRunResult(
            agent_name=self.name,
            input_prompt=input_text,
            output=result_text or "",
            latency_ms=latency_ms,
            request_id=req.request_id,
            workflow_id=req.workflow_id,
            priority=req.priority.name,
        )

    def run_async(
        self,
        input_text: str,
        workflow_id: Optional[str] = None,
        callback: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        """Submit request asynchronously; returns request_id."""
        prompt = self.prompt_template.format(input=input_text)
        req = self.scheduler.submit(
            agent_name=self.name,
            prompt=prompt,
            workflow_id=workflow_id,
            callback=callback,
        )
        return req.request_id

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role.value!r})"


class Pipeline:
    """
    A sequential chain of agents sharing a workflow_id.

    Usage:
        pipeline = Pipeline([retriever, reasoner, writer])
        final_result = pipeline.run("Explain transformers")
    """

    def __init__(self, agents: List[Agent], name: str = "pipeline"):
        self.agents = agents
        self.name = name

    def run(self, initial_input: str, workflow_id: Optional[str] = None) -> List[AgentRunResult]:
        """Run all agents in sequence, feeding each output to the next."""
        import uuid
        wf_id = workflow_id or f"wf-{uuid.uuid4().hex[:8]}"

        results: List[AgentRunResult] = []
        current_input = initial_input

        for agent in self.agents:
            result = agent.run(current_input, workflow_id=wf_id)
            results.append(result)
            current_input = result.output

        return results

    def summary(self, results: List[AgentRunResult]) -> str:
        lines = [f"Pipeline: {self.name}"]
        total_ms = sum(r.latency_ms for r in results)
        for r in results:
            lines.append(
                f"  [{r.priority:8s}] {r.agent_name:20s} {r.latency_ms:6.0f}ms"
            )
        lines.append(f"  {'TOTAL':28s} {total_ms:6.0f}ms")
        return "\n".join(lines)
