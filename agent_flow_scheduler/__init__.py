"""
agent-flow-scheduler: Workflow-aware scheduling for multi-agent LLM pipelines.
Inspired by: Pythia (arxiv 2604.25899, April 2026)
"""

from .workflow import WorkflowGraph, WorkflowNode
from .scheduler import WorkflowScheduler, ScheduledRequest, Priority
from .agent import Agent, AgentRole, Pipeline, AgentRunResult
from .cache import SpeculativePrefillCache, CacheEntry
from .metrics import MetricsCollector, SchedulingMetrics
from .llm_backend import LLMBackend, MockLLMBackend

__version__ = "0.1.0"
__all__ = [
    "WorkflowGraph", "WorkflowNode", "WorkflowScheduler", "ScheduledRequest",
    "Priority", "Agent", "AgentRole", "Pipeline", "AgentRunResult",
    "SpeculativePrefillCache", "CacheEntry", "MetricsCollector",
    "SchedulingMetrics", "LLMBackend", "MockLLMBackend",
]
