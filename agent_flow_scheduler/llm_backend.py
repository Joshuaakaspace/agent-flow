"""
llm_backend.py
--------------
LLM backend abstraction and a MockLLMBackend for local testing.

The LLMBackend interface is intentionally thin so you can swap in:
  - OpenAI API (openai.ChatCompletion)
  - Anthropic API (anthropic.Anthropic)
  - Local vLLM endpoint
  - Any other backend

MockLLMBackend simulates realistic per-agent latency distributions
so you can benchmark the scheduler without real API keys.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def call(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        """Call the LLM and return the response string."""
        ...

    def warm_prefix_cache(self, prefix_text: str) -> None:
        """
        Optionally pre-warm the KV cache for a system-prompt prefix.
        In production: triggers prefill computation on the GPU for prefix_text.
        Default implementation is a no-op.
        """
        pass

    def health_check(self) -> bool:
        return True


class MockLLMBackend(LLMBackend):
    """
    Simulates an LLM backend with configurable latency per agent role.

    Latency model:
      base_latency_ms + jitter + (prompt_len * token_latency_per_char)

    Prefix cache simulation:
      If the system prompt prefix was warmed, we reduce prefill latency by 40%.
    """

    # Realistic latency profiles per agent role
    ROLE_LATENCY_MS: Dict[str, float] = {
        "retriever":   150.0,
        "planner":     400.0,
        "reasoner":    800.0,
        "critic":      600.0,
        "summarizer":  350.0,
        "executor":    300.0,
        "validator":   250.0,
        "synthesizer": 500.0,
        "generic":     400.0,
    }

    ROLE_RESPONSES: Dict[str, List[str]] = {
        "retriever": [
            "Retrieved 5 relevant documents. Top result: 'Attention is All You Need' (Vaswani et al., 2017). Documents cover transformer architectures, self-attention mechanisms, and positional encodings.",
            "Found 3 matching papers on speculative decoding. Key papers: Chen et al. 2023, Leviathan et al. 2023. Relevant content on draft-verify paradigm extracted.",
        ],
        "planner": [
            "Plan: (1) Retrieve context. (2) Analyze key claims. (3) Identify gaps. (4) Synthesize findings. (5) Write summary.",
            "Execution plan: Step 1: Parse input query. Step 2: Decompose into sub-questions. Step 3: Assign to specialist agents. Step 4: Aggregate results.",
        ],
        "reasoner": [
            "Analysis: The proposed approach leverages workflow predictability to reduce scheduling uncertainty. Key insight: agents in a DAG have bounded output distributions, enabling proactive optimization. This yields 23% lower P95 latency vs. FIFO scheduling.",
            "Reasoning: The critical path through the workflow determines end-to-end latency. Prioritizing requests near the workflow exit minimizes average job completion time — consistent with SRPT (Shortest Remaining Processing Time) theory.",
        ],
        "critic": [
            "Critique: The analysis is sound but overlooks cold-start overhead. Speculative prefill has a ~15ms warmup cost that amortizes only across repeated workflow patterns. Recommend adding a pattern-frequency threshold before enabling speculation.",
            "Issues identified: (1) Priority inversion risk when CRITICAL requests queue behind long SPECULATIVE requests. (2) EMA update rate may be too slow for bursty workloads. Suggest adaptive alpha.",
        ],
        "summarizer": [
            "Summary: Workflow-aware scheduling reduces average wait time by 18-35% vs. FIFO by serving requests closer to completion first. Speculative prefill adds a further 8-12% gain for repetitive workflows.",
            "Key takeaway: The system prioritizes exit-node agents (CRITICAL), then penultimate agents (HIGH), falling back to FIFO within same-priority tiers. This is equivalent to a work-conserving SRPT policy adapted for DAG workflows.",
        ],
        "validator": [
            "Validation passed. Output consistency: 97.3%. No hallucinations detected in factual claims. Confidence: HIGH.",
            "Validation: 2 minor inconsistencies found. Agent 'reasoner' cited a paper from 2019 that was retracted in 2022. Flagging for human review.",
        ],
        "synthesizer": [
            "Synthesized report: Workflow-aware LLM scheduling is a promising direction for production agentic systems. The Pythia paper (arxiv 2604.25899) demonstrates 1.4x throughput improvement on real production traces. Our implementation reproduces the core scheduling logic.",
            "Final synthesis: Three key contributions: (1) DAG-based remaining-work estimation, (2) SRPT-inspired priority queuing, (3) speculative prefix cache warming. Together these reduce P95 E2E latency by ~28%.",
        ],
        "executor": [
            "Action executed: API call to knowledge base completed. Results: 12 records returned. Status: SUCCESS.",
            "Tool call completed: web_search('speculative decoding LLM 2026'). Top 5 results fetched and parsed.",
        ],
        "generic": [
            "Processed request successfully. Output generated based on input context.",
            "Task completed. Results available for downstream processing.",
        ],
    }

    def __init__(self, add_jitter: bool = True, jitter_factor: float = 0.3):
        self.add_jitter = add_jitter
        self.jitter_factor = jitter_factor
        self._warmed_prefixes: Set[str] = set()
        self._call_count = 0

    def call(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        self._call_count += 1

        # Determine role from agent_name (try suffix match)
        role = self._infer_role(agent_name)
        base_ms = self.ROLE_LATENCY_MS.get(role, 400.0)

        # Add prompt-length overhead (~0.1ms per word)
        prompt_overhead_ms = len(prompt.split()) * 0.1

        # Jitter (lognormal)
        if self.add_jitter:
            jitter_ms = random.gauss(0, base_ms * self.jitter_factor)
        else:
            jitter_ms = 0.0

        # Prefix cache hit reduces prefill latency
        prefill_saving_ms = 0.0
        if system_prompt and system_prompt in self._warmed_prefixes:
            prefill_saving_ms = base_ms * 0.40  # 40% reduction on cache hit

        total_ms = max(10.0, base_ms + prompt_overhead_ms + jitter_ms - prefill_saving_ms)
        time.sleep(total_ms / 1000.0)

        # Pick a response
        responses = self.ROLE_RESPONSES.get(role, self.ROLE_RESPONSES["generic"])
        return random.choice(responses)

    def warm_prefix_cache(self, prefix_text: str) -> None:
        """Simulate async KV-cache warm-up (non-blocking in real systems)."""
        self._warmed_prefixes.add(prefix_text)

    @property
    def call_count(self) -> int:
        return self._call_count

    def _infer_role(self, agent_name: str) -> str:
        name_lower = agent_name.lower()
        for role in self.ROLE_LATENCY_MS:
            if role in name_lower:
                return role
        return "generic"


class OpenAIBackend(LLMBackend):
    """
    Real OpenAI backend.

    Install: pip install openai
    Set:     OPENAI_API_KEY environment variable

    Usage:
        from agent_flow_scheduler.llm_backend import OpenAIBackend
        backend = OpenAIBackend(model="gpt-4o-mini")
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        try:
            from openai import OpenAI
            self._client = OpenAI()
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model
        self.temperature = temperature

    def call(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """
    Real Anthropic backend.

    Install: pip install anthropic
    Set:     ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 512):
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError("pip install anthropic")
        self.model = model
        self.default_max_tokens = max_tokens

    def call(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 0,
    ) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens or self.default_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        import anthropic
        response = self._client.messages.create(**kwargs)
        return response.content[0].text
