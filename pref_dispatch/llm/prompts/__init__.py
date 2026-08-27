"""Prompt builders for LLM-driven evolution (skills now; combiner in M3).

Prompts are kept separate from the evolution control flow so they can be built
and eyeballed with no API call.
"""

from __future__ import annotations

from pref_dispatch.llm.prompts.skill_evolve import (
    REQUIRED_EXPLANATION_FIELDS,
    REQUIRED_FIELDS,
    build_skill_improve_prompt,
    build_skill_prompt,
)

__all__ = [
    "build_skill_prompt",
    "build_skill_improve_prompt",
    "REQUIRED_EXPLANATION_FIELDS",
    "REQUIRED_FIELDS",
]
