"""LLM integration for preference-dispatch skill / combiner evolution.

This subpackage turns the platform's LLM (local HuggingFace *or* an external
OpenAI-compatible API, default the latter) into an *offline* code generator:
it writes frozen Python scoring / combining functions during evolution, which
the dispatch loop then executes at ~zero online LLM cost (paradigm B). See
``llm_design.md`` for the full design.

Public surface (built incrementally, milestones 7.1-7.7):

* :mod:`pref_dispatch.llm.config`  -- :class:`LLMConfig`.
* :mod:`pref_dispatch.llm.client`  -- :class:`LLMClient`, API/HF backends.
* :mod:`pref_dispatch.llm.extract` -- robust JSON/code extraction.
"""

from __future__ import annotations

from pref_dispatch.llm.config import LLMConfig

__all__ = ["LLMConfig"]
