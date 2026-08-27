"""LLM configuration (provider / model / endpoint / secrets).

The key design rule: **the API key is never stored here or anywhere in the
repo** -- this config only names the *environment variable* to read it from
(loaded from a git-ignored ``.env`` via python-dotenv). See ``llm_design.md``
section 2.1.

Two providers are supported behind one :class:`~pref_dispatch.llm.client.LLMClient`
interface:

* ``"api"`` (default) -- an OpenAI-compatible HTTP endpoint. Verified working
  against the third-party yibuapi gateway with ``claude-opus-4-8`` (the backend
  is Anthropic Claude, which supports prompt caching).
* ``"hf"``  -- a local HuggingFace ``transformers`` model (offline, no key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """All knobs for talking to the model. Never holds the key itself."""

    provider: str = "api"  # "api" | "hf"
    model: str = "claude-opus-4-8"
    base_url: str = "https://yibuapi.com/v1"

    # Name of the environment variable that holds the API key. The key is read
    # from ``os.environ[api_key_env]`` (or a git-ignored .env) at client build
    # time -- it is deliberately NOT a field here so it cannot be pickled,
    # logged, or committed by accident.
    api_key_env: str = "YIBU_API_KEY"

    # Evolution wants diversity -> high default temperature. Deterministic
    # passes (final-product eval, sandbox trial runs) override this with 0.0
    # at call time.
    temperature: float = 0.9
    # Truncation shows up as an ExtractionError, NOT as a "too long" error, so it
    # is worth being generous here. History: 2048 cut LLM-authored skill JSON;
    # 4096 held until v6 added the two-parent CROSSOVER operator, whose reply has
    # to carry BOTH parents' merge rationale plus the merged program -- three
    # crossover replies in the 2026-08-10 run died mid-`code` string at exactly
    # 4096 (~9.5 KB; JSON escaping costs ~2.3 chars/token). Worse than losing the
    # slot, the ones that DID survive a cooler retry were the SHORT ones, which
    # selects for brevity instead of quality. 16384 leaves real headroom; it is a
    # cap, not a spend, so an unused ceiling costs nothing.
    max_tokens: int = 16384
    n_retry: int = 3
    timeout: float = 120.0

    # provider="hf": local weights path (ignored for the API provider).
    hf_model_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.provider not in ("api", "hf"):
            raise ValueError(
                f"provider must be 'api' or 'hf', got {self.provider!r}"
            )
        if self.provider == "hf" and not self.hf_model_path:
            raise ValueError("provider='hf' requires hf_model_path")
