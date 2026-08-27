"""LLM client backends behind one ``complete(system, user)`` interface.

Two implementations, selected by :class:`~pref_dispatch.llm.config.LLMConfig`:

* :class:`APIClient` (default) -- OpenAI-compatible HTTP. The key is read from
  the environment variable named by ``cfg.api_key_env`` (loaded from a
  git-ignored ``.env`` if python-dotenv is installed); it is never taken from
  config or code. Retries with exponential backoff.
* :class:`HFClient` -- a local ``transformers`` model, same interface, no key.
  Imported lazily so the API path has no heavy dependency.

Both return the raw completion string; parsing lives in
:mod:`pref_dispatch.llm.extract` so it is client-agnostic and unit-testable.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Protocol

from pref_dispatch.llm.config import LLMConfig


class LLMClient(Protocol):
    """Minimal chat interface the evolution loop depends on."""

    def complete(
        self, system: str, user: str, *, temperature: Optional[float] = None
    ) -> str:
        """Return the raw assistant message text for one system+user turn."""
        ...


def _load_dotenv_once() -> None:
    """Best-effort ``.env`` load so ``os.environ`` has the key. No-op if the
    package is absent or there is no file."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _read_key(cfg: LLMConfig) -> str:
    _load_dotenv_once()
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise RuntimeError(
            f"API key not found in environment variable {cfg.api_key_env!r}. "
            f"Set it in your shell or a git-ignored .env file "
            f"(e.g. `export {cfg.api_key_env}=sk-...`). "
            "The key must never be written into the repo."
        )
    return key


class APIClient:
    """OpenAI-compatible client (works for Anthropic/OpenAI/DeepSeek gateways).

    Switching backend is just ``base_url`` + ``model`` in the config.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._key = _read_key(cfg)
        # Usage of the most recent successful call: {"prompt_tokens",
        # "completion_tokens"} or None. Populated from the API response so a
        # metering wrapper (paradigm-C compute table) can report REAL token
        # counts rather than an estimate. Purely observational; never affects
        # control flow.
        self.last_usage: Optional[dict] = None
        # ``finish_reason`` of the most recent call: "stop" (the model chose to
        # end) vs "length" (it hit max_tokens mid-sentence). Recorded because a
        # cut-off reply reaches the extractor looking like a SYNTAX error, which
        # sends the diagnosis at the JSON parser instead of the cap -- three v6
        # crossover replies were lost that way. Observational only.
        self.last_finish_reason: Optional[str] = None

    def _fresh_client(self):
        from openai import OpenAI

        # Fresh client per attempt: a hung socket poisons the httpx connection
        # pool, and later attempts would reuse the stuck connection. A new
        # client means a new connection every time.
        return OpenAI(
            api_key=self._key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
        )

    def _call(self, client, system, user, temp: float) -> str:
        resp = client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temp,
            max_tokens=self.cfg.max_tokens,
        )
        usage = getattr(resp, "usage", None)
        self.last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
        self.last_usage = (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
            if usage is not None
            else None
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            # Empty completion (proxy hiccup / model refusal without text):
            # retry rather than let the extractor fail on a no-JSON string.
            raise RuntimeError("empty completion content")
        return content

    def complete(self, system, user, *, temperature=None):
        temp = self.cfg.temperature if temperature is None else temperature
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.n_retry):
            client = self._fresh_client()
            result: dict = {}

            def _attempt():
                try:
                    result["content"] = self._call(client, system, user, temp)
                except Exception as e:  # noqa: BLE001 -- network/transient errors
                    result["err"] = e

            t = threading.Thread(target=_attempt, name=f"llm-{attempt}", daemon=True)
            t.start()
            # Hard wall-clock bound. Some proxies hang on large generations and
            # the SDK timeout never fires (observed: 15-min stall with
            # timeout=120). join() caps the attempt regardless of socket state;
            # the daemon thread is abandoned, its connection dies with it.
            t.join(self.cfg.timeout + 5.0)
            if t.is_alive():
                result["err"] = RuntimeError(
                    f"LLM call exceeded {self.cfg.timeout + 5.0:.0f}s wall-clock "
                    "(socket hung; SDK timeout not firing)"
                )
            if "content" in result:
                return result["content"]
            last_err = result.get("err")
            if attempt < self.cfg.n_retry - 1:
                time.sleep(2.0 ** attempt)  # 1s, 2s, 4s, ...
        raise RuntimeError(
            f"LLM call failed after {self.cfg.n_retry} attempts: {last_err}"
        ) from last_err


class HFClient:
    """Local HuggingFace ``transformers`` backend (offline, no key)."""

    def __init__(self, cfg: LLMConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.cfg = cfg
        self._tok = AutoTokenizer.from_pretrained(cfg.hf_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_path, device_map="auto"
        )
        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=self._tok
        )

    def complete(self, system, user, *, temperature=None):
        temp = self.cfg.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self._pipe(
            prompt,
            max_new_tokens=self.cfg.max_tokens,
            do_sample=temp > 0,
            temperature=max(temp, 1e-6),
            return_full_text=False,
        )
        return out[0]["generated_text"]


def make_llm_client(cfg: Optional[LLMConfig] = None) -> LLMClient:
    """Build the client for ``cfg`` (defaults to :class:`LLMConfig`)."""
    cfg = cfg or LLMConfig()
    if cfg.provider == "api":
        return APIClient(cfg)
    if cfg.provider == "hf":
        return HFClient(cfg)
    raise ValueError(f"unknown provider {cfg.provider!r}")
