"""Live round-trip smoke check for the LLM client (milestone 7.1 verify).

Run once to confirm the configured model returns a JSON object carrying an
exec-able ``code`` field plus its natural-language explanation:

    export YIBU_API_KEY=sk-...          # never commit the key
    python -m pref_dispatch.llm.check_roundtrip

It reads the key ONLY from the environment (or a git-ignored ``.env``); the key
is never taken from code. Exits non-zero on any failure so it can gate CI.
"""

from __future__ import annotations

import sys

from pref_dispatch.llm.client import make_llm_client
from pref_dispatch.llm.config import LLMConfig
from pref_dispatch.llm.extract import extract_json, require_explanation

_SYSTEM = (
    "You are a Python code generator. Reply with ONE JSON object only, no prose "
    "outside it."
)
_USER = (
    "Return a JSON object with fields: 'description' (a one-sentence natural-"
    "language explanation of what the function does) and 'code' (a Python "
    "function `def add(a, b): return a + b`). Nothing else."
)


def main() -> int:
    cfg = LLMConfig()
    print(f"provider={cfg.provider} model={cfg.model} base_url={cfg.base_url}")
    try:
        client = make_llm_client(cfg)
    except RuntimeError as e:
        print(f"[SETUP] {e}")
        return 2

    # Deterministic for a verification call.
    raw = client.complete(_SYSTEM, _USER, temperature=0.0)
    print(f"--- raw completion ({len(raw)} chars) ---\n{raw}\n---")

    obj = extract_json(raw)
    require_explanation(obj, ("description",))
    code = obj.get("code", "")
    if "def add" not in code:
        print(f"[FAIL] extracted object has no usable code: {obj}")
        return 1

    # Actually exec the returned code and call it.
    ns: dict = {}
    exec(code, {"__builtins__": {}}, ns)  # noqa: S102 -- trusted verify only
    result = ns["add"](2, 3)
    assert result == 5, f"exec'd add(2,3) returned {result!r}"

    print(f"[OK] round-trip: description={obj['description']!r}; add(2,3)={result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
