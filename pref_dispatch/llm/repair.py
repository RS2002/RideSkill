"""Shared repair-pass policy for every phase's propose-validate-retry loop.

All three phases ask the model for code, compile it in the sandbox, and -- when
that fails -- hand the error back for a retry. Two rules were learned the hard way
in Phase 2 and belong to all of them, so they live here instead of being re-typed:

* **Retry COOLER, not identical.** Retrying at the same high temperature just
  re-rolls the same dice. Four Phase-2 runs died at generation 0 on a JSON payload
  that would not parse, all four at temperature 1.0, while the 0.9 run succeeded.
  Malformed output is exactly the failure mode high temperature causes, so the
  recovery attempt should ask more conservatively -- diversity is worth paying for
  in the FIRST attempt, not in the repair.
* **Keep the payload that failed.** Those same four runs recorded nothing but the
  exception, so every fix was a guess at what the model had actually sent.
  :func:`dump_unparseable` writes the completion to disk. It is model output, not
  credentials, and the PROMPT is deliberately not written.

Lifted out of ``evolve_combiner`` on 2026-08-10 when Phases 1 and 3 were rebuilt on
the same (mu+lambda) design; behaviour is unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Optional

# Each repair attempt drops the temperature by this much, floored at the minimum.
REPAIR_COOLDOWN = 0.25
REPAIR_MIN_TEMPERATURE = 0.3

# Where unparseable completions are dumped for diagnosis.
UNPARSEABLE_DIR = os.path.join("cache", "unparseable")


def repair_temperature(base: Optional[float], attempt: int) -> Optional[float]:
    """Temperature for repair ``attempt`` (0 = the first, full-temperature try)."""
    if base is None or attempt <= 0:
        return base
    return max(REPAIR_MIN_TEMPERATURE, base - REPAIR_COOLDOWN * attempt)


def dump_unparseable(raw: str, header: str = "") -> Optional[str]:
    """Write a completion that would not parse to disk, and return its path.

    ``header`` carries the API's own verdict on the reply (``finish_reason`` and the
    completion-token count) as a ``#`` comment line. It is worth the two extra
    fields: the 2026-08-10 run lost three crossover slots to replies cut off at
    ``max_tokens``, and the only way to tell that from a genuinely malformed payload
    was to eyeball six dumps and notice they all stopped at the same byte count.
    ``finish_reason == "length"`` says it in one word.
    """
    try:
        os.makedirs(UNPARSEABLE_DIR, exist_ok=True)
        # Millisecond stamps collide when two candidates fail back-to-back, which
        # would drop one of the very payloads this exists to preserve. Open with
        # "x" (exclusive create) and walk a suffix until a free name is found.
        stem = os.path.join(UNPARSEABLE_DIR, f"completion_{int(time.time() * 1000)}")
        body = raw if isinstance(raw, str) else repr(raw)
        if header:
            body = f"# {header}\n{body}"
        for suffix in ("", *(f"_{k}" for k in range(1, 1000))):
            path = f"{stem}{suffix}.txt"
            try:
                with open(path, "x", encoding="utf-8") as fh:
                    fh.write(body)
                return path
            except FileExistsError:
                continue
        return None
    except Exception:  # noqa: BLE001 -- diagnostics must never break the run
        return None


def client_reply_header(client) -> str:
    """One-line ``finish_reason=... completion_tokens=...`` header for a dump."""
    usage = getattr(client, "last_usage", None) or {}
    return (f"finish_reason={getattr(client, 'last_finish_reason', None)} "
            f"completion_tokens={usage.get('completion_tokens')}")
