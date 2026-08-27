"""Robust extraction of the model's JSON payload from a raw completion.

The models we use wrap their answer in Markdown code fences and (sometimes) a
reasoning preamble, so parsing must be defensive. This mirrors the three-tier
fallback proven in the reference system's ``hsUtils.json_load`` (see
``llm_design.md`` section 2.2), and is deliberately **client-agnostic** so it
can be unit-tested with fixed strings, no API needed.

Pipeline:

1. Strip reasoning blocks (``<think>...</think>``) if present.
2. Strip code fences: prefer a ```json / ```python fenced block; else use the
   whole text.
3. Parse: ``json.loads`` first; on failure fall back to the longest
   ``{...}`` regex match and retry; finally try a PYTHON-literal read
   (:func:`ast.literal_eval`), which recovers the near-JSON the models actually
   emit under high temperature -- single-quoted keys, trailing commas,
   ``True``/``False``/``None``. That third tier was added after three separate
   Phase-2 runs died at generation 0 on
   ``Expecting property name enclosed in double quotes: line 1 column 2``, i.e. a
   payload that was one keystroke away from valid.

The extracted object must carry the contract fields for the current phase; the
caller validates those (a skill needs ``code``; a combiner needs ``code`` too).
Every LLM output is required to carry a natural-language explanation
(``description`` and, for skills, ``objective``) -- interpretability is a
headline selling point, so :func:`require_explanation` enforces it.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A fenced block, optionally tagged ```json / ```python. Non-greedy body.
_FENCE_RE = re.compile(
    r"```(?:json|python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
)


class ExtractionError(ValueError):
    """Raised when no JSON object can be recovered from a completion."""


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks."""
    return _THINK_RE.sub("", text)


def _candidate_blocks(text: str) -> List[str]:
    """Yield parse candidates, best-first: fenced blocks, then whole text."""
    blocks = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    blocks.append(text.strip())  # whole-text fallback
    return [b for b in blocks if b]


def _longest_brace_span(text: str) -> Optional[str]:
    """Return the longest balanced-looking ``{...}`` substring, or None.

    Uses brace matching (not a naive greedy regex) so nested objects survive.
    """
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:  # earliest start -> longest span
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]
    return None


def _loads(block: str) -> Dict:
    """``json.loads`` with ``strict=False`` so literal newlines/tabs inside code
    strings (which models often emit unescaped) don't fail the parse."""
    obj = json.loads(block, strict=False)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("not an object", block, 0)
    return obj


def _relax_to_json(block: str) -> str:
    """Rewrite near-JSON into strict JSON: the tier-3 repair.

    Handles exactly the drift the models show at high temperature:

    * single-quoted strings/keys -> double-quoted;
    * BARE keys -> quoted (``{combiner_name: ...}`` -> ``{"combiner_name": ...}``),
      the JS-object-literal drift that produces the ``line 1 column 2`` failure
      four Phase-2 generation-0 calls died on;
    * trailing commas before ``}`` / ``]``;
    * ``True`` / ``False`` / ``None`` -> ``true`` / ``false`` / ``null``.

    It is a character scanner, not a regex, so quotes and escapes inside the
    ``code`` payload survive: text inside a string is only ever re-quoted, never
    interpreted. Literal newlines inside strings are escaped to ``\\n`` so the
    result parses even under strict JSON.

    ``ast.literal_eval`` alone cannot do this job -- a Python string literal may
    not contain a raw newline, and the models' ``code`` field is full of them.
    """
    out: List[str] = []
    i, n = 0, len(block)
    last_sig = ""            # last significant char emitted (for bare-key detection)
    while i < n:
        c = block[i]
        # A bare key: an identifier sitting where a key belongs (right after
        # ``{`` or ``,``) and followed by ``:``. Quote it. Values are untouched
        # because they follow ``:``, never ``{``/``,``.
        if (c.isalpha() or c == "_") and last_sig in "{,":
            j = i
            while j < n and (block[j].isalnum() or block[j] == "_"):
                j += 1
            k = j
            while k < n and block[k] in " \t\r\n":
                k += 1
            if k < n and block[k] == ":":
                out.append('"' + block[i:j] + '"')
                last_sig = '"'
                i = j
                continue
        if c in "\"'":
            quote = c
            i += 1
            body: List[str] = []
            while i < n:
                ch = block[i]
                if ch == "\\" and i + 1 < n:
                    nxt = block[i + 1]
                    if quote == "'" and nxt == "'":
                        body.append("'")        # \' is not a JSON escape
                    else:
                        body.append(ch + nxt)
                    i += 2
                    continue
                if ch == quote:
                    i += 1
                    break
                if ch == '"':
                    body.append('\\"')          # bare " inside a '...' string
                elif ch == "\n":
                    body.append("\\n")
                elif ch == "\r":
                    body.append("\\r")
                elif ch == "\t":
                    body.append("\\t")
                else:
                    body.append(ch)
                i += 1
            out.append('"' + "".join(body) + '"')
            last_sig = '"'
            continue
        if c == ",":
            j = i + 1
            while j < n and block[j] in " \t\r\n":
                j += 1
            if j < n and block[j] in "}]":
                i += 1                          # drop the trailing comma
                continue
        for word, repl in (("True", "true"), ("False", "false"), ("None", "null")):
            if block.startswith(word, i):
                before = block[i - 1] if i else " "
                after = block[i + len(word)] if i + len(word) < n else " "
                if not (before.isalnum() or before == "_") and \
                   not (after.isalnum() or after == "_"):
                    out.append(repl)
                    i += len(word)
                    last_sig = repl[-1]
                    break
        else:
            out.append(c)
            if not c.isspace():
                last_sig = c
            i += 1
            continue
    return "".join(out)


def _loads_relaxed(block: str) -> Dict:
    """Tier-3 parse: strict-JSON rewrite first, then a PYTHON literal read.

    Neither path can execute model-supplied code -- the rewrite is textual and
    ``ast.literal_eval`` accepts data only (no calls, names or attribute access).
    The ``code`` field stays an inert string the caller validates and compiles
    exactly as before. Added after three Phase-2 runs died at generation 0 on a
    payload that was one quote character away from valid."""
    try:
        return _loads(_relax_to_json(block))
    except json.JSONDecodeError as first:
        try:
            obj = ast.literal_eval(block)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            raise ValueError(f"relaxed rewrite failed: {first}") from None
        if not isinstance(obj, dict):
            raise ValueError(f"python literal is {type(obj).__name__}, not a dict")
        # JSON object keys are strings; keep the contract identical downstream.
        return {str(k): v for k, v in obj.items()}


def _brace_balance(text: str) -> int:
    """``{`` count minus ``}`` count, ignoring braces inside string literals.

    Positive = the reply stops before closing what it opened (cut off at the END).
    Negative = it closes braces it never opened, i.e. the BEGINNING is missing.
    """
    s = strip_reasoning(text).strip()
    depth, in_str, esc, quote = 0, False, False, ""
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return depth


def _looks_truncated(text: str) -> bool:
    """True when the completion was CUT OFF AT THE END rather than mis-formatted.

    A reply that stops mid-answer is not a parse problem and no repair prompt can
    fix it. It is worth naming, because it presents itself as an ordinary syntax
    error: three v6 crossover replies died mid-``code`` string at the 4096-token
    cap and reported ``Expecting property name enclosed in double quotes``, which
    sent the diagnosis at the JSON parser instead of the cap.

    The test is deliberately narrow: the text opens an object and never closes it.
    It says the reply is INCOMPLETE, not why -- hitting ``max_tokens`` is the
    usual cause but not the only one, so the caller must check ``finish_reason``
    before blaming the cap (a 988-char reply under a 16384-token ceiling did not
    run out of room).
    """
    s = strip_reasoning(text).strip()
    if "{" not in s:
        return False
    depth, in_str, esc, quote = 0, False, False, ""
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return in_str or depth > 0


def _looks_head_chopped(text: str) -> bool:
    """True when the reply is missing its BEGINNING, not its end.

    A separate failure from truncation and it needs the opposite response. Seen
    on the 2026-08-10 gateway: three dumps opened mid-expression
    (``0)))\\n            v_2p = float(w(...``) and ended with a perfectly formed
    ``}``, closing more braces than they opened. Nothing about the prompt or the
    token cap causes that -- only a re-ask recovers it, which is what the repair
    loop already does. Naming it stops the next person from raising
    ``max_tokens`` again for a reply that was never too long.
    """
    return _brace_balance(text) < 0


def extract_json(text: str) -> Dict:
    """Recover the JSON object the model intended to return.

    Raises :class:`ExtractionError` if nothing parses.
    """
    cleaned = strip_reasoning(text)
    errors: List[str] = []
    for block in _candidate_blocks(cleaned):
        # Tier 1: direct parse.
        try:
            return _loads(block)
        except json.JSONDecodeError as e:
            errors.append(str(e))
        # Tier 2: longest balanced brace span within this block.
        span = _longest_brace_span(block)
        if span:
            try:
                return _loads(span)
            except json.JSONDecodeError as e:
                errors.append(str(e))
        # Tier 3: relaxed (near-JSON) read of the block, then of the brace span.
        for cand in (block, span):
            if not cand:
                continue
            try:
                return _loads_relaxed(cand)
            except (ValueError, SyntaxError, MemoryError, RecursionError) as e:
                errors.append(f"relaxed: {e}")
    if _looks_head_chopped(text):
        raise ExtractionError(
            f"completion is missing its BEGINNING ({len(text)} chars, closes more "
            "braces than it opens) -- the gateway returned a partial body. Not a "
            "format problem and not the token cap; only a re-ask recovers it."
        )
    if _looks_truncated(text):
        raise ExtractionError(
            f"completion is INCOMPLETE ({len(text)} chars, object never closed) -- "
            "the model stopped mid-answer, so repairing the prompt cannot fix it. "
            "Check finish_reason: 'length' means raise LLMConfig.max_tokens; "
            "anything else means the reply died early and a re-ask is the fix."
        )
    raise ExtractionError(
        "no JSON object found in completion; tried "
        f"{len(errors)} parse(s). Last error: {errors[-1] if errors else 'n/a'}"
    )


def require_explanation(obj: Dict, fields: tuple = ("description",)) -> None:
    """Enforce that natural-language explanation fields are present & non-empty.

    Interpretability is a headline contribution, so we reject any skill /
    combiner that ships code without a human-readable rationale.
    """
    for f in fields:
        v = obj.get(f)
        if not isinstance(v, str) or not v.strip():
            raise ExtractionError(
                f"missing/empty explanation field {f!r} (interpretability is "
                "required for every LLM output)"
            )
