"""Per-generation leader checkpoints for all three training phases.

A phase is hours long and its inner searches are LLM-driven, so the run can end
for reasons that have nothing to do with the programs it found: the API drops, a
reply comes back unparseable at the wrong moment, the box reboots. Before this,
losing the process lost every program it had evolved -- the artifacts are only
written by ``freeze_*`` at the very END of a phase.

So every inner search calls ``checkpoint_fn(*labels, leader)`` after each
generation's selection, and :class:`LeaderCheckpoint` writes that leader to
``cache/phase{N}_ckpt/<run>/``: the score body as plain ``.py`` source, the full
meta + measured evaluation as ``.json``, and one line per checkpoint in
``history.jsonl`` so the whole trajectory is auditable afterwards.

**Deliberately NOT the frozen artifact path.** ``freeze_skill`` /
``freeze_combiner`` write into ``pref_dispatch/evolved/``, which
:func:`pref_dispatch.llm.basis.load_basis` LOADS. Checkpointing there would put
half-trained programs into the live repository -- every later round of the same
run would then be measured against them. Checkpoints live under ``cache/`` and
are inert until a human moves one.

Naming: the file stem is the labels EXCLUDING the last one, which is always the
generation. Phase 1 checkpoints ``("directed", 0, gen)`` -> ``directed_0``, and
``("qd", 3, gen)`` -> ``qd_3``, so each search keeps its own latest leader while
generations of one search overwrite each other. Phases 2 and 3 pass only ``(gen,)``
-> ``leader``, one search per run.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

_EVAL_FIELDS = (
    # group / combiner evals (whichever the phase produced -- duck-typed, since a
    # checkpoint must never be the reason a run dies)
    "fitness", "raw_fitness", "invalid_rate", "fallback_rate",
    "per_scenario_adv", "per_scenario_raw", "per_band", "per_family", "labels",
)


def _eval_dict(ev: Any) -> Optional[Dict]:
    if ev is None:
        return None
    out: Dict[str, Any] = {"type": type(ev).__name__}
    for f in _EVAL_FIELDS:
        if not hasattr(ev, f):
            continue
        v = getattr(ev, f)
        try:
            if isinstance(v, dict):
                out[f] = {str(k): float(x) for k, x in v.items()}
            elif isinstance(v, (list, tuple)):
                out[f] = [x if isinstance(x, str) else float(x) for x in v]
            else:
                out[f] = float(v)
        except Exception:  # noqa: BLE001 -- an unconvertible field is not fatal
            out[f] = repr(v)
    return out


class LeaderCheckpoint:
    """Callable that writes the current leader of an inner search to ``cache/``.

    Accepts ``(*labels, candidate)`` so ONE writer serves every phase: Phase 1
    passes ``(stage, index, generation, leader)``, Phases 2 and 3 pass
    ``(generation, leader)``.

    Every write is wrapped: a checkpoint that raised would kill a run it exists to
    protect, so a failure is logged and the search continues.
    """

    def __init__(
        self,
        phase: int,
        *,
        run: Optional[str] = None,
        root: str = os.path.join("cache", ""),
        log: Callable[[str], None] = print,
    ) -> None:
        self.phase = int(phase)
        self.run = run or time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root or ".", f"phase{self.phase}_ckpt", self.run)
        self.log = log
        self.n = 0
        os.makedirs(self.dir, exist_ok=True)

    def __call__(self, *args: Any) -> None:
        if not args:
            return
        labels: List[Any] = list(args[:-1])
        cand = args[-1]
        try:
            self._write(labels, cand)
        except Exception as e:  # noqa: BLE001 -- see the class docstring
            self.log(f"    [ckpt] FAILED to write leader ({type(e).__name__}: {e})")

    def _write(self, labels: List[Any], cand: Any) -> None:
        stem = "_".join(str(x) for x in labels[:-1]) or "leader"
        meta = dict(getattr(cand, "meta", {}) or {})
        code = str(meta.get("code", "") or "")
        record = {
            "phase": self.phase,
            "run": self.run,
            "at": [str(x) for x in labels],
            "generation": labels[-1] if labels else None,
            "name": getattr(cand, "name", "?"),
            "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta,
            "evaluation": _eval_dict(getattr(cand, "evaluation", None)),
        }
        if code:
            with open(os.path.join(self.dir, f"{stem}.py"), "w",
                      encoding="utf-8") as f:
                f.write(f"# Phase-{self.phase} CHECKPOINT (not a frozen artifact): "
                        f"{record['name']} at {record['at']}\n"
                        f"# Written {record['wall_clock']}. Inert until moved into "
                        f"pref_dispatch/evolved/ by hand.\n\n" + code.rstrip() + "\n")
        with open(os.path.join(self.dir, f"{stem}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.dir, "history.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps({k: record[k] for k in
                                ("at", "name", "wall_clock", "evaluation")},
                               ensure_ascii=False) + "\n")
        self.n += 1
        ev = record["evaluation"] or {}
        self.log(f"    [ckpt] {record['name']!r} -> "
                 f"{os.path.join(self.dir, stem)}.json"
                 + (f" (fitness {ev['fitness']:+.3f})" if "fitness" in ev else ""))
