"""Read leader checkpoints back in, so an interrupted phase restarts where it stopped.

:mod:`pref_dispatch.llm.checkpoint` is the WRITE half: after every generation of
every inner search it drops the current leader under ``cache/phase{N}_ckpt/<run>/``.
This module is the READ half. Without it the checkpoints were only ever an audit
trail -- a run that died at direction 4 of 4 threw away three finished skills and
about an hour of search, because the artifacts are written by ``freeze_*`` at the
very END of a phase.

**Only a FINISHED search is resumed.** The writer overwrites one file per search
(``directed_2.json`` holds whichever leader was best as of the latest generation),
so a checkpoint on disk may be a half-evolved program from generation 2 of 5.
Restarting from that would silently accept a skill the search had not finished
improving, and nothing downstream would ever say so. :func:`load_directed_leader`
therefore compares the record's ``generation`` against the ``generations`` the new
run is configured for and returns ``None`` for a partial one -- that direction is
searched again from scratch. Raising ``--generations`` on resume correctly
re-opens every direction, because none of them reached the new final generation.

What resume does NOT cover: the QD stage (1b/1c). Its round count is not fixed in
advance and the repository state it mutates -- which members were accepted, which
were evicted, the fitted signature scaler -- is not in the checkpoints, so there is
no honest way to rebuild it mid-flight. A resumed run re-enters QD from the
recovered directed skills, which is where QD starts anyway.

**Phases 2 and 3 resume differently, and weaker.** Phase 1 runs several independent
directed searches, so a finished one can be REUSED as its own final answer. Phases 2
and 3 are ONE ``(mu+lambda)`` search each, and the checkpoint holds only that round's
leader -- not the archive, not the surviving population, not the objective batches
already drawn. So :func:`load_leader_seed` offers a WARM START, not a fast-forward:
the checkpointed champion is injected into generation 0 (the ``seed_code`` path that
already exists for fine-tuning), and the run then does all ``--generations`` rounds
again. Resuming a run that died in round 6 of 8 costs the full 8 rounds; what it
saves is the *program*, not the wall clock.

That also flips the completeness rule. Phase 1 refuses a partial checkpoint because
there the checkpoint would BE the answer. Here it is only a seed that must re-earn
its place against gen-0 rivals, so a leader from any generation is a legitimate
starting point and none is refused.

Nothing here touches the network or an API key.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from pref_dispatch.llm.evolve import Candidate
from pref_dispatch.llm.sandbox import (
    SandboxError,
    compile_combiner,
    compile_fitness,
    compile_repositioner,
    compile_skill,
)

CHECKPOINT_ROOT = "cache"


def checkpoint_dir(phase: int, run: str, root: str = CHECKPOINT_ROOT) -> str:
    """Directory :class:`~pref_dispatch.llm.checkpoint.LeaderCheckpoint` writes to."""
    return os.path.join(root or ".", f"phase{int(phase)}_ckpt", str(run))


def _read_record(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def candidate_from_record(record: Dict[str, Any]) -> Candidate:
    """Rebuild a runnable :class:`Candidate` from a checkpoint record.

    The score body and the LLM-authored fitness are recompiled through the SAME
    sandbox that admitted them originally, so a checkpoint cannot smuggle in code
    the live path would have rejected -- the file sits in ``cache/`` where anything
    could have edited it.

    ``evaluation`` is left ``None``: the JSON keeps only a flattened summary of the
    measured evaluation (floats and labels), not the object, and a resumed
    direction never reads it -- ``_basis_from_directed`` re-measures signatures on
    the fixed signature batch, and ``_discover_from_directed`` uses only ``.skill``
    and ``.meta``.
    """
    meta = dict(record.get("meta") or {})
    code = str(meta.get("code", "") or "")
    fitness_code = str(meta.get("fitness_code", "") or "")
    if not code:
        raise SandboxError("checkpoint record has no skill code")
    if not fitness_code:
        raise SandboxError("checkpoint record has no fitness code")
    skill = compile_skill(code, name=str(meta.get("skill_name", "resumed")))
    fitness_fn = compile_fitness(fitness_code)
    return Candidate(meta=meta, skill=skill, fitness_fn=fitness_fn, evaluation=None)


def load_directed_leader(
    run: str,
    index: int,
    *,
    generations: int,
    root: str = CHECKPOINT_ROOT,
    log: Callable[[str], None] = print,
) -> Optional[Candidate]:
    """The finished skill for directed direction ``index``, or ``None`` to re-evolve.

    Returns ``None`` -- meaning "search this direction again" -- when the checkpoint
    is absent, partial (see the module docstring), or no longer compiles. A resume
    must never be the reason a run dies, so every failure degrades to re-evolving.
    """
    path = os.path.join(checkpoint_dir(1, run, root), f"directed_{int(index)}.json")
    try:
        record = _read_record(path)
    except (OSError, ValueError) as e:
        log(f"[resume] direction {index}: unreadable checkpoint ({e}); re-evolving")
        return None
    if record is None:
        return None

    gen = record.get("generation")
    try:
        reached = int(gen)
    except (TypeError, ValueError):
        log(f"[resume] direction {index}: checkpoint has no generation; re-evolving")
        return None
    if reached < int(generations):
        log(f"[resume] direction {index}: checkpoint stopped at generation {reached} "
            f"of {generations} (unfinished); re-evolving from scratch")
        return None

    try:
        cand = candidate_from_record(record)
    except (SandboxError, KeyError, ValueError) as e:
        log(f"[resume] direction {index}: checkpoint will not recompile "
            f"({type(e).__name__}: {e}); re-evolving")
        return None

    log(f"[resume] direction {index}: reusing {cand.name!r} from generation {reached} "
        f"({path})")
    return cand


def load_leader_seed(
    phase: int,
    run: str,
    *,
    root: str = CHECKPOINT_ROOT,
    log: Callable[[str], None] = print,
) -> Optional[Dict[str, Any]]:
    """The checkpointed leader of a Phase-2/3 run, as a warm-start ``seed_meta``.

    Returns the record's ``meta`` (which carries ``code``) ready to hand to
    ``evolve_*``'s ``seed_code`` / ``seed_meta`` arguments, or ``None`` to start
    cold. Unlike :func:`load_directed_leader` this accepts a leader from ANY
    generation -- see the module docstring: here the checkpoint is a seed that has
    to win gen-0 selection on its own, not a finished answer being trusted.

    The body is compiled through the same sandbox the live path uses, so a
    checkpoint that was edited or truncated under ``cache/`` fails here -- loudly,
    before the run starts -- instead of at the first rollout. Every failure
    degrades to a cold start, because a resume must never be the reason a run dies.
    """
    p = int(phase)
    path = os.path.join(checkpoint_dir(p, run, root), "leader.json")
    try:
        record = _read_record(path)
    except (OSError, ValueError) as e:
        log(f"[resume] phase {p}: unreadable checkpoint ({e}); starting cold")
        return None
    if record is None:
        log(f"[resume] phase {p}: no leader checkpoint at {path}; starting cold")
        return None

    meta = dict(record.get("meta") or {})
    code = str(meta.get("code", "") or "")
    if not code:
        log(f"[resume] phase {p}: checkpoint holds no code; starting cold")
        return None
    compile_fn = compile_combiner if p == 2 else compile_repositioner
    try:
        compile_fn(code)
    except (SandboxError, SyntaxError, ValueError) as e:
        log(f"[resume] phase {p}: checkpoint will not recompile "
            f"({type(e).__name__}: {e}); starting cold")
        return None

    log(f"[resume] phase {p}: warm-starting generation 0 from {record.get('name', '?')!r} "
        f"(checkpointed at generation {record.get('generation', '?')}, "
        f"{record.get('wall_clock', '?')}). All rounds will be re-run -- this "
        f"recovers the PROGRAM, not the elapsed search.")
    return meta


def describe_run(
    run: str,
    *,
    phase: int = 1,
    root: str = CHECKPOINT_ROOT,
) -> List[str]:
    """One human-readable line per checkpoint found (for ``--resume`` logs)."""
    d = checkpoint_dir(phase, run, root)
    if not os.path.isdir(d):
        return [f"[resume] no checkpoint directory at {d}"]
    out: List[str] = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "history.jsonl":
            continue
        try:
            rec = _read_record(os.path.join(d, fn)) or {}
        except (OSError, ValueError):
            out.append(f"[resume] {fn}: unreadable")
            continue
        out.append(f"[resume] {fn}: {rec.get('name', '?')} "
                   f"at generation {rec.get('generation', '?')} "
                   f"({rec.get('wall_clock', '?')})")
    return out or [f"[resume] {d} holds no checkpoints"]
