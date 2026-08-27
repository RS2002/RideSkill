"""Load the frozen skill basis (handwritten seeds + evolved skills) + their cards.

Phase 2 needs the frozen lower skills as a ``{name: Skill}`` map AND their prompt
cards (name / objective / description [/ behavioural signature]) so the combiner
prompt can reason about which skill fits which driver. This module assembles both
from:

* the three handwritten seeds (always present -- they pin the Pareto extremes);
* every ``<name>.py`` + ``<name>.meta.json`` under
  ``pref_dispatch/evolved/skills/`` produced by Phase-1 QD discovery (§4).

Evolved skill modules are import-safe and sandbox-clean by construction (they were
compiled through the sandbox before freezing), so they are loaded with a normal
``importlib`` from their frozen path.

Feature-3 reposition scorers are a SEPARATE artefact (never a method on a skill);
they are loaded by :func:`load_repositioner` from
``pref_dispatch/evolved/repositioners/`` and recompiled through the sandbox, the
same shape as :func:`load_frozen_combiner`.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pref_dispatch.skills import EnRouteSkill, RevenueSkill, ServiceSkill, Skill

EvolvedSkillsDir = os.path.join("pref_dispatch", "evolved", "skills")

# Cards for the handwritten seeds (objective/description the combiner reads).
_SEED_CARDS = {
    "revenue": {
        "objective": "maximise platform revenue (served trip-minutes) by favouring "
                     "long fares even at some pickup cost",
        "description": "Scores an order by its solo trip time; sends empty cars far "
                       "for lucrative fares. Revenue high, service time high.",
    },
    "service": {
        "objective": "minimise passenger service time (short pickup + ride); wait "
                     "rather than take a far/long order",
        "description": "Prefers nearby short trips, else idles. Service time low, "
                       "revenue low, often no-op.",
    },
    "enroute": {
        "objective": "protect committed onboard orders near their deadline; suppress "
                     "detour",
        "description": "When an onboard order has little slack, only tiny-detour "
                       "additions are accepted. Detour low, empty/no-op rate high.",
    },
}


@dataclass
class LoadedSkill:
    skill: Skill
    card: Dict[str, str]  # skill_name, objective, description [, signature_text]


def _load_evolved_module(py_path: str, name: str) -> Skill:
    """Import a frozen evolved skill module and wrap its score/noop_score."""
    spec = importlib.util.spec_from_file_location(f"evolved_skill_{name}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # frozen modules are sandbox-clean by construction

    score_fn = getattr(mod, "score")
    noop_fn = getattr(mod, "noop_score", None)

    class _FrozenSkill(Skill):
        pass

    sk = _FrozenSkill()
    sk.name = name
    sk.score = score_fn  # type: ignore[assignment]
    if callable(noop_fn):
        sk.noop_score = noop_fn  # type: ignore[assignment]
    # A worker process cannot receive this object: the wrapper class is local and
    # the functions live in a module built at runtime, so pickle can name neither.
    # It CAN reload the same file, so the path travels instead of the object --
    # see :func:`pref_dispatch.llm.parallel.skills_payload`. (Recompiling the
    # source through the sandbox is not an option here: a frozen module carries
    # its own imports, which the restricted exec deliberately forbids.)
    sk.source_path = os.path.abspath(py_path)  # type: ignore[attr-defined]
    return sk


def load_basis(
    include_evolved: bool = True,
    evolved_dir: str = EvolvedSkillsDir,
    extra_skill_dirs: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Skill], List[Dict]]:
    """Return ``({name: Skill}, [card, ...])`` for the frozen basis.

    Always includes the three handwritten seeds. When ``include_evolved`` and the
    evolved directory has frozen skills, those are appended (a later evolved skill
    with the same name as a seed overrides it, so a QD-improved specialist wins).

    ``extra_skill_dirs`` (§Phase-3 fine-tune): additional frozen-skill directories
    layered ON TOP of the generalist basis, in order. A fine-tuned skill with the
    same name as a generalist one OVERRIDES it -- so a scenario-specialised skill
    wins for that fine-tuned point, while the generalist basis on disk is untouched.
    """
    skills: Dict[str, Skill] = {
        "revenue": RevenueSkill(),
        "service": ServiceSkill(),
        "enroute": EnRouteSkill(),
    }
    cards: Dict[str, Dict] = {
        n: {"skill_name": n, **_SEED_CARDS[n]} for n in skills
    }

    scan_dirs: List[str] = []
    if include_evolved:
        scan_dirs.append(evolved_dir)
    scan_dirs.extend(extra_skill_dirs or [])

    for d in scan_dirs:
        if not os.path.isdir(d):
            continue
        for meta_path in sorted(glob.glob(os.path.join(d, "*.meta.json"))):
            name = os.path.basename(meta_path)[: -len(".meta.json")]
            py_path = os.path.join(d, f"{name}.py")
            if not os.path.exists(py_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            skills[name] = _load_evolved_module(py_path, name)
            card = {
                "skill_name": name,
                "objective": meta.get("objective", ""),
                "description": meta.get("description", ""),
            }
            cards[name] = card

    return skills, [cards[n] for n in skills]


EvolvedCombinersDir = os.path.join("pref_dispatch", "evolved", "combiners")


def load_frozen_combiner(
    name: Optional[str] = None,
    skill_names: Optional[Sequence[str]] = None,
    combiners_dir: str = EvolvedCombinersDir,
):
    """Load a frozen Phase-2 combiner as an :class:`LLMCombiner`.

    ``name`` selects ``<name>.py`` under ``combiners_dir``; if ``None`` the
    single frozen combiner is used (error if zero or many exist). ``skill_names``
    restricts the argmax to the frozen basis (defaults to the combiner meta's
    recorded ``skill_names``). Returns ``(LLMCombiner, meta)`` or raises.
    """
    from pref_dispatch.llm.combiner_adapter import LLMCombiner
    from pref_dispatch.llm.sandbox import compile_combiner

    if name is None:
        metas = sorted(glob.glob(os.path.join(combiners_dir, "*.meta.json")))
        if len(metas) != 1:
            raise FileNotFoundError(
                f"expected exactly one frozen combiner in {combiners_dir!r}, "
                f"found {len(metas)}; pass name= to disambiguate."
            )
        name = os.path.basename(metas[0])[: -len(".meta.json")]

    py_path = os.path.join(combiners_dir, f"{name}.py")
    meta_path = os.path.join(combiners_dir, f"{name}.meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(py_path, encoding="utf-8") as f:
        src = f.read()
    # The frozen module is `docstring + imports + def skill_scores...`; recompile
    # just the function body through the sandbox (keeps the same safety contract).
    body = src[src.index("def skill_scores"):]
    scorer = compile_combiner(body)
    names = tuple(skill_names or meta.get("skill_names") or ())
    if not names:
        raise ValueError("frozen combiner meta has no skill_names; pass skill_names=.")
    return LLMCombiner(scorer, names), meta


RepositionersDir = os.path.join("pref_dispatch", "evolved", "repositioners")


def load_repositioner(
    name: Optional[str] = None,
    repositioners_dir: str = RepositionersDir,
):
    """Load a frozen Feature-3 reposition scorer as a bound ``reposition_scores``.

    ``name`` selects ``<name>.py`` under ``repositioners_dir``; if ``None`` the
    single frozen scorer is used (error if zero or many exist). Returns the bound
    callable to drop into ``Repositioner(scores_fn=...)`` -- exactly the shape of
    :func:`load_frozen_combiner` (recompile the frozen function body through the
    sandbox, keeping the same safety contract).
    """
    from pref_dispatch.llm.sandbox import compile_repositioner

    if name is None:
        metas = sorted(glob.glob(os.path.join(repositioners_dir, "*.meta.json")))
        if len(metas) != 1:
            raise FileNotFoundError(
                f"expected exactly one frozen repositioner in {repositioners_dir!r}, "
                f"found {len(metas)}; pass name= to disambiguate."
            )
        name = os.path.basename(metas[0])[: -len(".meta.json")]

    py_path = os.path.join(repositioners_dir, f"{name}.py")
    meta_path = os.path.join(repositioners_dir, f"{name}.meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(py_path, encoding="utf-8") as f:
        src = f.read()
    # The frozen module is `docstring + imports + def reposition_scores...`;
    # recompile just the function body through the sandbox.
    body = src[src.index("def reposition_scores"):]
    scorer = compile_repositioner(body, name=name)
    return scorer.reposition_scores, meta

