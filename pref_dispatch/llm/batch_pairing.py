"""How a round's scenes and objectives are paired, shared by Phases 1, 2 and 3.

Each training round draws a batch of real one-hour demand windows (stratified over
fleet bands) and, independently, a batch of objectives. Independently is the
problem: a whole objective family can land inside one fleet band by luck and the
candidate then learns that band's rule as if it were universal. This module holds
the pairing that closes that hole, plus the fleet bands the three phases share, so
the rule lives in ONE place instead of being re-typed per phase.

Lifted out of ``run_phase2_full`` on 2026-08-10 (unchanged behaviour) when Phase 1
and Phase 3 were rebuilt on the same rotating-batch design. Phase 1 has no
objective families to pair, so it uses the other half of this module,
:class:`BandedWindowSampler`, which stratifies the SCENES across the same bands.
Phase 3 has one axis more -- the fairness strength -- and uses
:func:`pair_by_strength_band` on top of the objective pairing, for the same reason
and by the same rule.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from pref_dispatch.llm.objective_sampler import _STRUCTURAL_FAMILIES

# The fleet bands every phase stratifies its scene batch over: scarce, mid, large.
# The gate's hard cells live in the first band (150-500 cars serving a full real
# hour) and the deployment fleet in the last.
DEFAULT_FLEET_BANDS: List[Tuple[int, int]] = [(200, 500), (500, 1000), (1000, 1500)]


def band_index(scenario, bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS) -> int:
    """Which fleet band ``scenario`` falls in; the NEAREST band when it fits none.

    A scene outside every band (the sampler's envelope is wider than the bands)
    still has to belong somewhere, or it would silently drop out of whatever is
    being stratified. Nearest-by-midpoint keeps it in the band it most resembles.
    """
    for bi, (lo, hi) in enumerate(bands):
        if lo <= scenario.num_drivers <= hi:
            return bi
    return min(range(len(bands)),
               key=lambda bi: abs(scenario.num_drivers - sum(bands[bi]) / 2.0))


def band_label(scenario, bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS) -> str:
    """Readable band name, e.g. ``fleet200-500``.

    Phase 2 groups a candidate's per-cell advantages by OBJECTIVE FAMILY, so a
    program that is strong on average but hopeless on one family is caught. Phase 1
    trains one skill under ONE fixed objective, so it has no families -- its
    equivalent axis is SCALE, and this is the key it groups by. Same purpose: a
    skill that only works at fleet 1200 must not be able to hide behind its mean.
    """
    lo, hi = bands[band_index(scenario, bands)]
    return f"fleet{lo}-{hi}"


def pair_by_fleet_band(scenarios: Sequence[object], objectives: Sequence[object],
                       bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS) -> list:
    """Reorder ``objectives`` so each FAMILY's draws land in DIFFERENT fleet bands.

    The scene batch is already stratified across fleet bands, but the objective
    draw is independent -- so a family can still land entirely inside one band by
    luck. That is exactly how the 2026-08-09 v2 run trained ``completion`` on a
    single fleet-938 scene and then lost all three scarce-fleet completion cells
    in the gate (reading ``w`` was worth -267..-593 reward there and +221..+557 at
    fleet 1000+): the champion learned the at-scale rule as if it were universal.

    Rule, in three passes over the families that have >= 2 draws (structural
    families FIRST -- completion / pooling are the gate's hard ones --
    then the rest by descending draw count, ties alphabetical for determinism):

      1. give each such family one draw in the SCARCEST band;
      2. give each such family one draw in the LARGEST band;
      3. place every remaining draw in whichever free band that family occupies
         least, so extra draws widen coverage instead of piling up.

    That makes "seen at both scarcity and scale" a property of the ALGORITHM, not
    of the seed. An earlier version rotated a band offset by family index, which
    left WHICH family reached the scarce band up to the alphabet: the 2026-08-09
    v3 grid put completion on fleet 938 and 1137 -- both mid/large, i.e. exactly
    the hole it was written to close.

    Returns a new objectives list of the same length; ``scenarios`` is left
    untouched, so the fixed (scenario x objective) grid contract is unchanged --
    only WHICH objective sits on which scene moves.
    """
    n = len(scenarios)
    if n != len(objectives):
        raise ValueError("scenarios and objectives must be the same length")
    if not bands:
        return list(objectives)

    def _band_of(sc) -> int:
        return band_index(sc, bands)

    # Free scenario slots per band; scarcest and largest band by fleet size.
    free = {bi: [] for bi in range(len(bands))}
    for i, sc in enumerate(scenarios):
        free[_band_of(sc)].append(i)
    by_size = sorted(range(len(bands)), key=lambda bi: sum(bands[bi]))
    scarce, largest = by_size[0], by_size[-1]

    # Draws per family, and the CLAIM ORDER: the gate's hard structural families
    # first (so they cannot be crowded out of the scarce band by a wide raw
    # family), then everything else widest-first, ties alphabetical for determinism.
    by_family = {}
    for j, ob in enumerate(objectives):
        by_family.setdefault(getattr(ob, "family", "?"), []).append(j)
    ordered = [f for f in _STRUCTURAL_FAMILIES if f in by_family]
    ordered += sorted((f for f in by_family if f not in ordered),
                      key=lambda f: (-len(by_family[f]), f))

    placed = [None] * n
    pending = {f: list(js) for f, js in by_family.items()}
    hits = {f: {bi: 0 for bi in range(len(bands))} for f in by_family}

    def _claim(fam: str, bi: int) -> bool:
        """Place one of ``fam``'s unplaced draws on a free slot in band ``bi``."""
        if not pending[fam] or not free[bi]:
            return False
        placed[free[bi].pop(0)] = objectives[pending[fam].pop(0)]
        hits[fam][bi] += 1
        return True

    # Passes 1 and 2: every family with >= 2 draws gets scarcity AND scale.
    for bi in ([scarce] if scarce == largest else [scarce, largest]):
        for fam in ordered:
            if len(by_family[fam]) >= 2:
                _claim(fam, bi)

    # Pass 3: each remaining draw goes where its own family is thinnest (ties: the
    # band with the most room left, then lowest index -- all deterministic), so
    # extra draws widen a family's coverage instead of piling onto one scale.
    for fam in ordered:
        while pending[fam]:
            options = [bi for bi in range(len(bands)) if free[bi]]
            if not options:
                break
            bi = min(options, key=lambda b: (hits[fam][b], -len(free[b]), b))
            _claim(fam, bi)

    # Safety net: if the bands ran dry (a scene outside every band, so some slot
    # was never listed), place the rest anywhere free -- length must be preserved.
    for j, i in zip([j for fam in ordered for j in pending[fam]],
                    [i for i in range(n) if placed[i] is None]):
        placed[i] = objectives[j]
    return placed


# How a fairness-strength BAND is turned back into a number to train on. The band
# names and their cut points are ``reposition_eval.STRENGTH_BANDS`` /
# ``strength_label`` -- that module is the authority; these are only the intervals
# a round DRAWS from, kept here so this module does not have to import the whole
# rollout stack. "strong" starts just above 1.0 so a draw cannot land on the mild
# side of the cut. The deployment range we actually test is 0-1; "strong" reaches
# past it on purpose, because a scorer that has only ever seen the budget nudge
# close calls has never seen it override a large score gap.
STRENGTH_BAND_DRAWS = {"off": (0.0, 0.0), "mild": (0.05, 1.0), "strong": (1.05, 2.5)}

_STRENGTH_LABELS = ("off", "mild", "strong")


def pair_by_strength_band(
    scenarios: Sequence[object],
    objectives: Sequence[object],
    rng,
    bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
    draws=STRENGTH_BAND_DRAWS,
) -> List[float]:
    """The fairness strength for each cell, spread over families AND fleet bands.

    Phase 3's cell is ``(scenario, objective, fairness strength)`` -- one more axis
    than Phase 2. Drawing the strength independently per cell re-opens, on the new
    axis, exactly the hole :func:`pair_by_fleet_band` closes on the old one: a whole
    objective family can end up graded only with the budget OFF, and the round then
    cannot tell "works everywhere" apart from "works while the fairness knob is at
    zero" -- which is the single most common way a region scorer fails.

    Two things are spread, in this order:

      1. **Within a family.** Every family with >= 2 cells gets one ``off`` cell and
         one cell with the budget ON (alternating ``mild`` / ``strong`` down the
         claim order, so both live bands appear across the batch). That is the
         minimum needed for the per-family advantage to contain the comparison at
         all.
      2. **Across fleet bands.** When several of a family's cells could take a band,
         it goes to the one whose FLEET band has seen that strength band least. Both
         axes have three levels, so a naive round-robin would marry "strong" to
         "fleet 1000-1500" permanently and the scorer would learn the budget only in
         the regime where cars are plentiful.

    Leftover cells go to whichever band their own family has least of (ties: the
    band least used in the whole batch, then a fixed order) -- so extra cells widen
    coverage rather than piling onto one strength.

    ``rng`` is the round's sampler RNG; only the value WITHIN a band is random, the
    band assignment is deterministic given the batch. Returns a list of floats
    aligned with ``scenarios`` / ``objectives``; both are left untouched, so the
    fixed grid contract is unchanged.
    """
    n = len(scenarios)
    if n != len(objectives):
        raise ValueError("scenarios and objectives must be the same length")

    fleet_of = [band_index(sc, bands) for sc in scenarios]
    by_family: Dict[str, List[int]] = {}
    for i, ob in enumerate(objectives):
        by_family.setdefault(getattr(ob, "family", "?"), []).append(i)
    ordered = [f for f in _STRUCTURAL_FAMILIES if f in by_family]
    ordered += sorted((f for f in by_family if f not in ordered),
                      key=lambda f: (-len(by_family[f]), f))

    assigned: List[Optional[str]] = [None] * n
    seen = {lab: {bi: 0 for bi in range(len(bands))} for lab in _STRENGTH_LABELS}
    fam_hits = {f: {lab: 0 for lab in _STRENGTH_LABELS} for f in by_family}

    def _claim(fam: str, lab: str) -> bool:
        """Put band ``lab`` on one of ``fam``'s free cells, in the fleet band that
        has seen ``lab`` least (ties: lowest cell index, for determinism)."""
        free = [i for i in by_family[fam] if assigned[i] is None]
        if not free:
            return False
        i = min(free, key=lambda c: (seen[lab][fleet_of[c]], c))
        assigned[i] = lab
        seen[lab][fleet_of[i]] += 1
        fam_hits[fam][lab] += 1
        return True

    wide = [f for f in ordered if len(by_family[f]) >= 2]
    for fam in wide:                                    # pass 1: budget OFF
        _claim(fam, "off")
    for k, fam in enumerate(wide):                      # pass 2: budget ON
        _claim(fam, "mild" if k % 2 == 0 else "strong")
    for fam in ordered:                                 # pass 3: the remainder
        while any(assigned[i] is None for i in by_family[fam]):
            lab = min(_STRENGTH_LABELS,
                      key=lambda l: (fam_hits[fam][l], sum(seen[l].values()),
                                     _STRENGTH_LABELS.index(l)))
            if not _claim(fam, lab):
                break

    out: List[float] = []
    for lab in assigned:
        lo, hi = draws[lab or "off"]
        out.append(float(lo) if hi <= lo else float(rng.uniform(lo, hi)))
    return out


class BandedWindowSampler:
    """A :class:`~pref_dispatch.scenario.ScenarioSampler` that spans the bands.

    Phase 1 trains ONE skill under ONE objective, so it has no objective families
    to spread and :func:`pair_by_fleet_band` has nothing to do. Its equivalent
    hole is the fleet band: ``evolve_skill_group`` selects on
    ``fitness + beta * min(per_band)``, and if a round's batch happens to sit
    inside one band that term is just the mean again -- a skill that only works
    at fleet 1200 wins the round unopposed. So Phase 1 stratifies the SCENES
    themselves: slot ``i`` gets band ``i % len(bands)``, which makes "measured at
    scarcity AND at scale" a property of the batch rather than of the seed.

    The windows are real full hours (:meth:`ScenarioSampler.sample_real_windows`),
    matching what Phase 2 trains and what the gate scores; only the fleet size is
    then redrawn inside the slot's band. The band order is rotated once per batch
    from the sampler's own RNG so a band is not permanently married to window 0 --
    the batch is drawn ONCE per round and shared by every candidate, so this stays
    reproducible and identical across the whole comparison grid.

    It exposes only ``sample_batch(k, base_seed=...)``, which is the entire
    sampler API Phase 1 uses -- so it drops into ``run_phase1`` / ``discover_basis``
    with no plumbing.
    """

    def __init__(
        self,
        sampler,
        *,
        bands: Sequence[Tuple[int, int]] = DEFAULT_FLEET_BANDS,
        min_high_volume: int = 1,
        ranges=None,
    ) -> None:
        self.sampler = sampler
        self.bands = list(bands)
        self.min_high_volume = int(min_high_volume)
        self.ranges = ranges if ranges is not None else sampler.ranges
        # Phase 1 reads these off the sampler it was handed; keep them visible.
        self.split = getattr(sampler, "split", "train")
        self.rng = getattr(sampler, "rng", None)

    def sample_batch(self, k: int, *, base_seed: Optional[int] = None) -> List:
        scs = self.sampler.sample_real_windows(
            k, base_seed=base_seed,
            min_high_volume=min(self.min_high_volume, k),
            ranges=self.ranges,
        )
        if not self.bands:
            return scs
        off = self.sampler.rng.randrange(len(self.bands)) if self.rng else 0
        out = []
        for i, sc in enumerate(scs):
            lo, hi = self.bands[(i + off) % len(self.bands)]
            fleet = self.sampler._sample_fleet(
                dataclasses.replace(self.ranges, fleet=(float(lo), float(hi)))
            )
            out.append(dataclasses.replace(sc, num_drivers=int(fleet)))
        self._log_band_coverage(out)
        return out

    def _log_band_coverage(self, scs: Sequence[object]) -> None:
        """Software scene-coverage note for a Phase-1 batch (advisory, log-only).

        Phase 1 has no objective families; its generality axis is the SCENE. The
        BandedWindowSampler already forces one fleet band per slot, so this only
        reports what actually came back (bands / regimes / distinct windows) so a
        run is auditable -- a round that drew 3 windows and no peak hour is visible
        rather than silent. Nothing is rejected and nothing is sampled here; this is
        the software counterpart of the LLM scene-diversity audit, and it cannot
        hang the search.
        """
        try:
            from pref_dispatch.llm.batch_check import scene_coverage_report
            rep = scene_coverage_report(scs, self.bands)
        except Exception:  # noqa: BLE001 -- advisory; never fatal
            return
        print(f"[phase1] scene batch coverage: bands={rep['bands']} "
              f"regimes={rep['regimes']} windows={rep['n_distinct_windows']} "
              f"(ok={rep['ok']})")
