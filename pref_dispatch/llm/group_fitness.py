"""GRPO group-relative fitness, shared by Phase 1, Phase 2 and Phase 3.

Every phase scores a candidate the same way: roll the whole round's population on
the SAME cell (Phase 1: one scenario; Phase 2/3: one (scenario, objective) pair),
then express each candidate's episode result as its **standardised advantage
inside that cell's group**::

    A = (r - mean(group)) / std(group)

and average the advantages over the round's cells. The group is the round's own
population, so nothing outside the round enters the number.

Why standardise instead of rescaling against a fixed ruler
----------------------------------------------------------
Raw episode results are not comparable across cells. A full 17:00 hour carries
~3.5k orders and a 09:00 hour ~1.2k, so a candidate that is mediocre in the busy
hour still out-earns a candidate that is excellent in the quiet one; averaging raw
numbers is therefore just a weighted vote for whichever cells happen to be big.
The pre-v6 fix was a per-cell min/max ruler built from the single skills, and it
failed for two measured reasons: (1) an objective's INTERNAL scale is arbitrary --
doubling a weight doubles the numbers without changing the problem -- and (2) any
mixture of skills sits above the single-skill band, so nearly every candidate
saturated the ruler near 1.0 and the arbitrary gen-0 seed never lost a generation.

Standardising inside the round's own group removes both: the mean absorbs the
cell's scale and the std absorbs its spread, so ``A`` answers only "how far from
this round's average was I, in units of how spread out this round was".

Why the advantage and not the percentile rank (2026-08-10)
----------------------------------------------------------
v6/v7 used the percentile rank of ``r`` within the group. Ranks are scale-free,
which fixed the ruler problem, but they throw away the MARGIN: finishing second by
a hair and finishing second by a mile are both 0.86, so a candidate that nearly
matched the leader everywhere scored the same as one that was distantly second
everywhere, and a breakthrough on one cell was capped at "+1 place". The
standardised advantage keeps the margin, which is what makes the selection key
discriminating -- and it is the ordinary GRPO estimator, so the training signal is
the one the method is named after.

Three practical rules baked in here
-----------------------------------
* **The group INCLUDES the candidate itself.** That is the GRPO definition, and it
  is what makes the advantages of a cell sum to zero. (The v6/v7 rank excluded the
  candidate's own rollout, which is the right convention for a rank and the wrong
  one for a z-score.)
* **Advantages are clipped to +/-** :data:`Z_CLIP`. One catastrophic member --
  a skill that scores -1e9 because it refuses every order -- would otherwise
  inflate ``std`` on its own and squash the whole rest of the round toward 0,
  costing the round its entire signal. Ranks were immune to this; z-scores are not.
* **A dead cell scores 0, not "average".** When ``std == 0`` every member produced
  the same result, so nobody distinguished themselves and everyone gets 0.0. Under
  the old rank this printed as 0.50, which reads as "middle of the field" and is
  indistinguishable from a genuinely competitive cell -- the v6 run was degenerate
  for five straight generations behind exactly that number.

Scale note: only ONE of the two constants moves
-----------------------------------------------
Advantages span roughly ``[-2, +2]`` where ranks spanned ``[0, 1]``, i.e. about 4x
wider. That matters for a constant multiplying a UNITLESS quantity and not for one
multiplying a quantity already measured in the same units as the fitness:

* :data:`DEFAULT_FALLBACK_PENALTY` **is scaled 4x (0.5 -> 2.0)**. It multiplies the
  fallback RATE, a fraction in [0, 1], and subtracts the product from a fitness
  that just got 4x wider. Left at 0.5, a program that fell back on every single
  decision would lose a quarter of a standard deviation -- nothing.
* :data:`DEFAULT_FAMILY_BETA` **is 0 (2026-08-13)**. It used to trade mean
  advantage against weakest-family advantage at 0.15. The user requirement is that
  the FINAL selection key be the pure GRPO fitness with NO hand-set constant --
  survivors are the top-``mu`` by plain mean advantage, plus one reserved slot per
  objective family so the best specialist on a hard family survives as crossover
  material. beta = 0 makes ``selection_score`` exactly the mean advantage.
"""

from __future__ import annotations

import math
from typing import List, Sequence

# Advantages are clipped to this many standard deviations (see module docstring).
Z_CLIP = 3.0

# Selection constants. Only the first is on the advantage scale -- see the
# "Scale note" in the module docstring for why the second must NOT be rescaled.
DEFAULT_FALLBACK_PENALTY = 2.0
# 0 = pure GRPO mean advantage as the selection key (user requirement 2026-08-13).
# The reserved per-family elite slots in select_survivors keep hard-family
# specialists alive as crossover material WITHOUT any constant in the key.
DEFAULT_FAMILY_BETA = 0.0


def _finite(values: Sequence[float]) -> List[float]:
    """The finite members of ``values`` (nan/inf cannot enter a mean or a std)."""
    return [float(x) for x in values if isinstance(x, (int, float))
            and not isinstance(x, bool) and math.isfinite(float(x))]


def group_advantage(value: float, group: Sequence[float], *, clip: float = Z_CLIP) -> float:
    """Standardised advantage of ``value`` inside ``group``.

    ``group`` is the cell's FULL population and must therefore contain ``value``
    itself (see the module docstring). Returns ``(value - mean) / std`` clipped to
    ``+/-clip``; ``0.0`` when the group is empty or every member scored the same,
    and ``-clip`` when ``value`` is not finite (a broken candidate is the worst
    thing in the cell, not an undefined one).
    """
    vals = _finite(group)
    n = len(vals)
    if n == 0:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -clip
    if not math.isfinite(v):
        return -clip
    mean = sum(vals) / n
    # Population std (not sample): the group IS the population, there is nothing
    # being estimated from a sample of it.
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / n)
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    return max(-clip, min(clip, (v - mean) / std))


def delta_advantage(value: float, group: Sequence[float], *,
                    clip: float = Z_CLIP) -> float:
    """Standardised advantage of a DELTA, measured against a fixed baseline.

    Same shape as :func:`group_advantage` with the mean subtraction removed.
    Callers pass differences that are already anchored -- for Phase 3, ``value``
    is ``my reward - the reward of doing nothing`` on one cell and ``group`` is
    every candidate's version of that same difference. Returns
    ``value / std(group)`` clipped to ``+/-clip``.

    Why no mean subtraction. ``group_advantage`` answers "how do I compare to the
    others in this round", so its sign is relative: land in a round of weak
    programs and mediocrity scores well. Dropping the centring makes the sign
    ABSOLUTE -- positive iff you actually beat the baseline -- while the division
    keeps the cell scale-free, which is the whole point of the group treatment.
    The candidate and the baseline are rolled on the SAME seed, so the difference
    is paired and the shared scenario noise cancels before it reaches the std.

    Degenerate columns. If every candidate produced the same delta the std is 0,
    and centring is not there to rescue it: "everyone beat doing nothing by the
    same margin" must not collapse to 0, which under this fitness would read as
    "did nothing". The denominator is therefore floored at ``|mean| / clip``, so
    a unanimous column saturates at exactly ``+/-clip`` instead. That floor
    introduces no new constant -- it is the clip. A column where every delta is
    genuinely 0 still scores 0, which is correct: nobody moved the needle.

    ``-clip`` for a non-finite ``value`` (a crashed rollout is the worst thing in
    the cell, not an undefined one) and ``0.0`` for an empty group.
    """
    vals = _finite(group)
    n = len(vals)
    if n == 0:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return -clip
    if not math.isfinite(v):
        return -clip
    mean = sum(vals) / n
    # Population std, for the same reason as group_advantage: the group IS the
    # population. Note it is the spread ABOUT THE MEAN, not about zero -- it
    # measures how much the candidates disagree about the effect, which is the
    # natural unit for "how big is my effect".
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / n)
    denom = max(std, abs(mean) / clip)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return max(-clip, min(clip, v / denom))


def group_advantages(values: Sequence[float], *, clip: float = Z_CLIP) -> List[float]:
    """Advantage of every member of ``values`` against the whole of ``values``.

    Convenience for the common case where the cell's group is exactly the list
    being scored. Equivalent to ``[group_advantage(v, values) for v in values]``
    but computes the mean and std once.
    """
    vals = _finite(values)
    n = len(vals)
    if n == 0:
        return [0.0 for _ in values]
    mean = sum(vals) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / n)
    if std <= 0.0 or not math.isfinite(std):
        return [0.0 for _ in values]
    out: List[float] = []
    for x in values:
        try:
            v = float(x)
        except (TypeError, ValueError):
            out.append(-clip)
            continue
        out.append(-clip if not math.isfinite(v)
                   else max(-clip, min(clip, (v - mean) / std)))
    return out


def tie_rate(value: float, peers: Sequence[float]) -> float:
    """Share of ``peers`` (the group MINUS the candidate) that scored identically.

    The degeneracy read-out the prompts consume: 1.0 means the rest of the round
    produced the exact same episode result, so nothing in this cell distinguished
    anybody. Unlike the advantage itself this is deliberately measured against the
    peers only -- "how many OTHERS matched me".
    """
    peers = list(peers)
    if not peers:
        return 0.0
    return sum(1.0 for x in peers if x == value) / len(peers)
