"""Encode the *dynamic* environment into a natural-language profile for the LLM.

This is the most important prompt-engineering step in Phase 1 (paradigm B): the
LLM never sees raw per-step state (1000 drivers x coordinates would not fit and
is not needed). It writes a *frozen* function that generalises to every step, so
what it needs is the **semantics, units, and aggregate distribution** of the
environment -- not one step's exact numbers. We therefore emit three aggregate
profiles (MAP / ORDER / DRIVER), each as "dimensions + structure note + one
format example", mirroring the mod-opt OD-matrix prompt (``llm_design.md`` §3).

Key design points (all from §3):

* **region <-> coordinate binding.** A coordinate is continuous ``(lon, lat)``;
  a region is the discrete cell it falls in (``env.region_neighbours`` adjacency,
  ``self["current_region"]`` index). The frozen function's hot path only ever
  sees coordinates + ``dist()``; region is an *interpretive* aid. We additionally
  attach real Manhattan **taxi-zone names** (Financial District, Midtown, ...)
  as geographic anchors so the LLM's spatial prior can grab onto the coordinate
  system -- this is exactly the "use an LLM, not random search" lever.
* **ORDER profile comes from the PREVIOUS hour**, never the current window: at
  decision time the current hour's demand has not happened yet, so profiling it
  would be looking at the future (data leakage). §3.2.
* **DRIVER profile** is honest that the sim opens with all cars empty, but keeps
  the full ``self`` field list (en-route skills need it even when empty at t=0).

The public entry point is :func:`encode_env_profile`. A tiny CLI
(``python -m pref_dispatch.llm.encode --regime peak``) prints the profile for a
real Manhattan window so a human can read it before any LLM call (§3.6).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pref_dispatch.global_stats import GlobalStats, _nearest_region, od_matrix
from pref_dispatch.nyc_env import TIME_OF_DAY, _window_for_hour
from pref_dispatch.scenario import ScenarioRanges, simulated_clock

Coord = Tuple[float, float]
Dist = Callable[[Coord, Coord], float]

# Repo-relative taxi-zone centroid table (LocationID, zone, borough, lon, lat).
_ZONE_CSV = os.path.join("data", "nyc", "zone_centroids.csv")

# Well-known Manhattan anchor zones to spell out explicitly in the MAP profile.
# These names carry strong geographic priors an LLM already knows; they let it
# reason about "downtown vs uptown", airports, etc. purely from coordinates.
_ANCHOR_ZONES = (
    "Financial District North",  # lower Manhattan / downtown
    "TriBeCa/Civic Center",
    "Times Sq/Theatre District",  # midtown
    "Upper East Side North",  # uptown
    "Washington Heights South",  # far north
)


# --------------------------------------------------------------------------- #
# Zone table (geographic anchors -- interpretive only, never in the hot path). #
# --------------------------------------------------------------------------- #
def _load_zones(borough: str = "Manhattan") -> List[Tuple[str, Coord]]:
    """Load ``[(zone_name, (lon, lat)), ...]`` for one borough from the CSV.

    Returns ``[]`` (encoder degrades gracefully) if the file is absent.
    """
    if not os.path.exists(_ZONE_CSV):
        return []
    out: List[Tuple[str, Coord]] = []
    with open(_ZONE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if borough and row.get("borough") != borough:
                continue
            try:
                out.append((row["zone"], (float(row["lon"]), float(row["lat"]))))
            except (KeyError, ValueError):
                continue
    return out


def _nearest_zone(coord: Coord, zones: Sequence[Tuple[str, Coord]]) -> str:
    """Name of the nearest zone centroid (Euclidean on lon/lat -- fine for an
    interpretive label; the frozen function never uses this)."""
    if not zones:
        return "?"
    lon, lat = coord
    best, best_d = "?", float("inf")
    for name, (zlon, zlat) in zones:
        d = (lon - zlon) ** 2 + (lat - zlat) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def _anchor_lines(zones: Sequence[Tuple[str, Coord]]) -> str:
    """Format the explicit ``zone -> centroid`` anchor lines for the MAP block."""
    by_name = {name: coord for name, coord in zones}
    lines = []
    for name in _ANCHOR_ZONES:
        c = by_name.get(name)
        if c is not None:
            lines.append(f"    {name}: (lon={c[0]:.4f}, lat={c[1]:.4f})")
    if not lines:  # zone table unavailable -> honest fallback
        return "    (taxi-zone name table unavailable; reason in raw coordinates)"
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Previous-hour order distribution (avoids current-window leakage, §3.2).      #
# --------------------------------------------------------------------------- #
def _prev_hour(regime: str, back: int = 1) -> str:
    """The clock hour ``back`` hours before this regime's window (peak 18, back=2 -> 16)."""
    hour = int(TIME_OF_DAY[regime])
    return f"{hour - back:02d}"


def load_prev_window_orders(regime: str, split: str, back: int = 1) -> List[Dict]:
    """Load the order window ``back`` hours before this regime as lightweight dicts.

    Each dict: ``origin, destination (lon,lat tuples), num_passengers``. Requires
    pandas (only used offline for profiling, never in the frozen hot path).
    Returns ``[]`` if that window is absent from the split (encoder degrades).
    """
    import pandas as pd  # local import: heavy, offline-only

    try:
        path = _window_for_hour(_prev_hour(regime, back), split)
    except ValueError:
        return []
    df = pd.read_parquet(path)
    return [
        {
            "origin": (float(r.origin_x), float(r.origin_y)),
            "destination": (float(r.dest_x), float(r.dest_y)),
            "num_passengers": int(r.num_passengers),
        }
        for r in df.itertuples(index=False)
    ]


def load_prev_hour_orders(regime: str, split: str) -> List[Dict]:
    """Back-compat alias: the immediately-previous hour's orders (``back=1``)."""
    return load_prev_window_orders(regime, split, back=1)


def _quantiles(values: Sequence[float], qs=(0.1, 0.5, 0.9)) -> Dict[float, float]:
    """Simple nearest-rank quantiles (no numpy dependency needed here)."""
    if not values:
        return {q: float("nan") for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        out[q] = s[idx]
    return out


def _party_distribution(orders: Sequence[Dict]) -> Dict[int, float]:
    """Fraction of orders by party size, sorted by party size."""
    if not orders:
        return {}
    counts: Dict[int, int] = {}
    for o in orders:
        p = int(o["num_passengers"])
        counts[p] = counts.get(p, 0) + 1
    n = len(orders)
    return {p: counts[p] / n for p in sorted(counts)}


def _top_origin_zones(
    orders: Sequence[Dict], zones: Sequence[Tuple[str, Coord]], k: int = 5
) -> List[Tuple[str, float]]:
    """Top-k origin zones (by nearest-centroid) and their share of demand."""
    if not orders or not zones:
        return []
    counts: Dict[str, int] = {}
    for o in orders:
        z = _nearest_zone(o["origin"], zones)
        counts[z] = counts.get(z, 0) + 1
    n = len(orders)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(z, c / n) for z, c in ranked]


def _region_of(point: Coord, centres: Sequence[Coord]) -> int:
    """Region index of ``point`` by the ONE partition rule (-1 with no layout).

    Delegates to :func:`pref_dispatch.global_stats._nearest_region`, the same
    function the runtime uses, so a printed index can never disagree with the
    index the frozen program will see."""
    if not centres:
        return -1
    import numpy as np
    return int(_nearest_region(
        np.asarray([point], dtype=float), np.asarray(centres, dtype=float)
    )[0])


def _top_od_pairs(
    orders: Sequence[Dict],
    centres: Sequence[Coord],
    zones: Sequence[Tuple[str, Coord]],
    k: int = 5,
) -> List[Tuple[int, int, float]]:
    """Top-k ``(origin_region, destination_region, share)`` flows in ``orders``.

    Bucketed by :func:`pref_dispatch.global_stats.od_matrix` -- the SAME call the
    runtime uses to build ``phi_ep.od_count`` -- so the region indices printed
    here are exactly the indices the frozen function will index that matrix with.
    Recomputing the bucketing locally would let the prompt and the runtime drift.
    """
    if not orders or not centres:
        return []
    od_count, _out, _in, _n = od_matrix(orders, tuple(tuple(c) for c in centres))
    flows = [
        (i, j, share)
        for i, row in enumerate(od_count)
        for j, share in enumerate(row)
        if share > 0.0
    ]
    flows.sort(key=lambda t: t[2], reverse=True)
    return flows[:k]


def _od_line(
    orders: Sequence[Dict],
    centres: Sequence[Coord],
    zones: Sequence[Tuple[str, Coord]],
    label: str = "prev 1h",
    k: int = 5,
) -> str:
    """One compact ``top-k OD flows`` line, with zone names as a reading aid."""
    top = _top_od_pairs(orders, centres, zones, k=k)
    if not top:
        return "  (no OD flows available: no previous window or no region layout)"
    parts = []
    for i, j, share in top:
        oz = _nearest_zone(centres[i], zones) if zones else ""
        dz = _nearest_zone(centres[j], zones) if zones else ""
        if i == j:
            parts.append(f"r{i}[{oz}] internal {share:.1%}")
        else:
            parts.append(f"r{i}[{oz}] -> r{j}[{dz}] {share:.1%}")
    return f"  [{label}] top OD flows: " + " | ".join(parts)


# --------------------------------------------------------------------------- #
# The three profile blocks.                                                    #
# --------------------------------------------------------------------------- #
def _encode_map(
    env, phi: GlobalStats, zones: Sequence[Tuple[str, Coord]]
) -> str:
    n_regions = len(getattr(env, "region_neighbours", ()) or ())
    deg = (
        len(env.region_neighbours[0])
        if n_regions and env.region_neighbours[0]
        else 0
    )
    return (
        "== MAP / ROAD NETWORK ==\n"
        "Real Manhattan road network (OpenStreetMap). All distances/times come "
        "from the all-pairs shortest-path matrix, in MINUTES.\n"
        "Two spatial granularities for the SAME location:\n"
        "  - coordinate (continuous): (lon, lat). Travel time between two points "
        "is dist(a, b) = network.shortest_path(a, b).travel_time (minutes).\n"
        f"  - region (discrete): {n_regions} cells. region(point) = index of the "
        f"nearest region centre; region_neighbours[r] lists r's {deg} nearest "
        "neighbour regions (env-provided adjacency). self['current_region'] is a "
        "driver's region index.\n"
        "Coordinate -> named taxi-zone anchors (attach your geographic prior "
        "here; these names are for YOUR reasoning only):\n"
        f"{_anchor_lines(zones)}\n"
        f"Scale reference: in this regime a typical direct trip is "
        f"mean_solo_time ~= {phi.mean_solo_time:.1f} minutes.\n"
        "The frozen function you write receives BOTH granularities: coordinates "
        "+ dist() for exact travel time, and region indices for structure -- "
        "each order carries origin_region / destination_region, each driver "
        "carries current_region, and phi_ep carries region_centres, "
        "region_neighbours and the previous window's OD matrix (od_count / "
        "od_out / od_in). Every one of those indices comes from the SAME rule "
        "stated above, so they are directly comparable. Express every threshold "
        "in units of phi.mean_solo_time (never hard-code a distance constant) so "
        "the skill stays stable across regimes."
    )


def _window_stats_line(
    orders: Sequence[Dict],
    zones: Sequence[Tuple[str, Coord]],
    dist: Optional[Dist],
    rng: random.Random,
    label: str,
) -> str:
    """One compact demand-stats line for a single previous window."""
    n = len(orders)
    party = _party_distribution(orders)
    party_str = ", ".join(f"{p}p: {frac:.0%}" for p, frac in party.items()) or "n/a"

    solo_str = "(needs dist backend)"
    if dist is not None and orders:
        sample = orders if n <= 800 else rng.sample(list(orders), 800)
        solos = []
        for o in sample:
            try:
                solos.append(dist(o["origin"], o["destination"]))
            except Exception:  # noqa: BLE001 -- snap/backend hiccup: skip
                continue
        if solos:
            q = _quantiles(solos)
            solo_str = f"p10={q[0.1]:.1f}/p50={q[0.5]:.1f}/p90={q[0.9]:.1f} min"

    hot = _top_origin_zones(orders, zones, k=3)
    hot_str = ", ".join(f"{z} {frac:.0%}" for z, frac in hot) or "n/a"
    return (
        f"  [{label}] ~{n} orders | party {party_str} | "
        f"solo-time {solo_str} | hot origins: {hot_str}"
    )


def _encode_orders(
    prev_windows: Sequence[Tuple[str, Sequence[Dict]]],
    regime: str,
    zones: Sequence[Tuple[str, Coord]],
    dist: Optional[Dist],
    rng: random.Random,
    centres: Sequence[Coord] = (),
) -> str:
    """DEMAND block: profile each supplied previous window (e.g. h-1 and h-2).

    ``prev_windows`` is an ordered ``[(label, orders), ...]`` list -- typically
    the previous 1 hour AND previous 2 hours, so the LLM sees the demand TREND
    (rising into a peak vs falling off it), not just one lagged snapshot. Two
    lagged windows are a strictly stronger prior than one for forecasting the
    (still-unknown) current hour.

    ``centres`` are the env's region centres. When given, the most recent window
    also gets a top-OD-flow line: the same matrix the frozen function can read at
    runtime as ``phi_ep.od_count``, summarised here so the model knows what is in
    it before it writes code against it.
    """
    lines = [
        _window_stats_line(orders, zones, dist, rng, label)
        for label, orders in prev_windows
    ]
    stats_block = "\n".join(lines) if lines else "  (no previous windows available)"

    recent = prev_windows[0][1] if prev_windows else []
    recent_label = prev_windows[0][0] if prev_windows else "prev 1h"
    od_block = (_od_line(recent, centres, zones, label=recent_label, k=5)
                if centres else "")

    # One format example (dimensions/units only) from the most recent window.
    ex = "  (no previous-hour orders available)"
    if recent:
        o = recent[rng.randrange(len(recent))]
        oz = _nearest_zone(o["origin"], zones)
        dz = _nearest_zone(o["destination"], zones)
        st = dist(o["origin"], o["destination"]) if dist else float("nan")
        # Real indices, by the same rule the runtime uses, so the example shows
        # what the field actually looks like rather than a placeholder.
        o_r, d_r = _region_of(o["origin"], centres), _region_of(o["destination"], centres)
        ex = (
            f"  order_id=42, origin=(lon={o['origin'][0]:.4f}, "
            f"lat={o['origin'][1]:.4f}) [{oz}], origin_region={o_r}, destination="
            f"(lon={o['destination'][0]:.4f}, lat={o['destination'][1]:.4f}) "
            f"[{dz}], destination_region={d_r}, "
            f"num_passengers={o['num_passengers']}, "
            f"solo_time~={st:.1f} min, waiting_time=1 min"
        )

    return (
        "== DEMAND / ORDERS (profiled from PREVIOUS hours; current hour is "
        "unknown) ==\n"
        "At decision time the current hour's order stream has not happened yet, "
        "so demand is profiled from this split's real FHVHV windows one and two "
        "hours EARLIER -- a sound, leakage-free prior (real platforms forecast "
        f"demand the same way). regime: {regime} (real FHVHV).\n"
        "Order fields: order_id, origin=(lon,lat), destination=(lon,lat), "
        "origin_region(int), destination_region(int), num_passengers, "
        "waiting_time(min).\n"
        "previous-window demand (most recent first; compare the two to read the "
        "demand TREND into the current hour):\n"
        f"{stats_block}\n"
        + (f"{od_block}\n" if od_block else "")
        + "format example (illustrates field UNITS only, not a real current "
        f"order):\n{ex}"
    )


def _encode_clock(regime: str, split: str, scenario=None) -> str:
    """SIMULATED-TIME block: tell the LLM the day-of-week + hour of the sim.

    This is the *simulated* calendar time of the replayed FHVHV window (NOT the
    wall-clock/system time), which lets the model bring its prior about weekday
    vs weekend and morning vs evening demand shape to bear on the current scene.
    """
    if scenario is not None:
        dow, hour = scenario.clock
    else:
        dow, hour = simulated_clock(regime, split)
    if dow is None:
        when = f"hour {hour:02d}:00" if hour is not None else f"regime {regime}"
        note = "(day-of-week unavailable; manifest missing)"
    else:
        when = f"{dow}, {hour:02d}:00"
        weekend = dow in ("Saturday", "Sunday")
        note = (
            "weekend demand profile (later, more leisure/outer-borough trips)"
            if weekend
            else "weekday demand profile (commute-shaped peaks)"
        )
    return (
        "== SIMULATED TIME (not wall-clock) ==\n"
        f"The current simulated time is {when} -- {note}. Use this to interpret "
        "the demand profile above (e.g. an 18:00 weekday window is an evening "
        "commute peak). This is the SIMULATION's own clock, derived from the "
        "replayed historical window, not the real time you are running at."
    )


def _encode_variability(ranges: Optional[ScenarioRanges]) -> str:
    """SCENARIO-VARIABILITY block: state the deployment envelope + demand scale-
    invariance, so the LLM writes a function that transfers with zero retraining.

    This is the single most important prompt lever for the v2 generalization
    goal: the function is evolved across a RANDOMIZED family of scenarios and
    must work at deployment on fleet/capacity/speed/demand/preference values it
    never saw, so it must never hard-code a fleet size or a distance constant.
    """
    if ranges is None:
        ranges = ScenarioRanges()
    r = ranges
    if r.order_limits:
        vol_line = (
            f"  - order volume:    per-episode order cap sampled from "
            f"{list(r.order_limits)} (None = the FULL hour window)\n"
        )
    else:
        vol_line = (
            f"  - order volume:    order cap {r.order_limit} "
            "(None = the FULL hour window)\n"
        )
    return (
        "== DEPLOYMENT VARIABILITY (write a SCALE-INVARIANT function) ==\n"
        "This function is frozen ONCE and then deployed, with NO retraining, "
        "across a whole family of scenarios that vary along five axes:\n"
        f"  - fleet size:      {r.fleet[0]}..{r.fleet[1]} vehicles "
        f"(log-uniform sampling: small fleets are as common as large ones)\n"
        f"  - driver capacity: {r.capacity[0]}..{r.capacity[1]} seats "
        "(read it live from self['capacity']; order party sizes never exceed it)\n"
        f"  - driver speed:    {r.speed_kmh[0]:g}..{r.speed_kmh[1]:g} km/h "
        "(already folded into every dist()/eta travel time in MINUTES)\n"
        f"  - time-of-day:     {', '.join(r.regimes)} (demand volume/shape shift)\n"
        f"  - platform pref:   revenue weight {r.revenue[0]:g}..{r.revenue[1]:g}\n"
        f"{vol_line}"
        "REQUIREMENT: never hard-code a fleet size, a raw distance/time constant, "
        "or an absolute revenue threshold. Express every threshold RELATIVELY -- "
        "in units of phi.mean_solo_time for time/distance, of self['capacity'] "
        "for load, and of phi.demand_pressure for congestion. A function that "
        "keys off the scene's own scale transfers to all five axes; one that "
        "bakes in constants from a single operating point does not."
    )


def _encode_drivers(
    env, phi: GlobalStats, zones: Sequence[Tuple[str, Coord]]
) -> str:
    num_drivers = len(getattr(env, "drivers", {}) or {})
    cap = next(iter(env.drivers.values())).capacity if num_drivers else "?"
    anchor = _ANCHOR_ZONES[2] if zones else "?"
    return (
        "== SUPPLY / FLEET ==\n"
        f"Fleet: {num_drivers} vehicles, capacity {cap}, default speed (= network "
        "speed).\n"
        "Initial state: the simulation OPENS with every car idle, "
        "committed_passengers=0, no onboard orders; the load fields fill in as "
        "the rollout proceeds.\n"
        "self observation fields (FULL list -- en-route skills need these even "
        "though they are empty at t=0, so none may be omitted):\n"
        "  location=(lon,lat), current_region, status(idle/enroute/...), "
        "capacity, committed_passengers,\n"
        "  assigned_order_details=[{order_id, origin, destination, "
        "num_passengers, onboard(bool), eta(min)}].\n"
        f"global pressure: demand_pressure(pending/free-capacity) = "
        f"{phi.demand_pressure:.2f} (low at t=0: few pending, lots of idle "
        "capacity).\n"
        "format example (what a car that has ALREADY picked up looks like, for "
        "field units):\n"
        f"  driver_id=7 status=enroute location=(lon,lat) [{anchor}] "
        "current_region=<idx> committed_passengers=2 "
        "assigned_order_details=[{order_id=.., onboard=True, eta~=6 min}, ...]"
    )


# --------------------------------------------------------------------------- #
# Public entry point.                                                          #
# --------------------------------------------------------------------------- #
def encode_env_profile(
    env,
    phi: GlobalStats,
    regime: str,
    split: str = "train",
    rng: Optional[random.Random] = None,
    dist: Optional[Dist] = None,
    prev_orders: Optional[Sequence[Dict]] = None,
    *,
    scenario=None,
    ranges: Optional[ScenarioRanges] = None,
    prev_windows: Sequence[int] = (1, 2),
) -> str:
    """Build the full MAP / DEMAND / TIME / VARIABILITY / DRIVER profile string.

    Parameters
    ----------
    env : RidePoolEnv
        A reset env for this regime (provides fleet, region graph, network).
    phi : GlobalStats
        Global context (scale = ``mean_solo_time``, ``demand_pressure``).
    regime, split :
        Select the previous-hour order window(s) for the DEMAND profile.
    rng :
        Seeded RNG for the (single) format example + solo-time sub-sampling, so
        the same inputs reproduce the same prompt (§3.5).
    dist :
        Optional ``(a, b) -> minutes`` for solo-time quantiles/example. When
        omitted the DEMAND block still renders (party/hot-zone), just without
        trip-time stats -- keeps the encoder testable with no road network.
    prev_orders :
        Pre-loaded PREVIOUS-1h orders (back-compat). When omitted, the previous
        windows in ``prev_windows`` are loaded via :func:`load_prev_window_orders`.
    scenario : Scenario, optional
        The v2 scenario (for the SIMULATED-TIME block: day-of-week + hour). When
        omitted the clock is derived from ``(regime, split)`` directly.
    ranges : ScenarioRanges, optional
        The domain-randomization envelope (for the DEPLOYMENT-VARIABILITY block
        that pushes the LLM to write a scale-invariant function). When omitted
        the default envelope is described; pass ``ranges`` explicitly during v2
        evolution so the stated ranges match the actual training distribution.
    prev_windows :
        Which hour offsets to profile in the DEMAND block (default the previous
        1h AND 2h, so the LLM reads the demand trend). ``(1,)`` reproduces the
        v1 single-window behaviour.
    """
    rng = rng or random.Random(0)
    zones = _load_zones("Manhattan")
    # The env's OWN region centres: the OD summary must be bucketed by the same
    # partition the runtime uses, or the indices the model reads in the prompt
    # would not be the indices it can act on in phi_ep.
    centres = tuple(
        tuple(c) for c in (getattr(env, "relocation_points", ()) or ())
    )

    # Assemble the ordered list of (label, orders) previous windows.
    windows: List[Tuple[str, Sequence[Dict]]] = []
    if prev_orders is not None:
        windows.append(("prev 1h", prev_orders))
        offsets = [b for b in prev_windows if b != 1]
    else:
        offsets = list(prev_windows)
    for back in offsets:
        try:
            orders = load_prev_window_orders(regime, split, back=back)
        except Exception:  # noqa: BLE001 -- profiling must never crash evolution
            orders = []
        windows.append((f"prev {back}h", orders))
    # Keep windows in ascending lag order (1h before 2h) for readability.
    windows.sort(key=lambda t: int("".join(ch for ch in t[0] if ch.isdigit()) or 0))

    return "\n\n".join(
        [
            _encode_map(env, phi, zones),
            _encode_orders(windows, regime, zones, dist, rng, centres),
            _encode_clock(regime, split, scenario),
            _encode_variability(ranges),
            _encode_drivers(env, phi, zones),
        ]
    )


# --------------------------------------------------------------------------- #
# CLI: print the profile for a real window so a human can sanity-read it.      #
# --------------------------------------------------------------------------- #
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Print the LLM env profile.")
    ap.add_argument("--regime", default="peak", choices=sorted(TIME_OF_DAY))
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-drivers", type=int, default=1000)
    ap.add_argument("--order-limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from pref_dispatch.nyc_env import make_nyc_env

    env = make_nyc_env(
        seed=args.seed,
        regime=args.regime,
        split=args.split,
        num_drivers=args.num_drivers,
        order_limit=args.order_limit,
    )
    env.reset(seed=args.seed)

    def dist(a: Coord, b: Coord) -> float:
        return env.network.shortest_path(a, b).travel_time

    obs = env._build_observations()
    phi = GlobalStats.from_observations(obs, dist=dist)
    rng = random.Random(args.seed)

    profile = encode_env_profile(
        env, phi, regime=args.regime, split=args.split, rng=rng, dist=dist
    )
    print(profile)


if __name__ == "__main__":
    _cli()
