"""One-factor roster-context regimes: only focus-player strength varies.

The previous regime experiment was confounded and is withdrawn as a causal
comparison. It changed the number of pre-owned players, and with it our money,
our open slots, our positional pattern, which players were left on the board,
which players rivals could reach, and therefore every rival's completion. Four
things moved at once and the labels did not even match the outcome --
``bye_contender`` finished sixth and ``favorite`` seventh.

This module changes exactly one thing.

Every regime is built from **one** fabricated auction. The focus team holds the
same player ids, at the same positions, bought for the same dollars, leaving the
same budget and the same open slots. Every rival roster, every rival budget, the
remaining board, the cost book, the market state and every seed are byte-identical.
The only difference is the ``base_mean`` of the PlayerSpecs the focus team
**already owns**. Those players are off the board in every regime, so scaling
their projected scoring cannot reach the auction at all: it changes how good our
existing roster is, and nothing else.

That is the whole design, and :func:`check_structural_equality` refuses an
experiment where anything else moved.

One expected difference is reported separately rather than hidden: the pool
fingerprint and the proxy strengths differ between regimes, because focus-player
scoring is the factor under test. Structural state is identical; player scoring
is not, by construction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.demo import build_demo_auction
from ..auction.state import AuctionState
from ..league import Position
from ..players import PlayerSpec
from .demo import TacticalDemo, build_tactical_demo, demo_sales

__all__ = [
    "ContextRegime",
    "CONTEXT_REGIMES",
    "N_PREOWNED",
    "PREOWNED_PRICE",
    "build_context_regime",
    "build_all_regimes",
    "StructuralDifference",
    "check_structural_equality",
    "expected_differences",
    "RegimeMetrics",
    "measure_regime",
]

#: Held identical in every regime. Eight pre-owned players gives the strength
#: factor real leverage over the finished roster without emptying the board.
N_PREOWNED = 8
PREOWNED_PRICE = 13


@dataclass(frozen=True)
class ContextRegime:
    """One stated roster context. The scale is the ONLY thing that varies."""

    name: str
    description: str
    strength_scale: float
    """Multiplier on ``base_mean`` for the focus team's pre-owned players.

    Calibrated against measured playoff, bye and championship-equity outcomes
    -- never assigned from proxy points, and never from an assumption about
    which context ought to value a player most."""

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "description": self.description,
                "strength_scale": self.strength_scale}


#: Scales calibrated by simulated outcome; see ``docs/TACTICAL_CONTEXT.md``.
CONTEXT_REGIMES: Dict[str, ContextRegime] = {r.name: r for r in (
    ContextRegime("underdog",
                  "clearly outside the playoff field (measured p(playoff) "
                  "0.049), but not pinned at zero equity", 0.88),
    ContextRegime("playoff_bubble",
                  "near the sixth-place playoff boundary (measured p(playoff) "
                  "0.461 -- materially uncertain)", 0.99),
    ContextRegime("bye_bubble",
                  "near the top-two bye boundary (measured p(bye) 0.514 -- "
                  "materially uncertain)", 1.06),
    ContextRegime("favorite",
                  "clearly among the strongest rosters (measured p(bye) 0.918, "
                  "CE rank 1)", 1.16),
)}


def _base_world() -> Tuple[TacticalDemo, Tuple[int, ...]]:
    """The single shared auction every regime is derived from.

    Built once, from the unscaled pool, so the rival assignment -- which is
    computed from projections -- is identical for every regime. Scaling the
    focus specs afterwards cannot disturb it.
    """
    base = build_tactical_demo()
    fresh = build_demo_auction(focus_keep=0)
    state = fresh.state
    focus = state.focus_owner_id
    board = sorted(state.available_specs,
                   key=lambda sp: (-sp.base_mean, sp.player_id))
    held: List[int] = []
    i = 0
    while len(held) < N_PREOWNED and i < len(board):
        spec = board[i]
        i += 1
        if state.purchase_shortfall(spec.player_id, focus,
                                    PREOWNED_PRICE) is not None:
            continue
        state = state.apply_purchase(spec.player_id, focus, PREOWNED_PRICE)
        held.append(spec.player_id)
    if len(held) < N_PREOWNED:
        raise ValueError(f"only {len(held)} of {N_PREOWNED} pre-owned players fit")
    state.validate()
    return base.with_state(state), tuple(held)


def build_context_regime(name: str, *, with_sales: bool = True) -> TacticalDemo:
    """A world identical to every other regime except focus-player strength."""
    if name not in CONTEXT_REGIMES:
        raise ValueError(
            f"unknown context regime {name!r}; "
            f"known: {', '.join(sorted(CONTEXT_REGIMES))}")
    scale = CONTEXT_REGIMES[name].strength_scale
    base, held = _base_world()
    owned = set(held)
    # Only the focus team's OWN specs are rescaled. Every other spec, and so the
    # whole remaining board, is passed through untouched.
    pool = tuple(
        replace(s, base_mean=round(s.base_mean * scale, 6),
                data_source=f"{s.data_source}|context:{name}")
        if s.player_id in owned else s
        for s in base.state.pool)
    state = replace(base.state, pool=pool)
    state.validate()
    d = base.with_state(state)
    if with_sales:
        d = d.with_market(d.market.observe_all(demo_sales(d, 12)))
    return d


def build_all_regimes(*, with_sales: bool = True) -> Dict[str, TacticalDemo]:
    return {n: build_context_regime(n, with_sales=with_sales)
            for n in CONTEXT_REGIMES}


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralDifference:
    """One prohibited field that moved between two regimes."""

    field: str
    left_regime: str
    right_regime: str
    left: str
    right: str

    def __str__(self) -> str:
        return (f"{self.field}: {self.left_regime}={self.left} "
                f"!= {self.right_regime}={self.right}")

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _structural(d: TacticalDemo) -> Dict[str, object]:
    """Everything that must be identical across regimes."""
    st = d.state
    focus = st.focus_owner_id
    o = st.owner(focus)
    return {
        "focus_owner_id": focus,
        "focus_budget_remaining": o.budget_remaining,
        "focus_spent": o.spent,
        "focus_roster_size": o.n_players,
        "focus_open_slots": o.open_slots,
        "focus_player_ids": tuple(sorted(o.player_ids)),
        "focus_positions": tuple(sorted(
            (f.player_id, int(f.position)) for f in o.filled)),
        "focus_prices": tuple(sorted((f.player_id, f.price) for f in o.filled)),
        "remaining_board_ids": tuple(sorted(st.available_ids)),
        "remaining_board_costs": tuple(
            (pid, d.costs.cost_of(pid, 1)) for pid in sorted(st.available_ids)),
        "rival_states": tuple(sorted(
            (r.owner_id, tuple(sorted(r.player_ids)), r.spent,
             r.budget_remaining, r.open_slots)
            for r in st.owners if r.owner_id != focus)),
        "market_fingerprint": d.market.fingerprint(),
        "cost_book_fingerprint": d.costs.fingerprint(),
        "league_settings": st.settings_fingerprint(),
        "withdrawn": tuple(sorted(st.withdrawn)),
        "cast_team_names": tuple(d.cast.team_names),
        "cast_rival_rosters": tuple(sorted(
            tuple(sorted(t)) for i, t in enumerate(d.cast.rosters)
            if i != d.cast.focus_team_index)),
    }


def check_structural_equality(
    regimes: Dict[str, TacticalDemo],
    *,
    candidate_id: Optional[int] = None,
    price: Optional[int] = None,
    recipient: Optional[str] = None,
    leader: Optional[str] = None,
    seeds: Sequence[int] = (),
) -> Tuple[StructuralDifference, ...]:
    """Every prohibited field, compared across every pair. Empty means valid.

    The experiment is invalid if this returns anything. Constant budget alone is
    not enough and this is what proves the rest.
    """
    names = sorted(regimes)
    tables = {n: _structural(regimes[n]) for n in names}
    for n in names:
        tables[n]["candidate_id"] = candidate_id
        tables[n]["candidate_price"] = price
        tables[n]["pass_recipient"] = recipient
        tables[n]["current_leader"] = leader
        tables[n]["seeds"] = tuple(seeds)
    out: List[StructuralDifference] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for field in tables[a]:
                if tables[a][field] != tables[b][field]:
                    la, lb = str(tables[a][field]), str(tables[b][field])
                    out.append(StructuralDifference(
                        field=field, left_regime=a, right_regime=b,
                        left=la[:80], right=lb[:80]))
    return tuple(out)


def expected_differences(regimes: Dict[str, TacticalDemo]) -> Dict[str, object]:
    """Differences that SHOULD exist: the factor under test, and only it."""
    names = sorted(regimes)
    pools = {n: regimes[n].state.pool_fingerprint() for n in names}
    focus_means: Dict[str, float] = {}
    other_means: Dict[str, float] = {}
    for n in names:
        st = regimes[n].state
        owned = set(st.owner(st.focus_owner_id).player_ids)
        focus_means[n] = round(sum(s.base_mean for s in st.pool
                                   if s.player_id in owned), 4)
        other_means[n] = round(sum(s.base_mean for s in st.pool
                                   if s.player_id not in owned), 4)
    return {
        "pool_fingerprints_differ": len(set(pools.values())) == len(names),
        "pool_fingerprints": pools,
        "focus_owned_projection_total": focus_means,
        "every_other_projection_total": other_means,
        "non_focus_projections_identical":
            len(set(other_means.values())) == 1,
        "note": ("pool fingerprints differ BY DESIGN: focus-owned scoring is "
                 "the factor under test. Every projection outside the focus "
                 "roster is identical, which is what keeps the board fixed."),
    }


# ---------------------------------------------------------------------------
# Measured pre-acquisition state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeMetrics:
    """What the simulator actually says about a regime, before any purchase."""

    regime: str
    strength_scale: float
    focus_proxy: float
    field_proxy_mean: float
    weekly_points: float
    mean_wins: float
    playoff_probability: float
    bye_probability: float
    championship_equity: float
    ce_rank: int
    league_ce_sum: float
    n_sims: int

    def to_dict(self) -> Dict[str, object]:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in dataclasses.asdict(self).items()}


def measure_regime(name: str, d: TacticalDemo, completion, costs, *,
                   proxy, sims: int, seed: int,
                   board_settings, market, key_by_id,
                   chunk: int = 64) -> RegimeMetrics:
    """Pre-acquisition playoff, bye and equity metrics for one regime.

    Uses the same reconciled joint-world machinery every other measurement in
    this project uses, so the number quoted here is the number the candidate
    comparison starts from.
    """
    from ..auction.completion import _build_roster_set
    from ..simulate import simulate_seasons
    from .joint import build_joint_worlds, evaluate_joint_arm

    st = d.state
    idx = d.cast.focus_team_index
    worlds = build_joint_worlds(
        st, d.cast, costs, board_settings=board_settings,
        completion=completion, market=market, key_by_id=key_by_id,
        proxy=proxy, default_cost=1, max_worlds=3)
    arm = evaluate_joint_arm(
        worlds, focus_team_index=idx, proxy=proxy,
        selection_sims=completion.selection_sims,
        selection_seed=completion.selection_seed,
        holdout_sims=sims, holdout_seed=seed, chunk=chunk)
    rosters = _build_roster_set(arm.world.state, arm.world.cast,
                               arm.world.focus_roster)
    out = simulate_seasons(rosters, sims, seed, chunk)
    ce = out.championship_equity()
    order = sorted(range(len(ce)), key=lambda i: -float(ce[i]))
    return RegimeMetrics(
        regime=name, strength_scale=CONTEXT_REGIMES[name].strength_scale,
        focus_proxy=arm.focus_proxy, field_proxy_mean=arm.field_mean,
        weekly_points=float(out.points[:, idx].mean()),
        mean_wins=float(out.wins[:, idx].mean()),
        playoff_probability=float(out.made_playoffs[:, idx].mean()),
        bye_probability=float(out.has_bye[:, idx].mean()),
        championship_equity=float(ce[idx]), ce_rank=order.index(idx) + 1,
        league_ce_sum=float(ce.sum()), n_sims=sims)
