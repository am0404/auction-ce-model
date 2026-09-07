"""ROUGH CE -- APPROXIMATE AUCTION FUTURES. Not audited, not certified.

This module exists for one evening. It answers "is bidding one more dollar
better than stopping?" in seconds rather than minutes, and it buys that speed
by spending accuracy in ways that are named here and labelled everywhere the
answer is shown.

What is real
------------
The championship equity is a real simulation: real ``PlayerSpec``s, the real
league's scoring and lineup eligibility, the real median-matchup regular
season, the real byes and bracket. There is no regression from lineup points
to CE and no invented dollar-to-CE coefficient. Both arms are completed by the
existing bounded completion search over the actual current
:class:`AuctionState`, and every world is the existing reconciled joint world,
so duplicate ownership, exact budgets, the $1-per-open-slot reserve and roster
legality are enforced by the same code the certified path uses.

What is approximate
-------------------
Three things, and they are the reason for the label:

1. **Few futures.** The certified path runs eleven exchangeable rotations,
   which for eleven rivals is the whole balanced set and therefore exact. Here
   ``k`` is three or four. The ensemble mean is then a *sample* of the future
   auctions, not all of them, and the spread across them is reported as a
   range rather than smoothed into a single number.
2. **Few seasons.** Monte Carlo error is real and is reported beside, never
   added to, the between-future spread.
3. **A narrow beam.** ``ROUGH_COMPLETION`` searches far less of the board than
   the certified settings, so each arm's continuation is a good roster rather
   than a proved-best one.

The dominant uncertainty here is approximation uncertainty, not Monte Carlo
error. More seasons cannot shrink the disagreement between futures, so the
verdict rules below are about *signs across futures*, not about a t-test.

The cached worlds
-----------------
Benchmarking found that generating the player-week draws was 82% of the cost of
an evaluation, and that those draws do not depend on the rosters at all -- only
on the pool, the seed and the season index. They are also keyed by player
identity, so a player's draws are byte-identical whether the pool handed to the
generator holds 180 players or all 549 (asserted in the tests). So they are
generated once, before the draft, and reused by every arm and every future.

That reuse is not only a speed trick: it is what makes the comparison *paired*.
Buy and pass see the identical seasons, so a player on both rosters contributes
an exact zero to the difference and the season-to-season noise that would
otherwise dominate cancels.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..auction.completion import (ComparisonCast, CompletionSettings,
                                  _build_roster_set)
from ..auction.costs import CostBook
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionRuleError, AuctionState
from ..league import LeagueSettings
from ..lineup_vec import select_lineups_mask
from ..playoffs import run_bracket
from ..schedule import opponents_for_batch
from ..standings import regular_season
from ..tactical.board import BoardSettings
from ..tactical.ensemble import balanced_schedule
from ..tactical.joint import build_joint_worlds
from ..worlds import build_pool_arrays, generate_world

__all__ = [
    "ROUGH_LABEL",
    "ROUGH_DISCLAIMER",
    "ROUGH_COMPLETION",
    "ROUGH_BOARD",
    "RoughWorldCache",
    "RoughFuture",
    "RoughResult",
    "RoughLadder",
    "rough_buy_vs_pass",
    "rough_price_ladder",
    "cache_path_for",
    "build_cache",
]

#: The only names this result may ever be shown under.
ROUGH_LABEL = "ROUGH CE — APPROXIMATE AUCTION FUTURES"
ROUGH_DISCLAIMER = ("Not audited. Not a certified maximum bid. "
                    "A small sample of approximate auction futures.")

#: Words this module refuses to emit. The tests assert on this list, because
#: the whole point of the branch is that a rough answer is never mistaken for
#: the certified one.
FORBIDDEN_WORDS = ("CE AUDITED", "CERTIFIED", "AUDITED", "EXACT", "OPTIMAL",
                   "GUARANTEED", "MAX BID", "MAXIMUM BID")

DEFAULT_SEASONS = 1200
DEFAULT_WORLD_SEED = 917_324_011
DEFAULT_FUTURES = 3

ROUGHCE_DIR = Path("local_data/roughce")

#: Deliberately narrower than the certified settings, and narrower again than
#: the tactical LEAN. Every one of these numbers is a real bound on what was
#: searched, which is why the answer stays labelled ROUGH.
ROUGH_COMPLETION = CompletionSettings(
    beam_width=16, candidate_pool=20, proxy_candidates=16, finalists=1,
    proxy_reps=16, max_candidates=64, selection_sims=200, evaluation_sims=200,
    rival_selection="proxy")

#: The rivals' continuation must still be able to fill twelve rosters of
#: fifteen, so the pool depth and allocation count are NOT reduced: cutting
#: them below ~200 makes the continuation illegal rather than merely rough.
ROUGH_BOARD = BoardSettings()


# ---------------------------------------------------------------------------
# The cached player-week draws
# ---------------------------------------------------------------------------


def _pool_digest(pool: Sequence[object], settings: LeagueSettings) -> str:
    """Identity of the simulated universe: the specs and the league rules.

    Anything that changes a player's draws or the season structure must change
    this, or a cache built under one performance scenario would silently answer
    questions asked under another.
    """
    h = hashlib.sha256()
    for spec in pool:
        h.update(repr(spec).encode())
    h.update(repr(settings).encode())
    return h.hexdigest()[:16]


def cache_path_for(pool: Sequence[object], settings: LeagueSettings, *,
                   seasons: int = DEFAULT_SEASONS,
                   seed: int = DEFAULT_WORLD_SEED,
                   scenario: str = "base",
                   root: Path = ROUGHCE_DIR) -> Path:
    fp = _pool_digest(pool, settings)
    return Path(root) / f"worlds_{fp}_{scenario}_s{seasons}_x{seed}"


@dataclass
class RoughWorldCache:
    """Player-week draws for a fixed seed and season count, held once.

    The arrays are ``(S, P, W)`` over the FULL pool, indexed by the player's
    position in ``pool``. ``index_by_id`` is the only way a roster is turned
    into columns, so a cache can never be read against a different pool.
    """

    seasons: int
    seed: int
    scenario: str
    digest: str
    index_by_id: Dict[int, int]
    position: np.ndarray
    projection: np.ndarray
    available: np.ndarray
    realized: np.ndarray
    settings: LeagueSettings
    build_time_s: float = 0.0
    path: Optional[Path] = None

    @property
    def fingerprint(self) -> str:
        """What this cache is, in one string the result carries."""
        return f"{self.digest}/{self.scenario}/s{self.seasons}/x{self.seed}"

    def roster_matrix(self, rosters: Sequence[Sequence[int]]) -> np.ndarray:
        """``(T, R)`` columns into the cached arrays.

        Raises rather than guessing: a player the cache has never heard of is a
        stale cache or the wrong board, and inventing a column for him would
        quietly answer a different question.
        """
        out = []
        for team in rosters:
            row = []
            for pid in team:
                idx = self.index_by_id.get(int(pid))
                if idx is None:
                    raise AuctionRuleError(
                        f"player {pid} is not in the cached world "
                        f"({self.fingerprint}); the cache is stale for this "
                        f"board and is refused rather than extrapolated")
                row.append(idx)
            out.append(row)
        widths = {len(r) for r in out}
        if len(widths) != 1:
            raise AuctionRuleError(
                f"rosters are ragged ({sorted(widths)}); every team must be "
                f"the same size before a league can be simulated")
        return np.asarray(out, dtype=np.int64)

    def champion_indicator(self, rosters: Sequence[Sequence[int]],
                           team_index: int) -> np.ndarray:
        """Per-season 1/0 for ``team_index`` winning the title.

        The indicator, not just the mean, because it is what makes the paired
        difference possible: two arms read the same cached seasons, so
        differencing per season removes the noise that would otherwise swamp a
        CE gap of a few thousandths.
        """
        rm = self.roster_matrix(rosters)
        scores = self._scores(rm)
        opp = opponents_for_batch(self.seed, 0, self.seasons, self.settings)
        rs = regular_season(scores, opp, self.settings)
        po = run_bracket(scores, rs, self.settings)
        return (po.champion == team_index).astype(np.float64)

    def _scores(self, rm: np.ndarray) -> np.ndarray:
        """``(S, T, W)`` team scores, the same computation as ``team_scores``.

        The information barrier is preserved exactly as it is in
        :func:`ceauction.simulate.team_scores`: the lineup mask is built from
        the projection and availability alone, and ``realized`` is touched only
        after the mask exists.
        """
        proj = np.moveaxis(self.projection[:, rm, :], 2, -1)
        avail = np.moveaxis(self.available[:, rm, :], 2, -1)
        pos = np.broadcast_to(self.position[rm][None, :, None, :], proj.shape)
        mask = select_lineups_mask(proj, avail, pos)
        realized = np.moveaxis(self.realized[:, rm, :], 2, -1)
        return np.einsum("stwr,stwr->stw", mask, realized)

    # --- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "projection.npy", self.projection)
        np.save(path / "available.npy", self.available)
        np.save(path / "realized.npy", self.realized)
        np.save(path / "position.npy", self.position)
        meta = {"seasons": self.seasons, "seed": self.seed,
                "scenario": self.scenario, "digest": self.digest,
                "build_time_s": round(self.build_time_s, 2),
                "index_by_id": {str(k): v for k, v in self.index_by_id.items()},
                "warning": "LOCAL ONLY -- gitignored approximate world draws."}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.path = path

    @classmethod
    def load(cls, path: Path, settings: LeagueSettings) -> "RoughWorldCache":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(
            seasons=int(meta["seasons"]), seed=int(meta["seed"]),
            scenario=str(meta["scenario"]), digest=str(meta["digest"]),
            index_by_id={int(k): int(v) for k, v in meta["index_by_id"].items()},
            position=np.load(path / "position.npy"),
            # mmap: the arrays are ~130MB and only ever read.
            projection=np.load(path / "projection.npy", mmap_mode="r"),
            available=np.load(path / "available.npy", mmap_mode="r"),
            realized=np.load(path / "realized.npy", mmap_mode="r"),
            settings=settings, build_time_s=float(meta.get("build_time_s", 0)),
            path=path)


def build_cache(pool: Sequence[object], settings: LeagueSettings, *,
                seasons: int = DEFAULT_SEASONS,
                seed: int = DEFAULT_WORLD_SEED, scenario: str = "base",
                chunk: int = 256, progress=None) -> RoughWorldCache:
    """Generate and hold the draws. This is the slow, once-per-draft step."""
    t0 = time.perf_counter()
    pool = list(pool)
    arrays = build_pool_arrays(pool, settings)
    n_players = len(pool)
    n_weeks = arrays.n_weeks
    projection = np.empty((seasons, n_players, n_weeks), dtype=np.float32)
    available = np.empty((seasons, n_players, n_weeks), dtype=bool)
    realized = np.empty((seasons, n_players, n_weeks), dtype=np.float32)
    for start in range(0, seasons, chunk):
        size = min(chunk, seasons - start)
        w = generate_world(arrays, seed, start, size)
        sl = slice(start, start + size)
        projection[sl] = w.pregame.projection.astype(np.float32)
        available[sl] = w.availability.available
        realized[sl] = w.realized.points.astype(np.float32)
        if progress is not None:
            progress(min(start + size, seasons), seasons)
    return RoughWorldCache(
        seasons=seasons, seed=seed, scenario=scenario,
        digest=_pool_digest(pool, settings),
        index_by_id={int(spec.player_id): i for i, spec in enumerate(pool)},
        position=arrays.position, projection=projection, available=available,
        realized=realized, settings=settings,
        build_time_s=time.perf_counter() - t0)


def load_or_build_cache(pool: Sequence[object], settings: LeagueSettings, *,
                        seasons: int = DEFAULT_SEASONS,
                        seed: int = DEFAULT_WORLD_SEED,
                        scenario: str = "base",
                        root: Path = ROUGHCE_DIR,
                        progress=None) -> RoughWorldCache:
    """Read the cache, or build and persist it.

    The path carries the pool digest, so a changed performance scenario or a
    changed board does not hit a stale cache -- it misses, and rebuilds.
    """
    path = cache_path_for(pool, settings, seasons=seasons, seed=seed,
                          scenario=scenario, root=root)
    if (path / "meta.json").exists():
        try:
            return RoughWorldCache.load(path, settings)
        except (OSError, ValueError, KeyError):
            pass  # a corrupt cache is rebuilt, never patched
    cache = build_cache(pool, settings, seasons=seasons, seed=seed,
                        scenario=scenario, progress=progress)
    cache.save(path)
    return cache


# ---------------------------------------------------------------------------
# One future, and the ensemble of them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoughFuture:
    """One approximate future auction, evaluated on both arms."""

    index: int
    rotation: int
    seed: int
    delta_ce: float
    ce_buy: float
    ce_pass: float
    paired_se: float
    alloc_fingerprint: str
    conservation_ok: bool
    conservation_problems: Tuple[str, ...]
    league_ce_sum: float
    runtime_s: float

    def to_dict(self) -> Dict[str, object]:
        return {"index": self.index, "rotation": self.rotation,
                "seed": self.seed, "delta_ce": round(self.delta_ce, 6),
                "ce_buy": round(self.ce_buy, 5),
                "ce_pass": round(self.ce_pass, 5),
                "paired_se": round(self.paired_se, 6),
                "alloc_fingerprint": self.alloc_fingerprint,
                "conservation_ok": self.conservation_ok,
                "conservation_problems": list(self.conservation_problems),
                "league_ce_sum": round(self.league_ce_sum, 6),
                "runtime_s": round(self.runtime_s, 2)}


@dataclass(frozen=True)
class RoughResult:
    """The rough answer, its spread, and the words it may be shown under."""

    candidate_id: int
    candidate_name: str
    price: int
    stop_price: int
    recipient: str
    recipient_name: str
    focus_owner_id: str
    futures: Tuple[RoughFuture, ...]
    state_fingerprint: str
    cache_fingerprint: str
    seasons: int
    runtime_s: float
    label: str = ROUGH_LABEL
    disclaimer: str = ROUGH_DISCLAIMER
    stale: bool = False
    notes: str = ""

    @property
    def k(self) -> int:
        return len(self.futures)

    @property
    def deltas(self) -> Tuple[float, ...]:
        return tuple(f.delta_ce for f in self.futures)

    @property
    def mean_delta(self) -> float:
        return statistics.fmean(self.deltas) if self.futures else float("nan")

    @property
    def delta_range(self) -> Tuple[float, float]:
        """Spread ACROSS futures. Seasons cannot shrink this."""
        if not self.futures:
            return (float("nan"), float("nan"))
        return (min(self.deltas), max(self.deltas))

    @property
    def between_future_sd(self) -> float:
        if self.k < 2:
            return float("nan")
        return statistics.stdev(self.deltas)

    @property
    def monte_carlo_se(self) -> float:
        """Season noise inside a future. Reported beside, never added to, the
        between-future spread: they are different uncertainties and summing
        them would understate the one that matters."""
        if not self.futures:
            return float("nan")
        return statistics.fmean([f.paired_se for f in self.futures])

    @property
    def conservation_ok(self) -> bool:
        return all(f.conservation_ok for f in self.futures)

    @property
    def verdict(self) -> str:
        """Signs across futures, not a t-test.

        ROUGH BUY needs *every* evaluated future positive and the Monte Carlo
        interval on the mean clear of zero. Anything else is MIXED, including
        the case where the futures agree but the seasons cannot tell the
        difference from zero. The market answer never substitutes for MIXED.
        """
        if not self.futures:
            return "NO RESULT"
        if not self.conservation_ok:
            return "REFUSED — CONSERVATION FAILED"
        lo, hi = self.delta_range
        mc = self.monte_carlo_se
        crosses = (not math.isnan(mc)
                   and abs(self.mean_delta) <= 1.96 * mc)
        if lo > 0 and not crosses:
            return "ROUGH BUY"
        if hi < 0 and not crosses:
            return "ROUGH PASS"
        return "MIXED / TOO CLOSE"

    def headline(self) -> str:
        lo, hi = self.delta_range
        return (f"{self.label}\n"
                f"Bid ${self.price} vs stop at ${self.stop_price} to "
                f"{self.recipient_name}\n"
                f"ΔCE: {self.mean_delta:+.4f}\n"
                f"Future range: {lo:+.4f} to {hi:+.4f}  "
                f"({self.k} futures × {self.seasons} seasons)\n"
                f"{self.verdict}\n"
                f"{self.disclaimer}")

    def to_dict(self) -> Dict[str, object]:
        lo, hi = self.delta_range
        return {
            "label": self.label, "disclaimer": self.disclaimer,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "price": self.price, "stop_price": self.stop_price,
            "recipient": self.recipient, "recipient_name": self.recipient_name,
            "focus_owner_id": self.focus_owner_id,
            "verdict": self.verdict,
            "mean_delta_ce": round(self.mean_delta, 6),
            "delta_ce_low": round(lo, 6), "delta_ce_high": round(hi, 6),
            "between_future_sd": (None if math.isnan(self.between_future_sd)
                                  else round(self.between_future_sd, 6)),
            "monte_carlo_se": (None if math.isnan(self.monte_carlo_se)
                               else round(self.monte_carlo_se, 6)),
            "k_futures": self.k, "seasons": self.seasons,
            "futures": [f.to_dict() for f in self.futures],
            "state_fingerprint": self.state_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "conservation_ok": self.conservation_ok,
            "runtime_s": round(self.runtime_s, 2),
            "stale": self.stale, "notes": self.notes,
        }


def _arm(state: AuctionState, cast: ComparisonCast, costs: CostBook,
         cache: RoughWorldCache, *, board: BoardSettings,
         completion: CompletionSettings, proxy: ProxyEvaluator,
         market, key_by_id, acquired):
    """Build one reconciled joint world and read its equity off the cache."""
    worlds = build_joint_worlds(
        state, cast, costs, board_settings=board, completion=completion,
        market=market, key_by_id=key_by_id, proxy=proxy, default_cost=1,
        max_worlds=1, branch_acquired=acquired)
    if not worlds:
        raise AuctionRuleError(
            "no legal joint world for this arm; the rough path refuses rather "
            "than substituting an illegal league")
    w = worlds[0]
    rosters = _build_roster_set(w.state, w.cast, w.focus_roster)
    ids = [r.player_ids for r in rosters.rosters]
    ind = cache.champion_indicator(ids, w.cast.focus_team_index)
    return w, ind


def rough_buy_vs_pass(
    state: AuctionState,
    cast: ComparisonCast,
    costs: CostBook,
    cache: RoughWorldCache,
    candidate_id: int,
    current_bid: int,
    recipient: str,
    *,
    k_futures: int = DEFAULT_FUTURES,
    completion: CompletionSettings = ROUGH_COMPLETION,
    board: BoardSettings = ROUGH_BOARD,
    market=None,
    key_by_id: Optional[Dict[int, str]] = None,
    proxy: Optional[ProxyEvaluator] = None,
    name_by_id: Optional[Dict[int, str]] = None,
    team_names: Optional[Dict[str, str]] = None,
    runtime_budget_s: float = 30.0,
    buy_price: Optional[int] = None,
    pass_price: Optional[int] = None,
) -> RoughResult:
    """Bidding ``current_bid + 1`` versus letting ``recipient`` have him at
    ``current_bid``.

    The semantics are the live ones and they are not symmetric: we buy at
    ``q+1`` because that is what the next bid costs, and the named leader buys
    at ``q`` because that is the bid he has already made. Passing is not "the
    player evaporates" -- it is "that owner gets him at that price", which
    spends his money and fills his slot.
    """
    t0 = time.perf_counter()
    focus = state.focus_owner_id
    # Live semantics: our next bid is q+1 and the leader already stands at q.
    # The overrides exist for the calibration ladder, which holds the pass
    # price fixed while our bid climbs -- the certified FIXED_MARKET rule. A
    # reservation result is conditional on the pass rule, so the rule used is
    # carried on the result rather than left implicit.
    price = int(buy_price) if buy_price is not None else int(current_bid) + 1
    stop = int(pass_price) if pass_price is not None else int(current_bid)

    if not state.is_available(candidate_id):
        raise AuctionRuleError(
            f"player {candidate_id} is not available in this room")
    if recipient == focus:
        raise AuctionRuleError(
            "the current high bidder must be an opponent, not us; there is no "
            "buy-versus-pass question when we are already the leader")
    if recipient not in {o.owner_id for o in state.owners}:
        raise AuctionRuleError(f"no owner {recipient!r} in this room")
    bad = state.purchase_shortfall(candidate_id, focus, price)
    if bad is not None:
        raise AuctionRuleError(f"we cannot legally bid ${price}: {bad}")
    bad = state.purchase_shortfall(candidate_id, recipient, stop)
    if bad is not None:
        raise AuctionRuleError(
            f"{recipient} cannot legally hold him at ${stop}: {bad}. The pass "
            f"branch would not be a legal auction, so it is refused.")

    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings,
                               completion.proxy_reps, completion.proxy_seed)

    buy_state = state.apply_purchase(candidate_id, focus, price)
    pass_state = state.award_to_rival(candidate_id, recipient, stop)

    draws = balanced_schedule(len(state.owners), max(1, int(k_futures)))
    futures: List[RoughFuture] = []
    for d in draws:
        if futures and time.perf_counter() - t0 > runtime_budget_s:
            break
        ts = time.perf_counter()
        # The SAME future serves both arms. Buy and pass differ only in who
        # owns the candidate, never in which future auction they face.
        bs = dataclasses_replace(board, seed=d.seed,
                                 owner_priority=d.permutation)
        common = dict(board=bs, completion=completion, proxy=proxy,
                      market=market, key_by_id=key_by_id)
        wb, ind_buy = _arm(buy_state, cast, costs, cache,
                           acquired=frozenset({candidate_id}), **common)
        wp, ind_pass = _arm(pass_state, cast, costs, cache,
                            acquired=frozenset(), **common)
        d_arr = ind_buy - ind_pass
        problems = tuple(list(wb.conservation.problems())
                         + list(wp.conservation.problems()))
        futures.append(RoughFuture(
            index=d.index, rotation=d.rotation, seed=d.seed,
            delta_ce=float(d_arr.mean()),
            ce_buy=float(ind_buy.mean()), ce_pass=float(ind_pass.mean()),
            paired_se=_paired_se(d_arr),
            alloc_fingerprint=wb.fingerprint(),
            conservation_ok=bool(wb.conservation.ok and wp.conservation.ok),
            conservation_problems=problems,
            league_ce_sum=_league_sum(cache, wb),
            runtime_s=time.perf_counter() - ts))

    names = name_by_id or {}
    teams = team_names or {}
    return RoughResult(
        candidate_id=candidate_id,
        candidate_name=names.get(candidate_id, str(candidate_id)),
        price=price, stop_price=stop, recipient=recipient,
        recipient_name=teams.get(recipient, recipient), focus_owner_id=focus,
        futures=tuple(futures), state_fingerprint=state.fingerprint(),
        cache_fingerprint=cache.fingerprint, seasons=cache.seasons,
        runtime_s=time.perf_counter() - t0,
        notes=("pass rule: leader holds at $%d" % stop))


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def _paired_se(d: np.ndarray) -> float:
    n = d.size
    if n < 2:
        return float("nan")
    return float(d.std(ddof=1) / math.sqrt(n))


def _league_sum(cache: RoughWorldCache, world) -> float:
    """Every team's equity in this world, which must sum to one.

    A league in which somebody always wins has total equity of exactly 1. If
    this drifts, the simulated league is not a league and the answer is void.
    """
    rosters = _build_roster_set(world.state, world.cast, world.focus_roster)
    rm = cache.roster_matrix([r.player_ids for r in rosters.rosters])
    scores = cache._scores(rm)
    opp = opponents_for_batch(cache.seed, 0, cache.seasons, cache.settings)
    rs = regular_season(scores, opp, cache.settings)
    po = run_bracket(scores, rs, cache.settings)
    n_teams = rm.shape[0]
    return float(sum((po.champion == t).mean() for t in range(n_teams)))


# ---------------------------------------------------------------------------
# The sparse price ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoughLadder:
    """Rough verdicts at a handful of actually-evaluated prices.

    Never interpolated. The bracket is between two prices that were really
    run, because the whole failure mode this branch guards against is a
    confident dollar figure standing on nothing.
    """

    results: Tuple[RoughResult, ...]
    runtime_s: float
    label: str = ROUGH_LABEL

    @property
    def prices(self) -> Tuple[int, ...]:
        return tuple(r.price for r in self.results)

    @property
    def bracket(self) -> Optional[Tuple[int, int]]:
        """Highest evaluated ROUGH BUY and the next evaluated non-BUY above it."""
        buys = [r.price for r in self.results if r.verdict == "ROUGH BUY"]
        if not buys:
            return None
        hi = max(buys)
        above = [r.price for r in self.results
                 if r.price > hi and r.verdict != "ROUGH BUY"]
        if not above:
            return None
        return (hi, min(above))

    def to_dict(self) -> Dict[str, object]:
        b = self.bracket
        return {"label": self.label,
                "evaluated_prices": list(self.prices),
                "rungs": [{"price": r.price, "verdict": r.verdict,
                           "mean_delta_ce": round(r.mean_delta, 6),
                           "delta_ce_low": round(r.delta_range[0], 6),
                           "delta_ce_high": round(r.delta_range[1], 6)}
                          for r in self.results],
                "bracket": list(b) if b else None,
                "frontier": (f"between ${b[0]} and ${b[1]} (both evaluated)"
                             if b else "FRONTIER NOT FOUND"),
                "runtime_s": round(self.runtime_s, 2),
                "disclaimer": ROUGH_DISCLAIMER}


def rough_price_ladder(state: AuctionState, cast, costs, cache, candidate_id,
                       prices: Sequence[int], recipient: str, **kw
                       ) -> RoughLadder:
    """Evaluate a handful of given prices. No binary search, no assumption of
    monotonicity -- every rung is run and reported as run."""
    t0 = time.perf_counter()
    out: List[RoughResult] = []
    for p in sorted({int(x) for x in prices}):
        try:
            out.append(rough_buy_vs_pass(state, cast, costs, cache,
                                         candidate_id, p - 1, recipient,
                                         buy_price=p, **kw))
        except AuctionRuleError:
            continue  # an illegal rung is skipped, never approximated
    return RoughLadder(results=tuple(out), runtime_s=time.perf_counter() - t0)
