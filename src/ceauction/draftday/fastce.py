"""LIVE CE — RAPID SAMPLE, on certified completion sets.

This is the hybrid the rough-CE NO-GO pointed at. That experiment failed
because a narrow **beam** produced the wrong opportunity sets, not because fast
CE is impossible: the beam's buy/pass sign moved with beam width alone. The
beam is therefore not tuned here. It is removed.

Two existing pieces are joined:

**Certified completion sets.** :func:`build_certified_offers` produces the
with-candidate and without-candidate opportunity sets with a solver
certificate, in seconds for a single price. Supplying both to
``build_eval_context`` means ``complete_roster`` is never called on this path
-- asserted in the tests, not merely intended.

**Precomputed outcome banks.** The player-week draws are 82% of an evaluation
and depend only on the pool, the seed and the season index -- never on the
rosters -- and they are keyed by player identity. So a bank reproduces
``simulate_seasons`` *season for season*: the tests assert bit-equality of the
champion array, not similarity. Reusing them is memoisation, not
approximation, and it is what lets K=11 and ``max_worlds=8`` run on a bidding
clock instead of being traded away for speed.

What is genuinely reduced
-------------------------
The season count, and only the season count. 200 selection / 1,000 holdout
against the recorded fixture's 4,000. That widens the interval; it does not
move the estimate's centre or change which rosters were on offer. Every
structural quantity -- K=11 rotations, ``max_worlds=8``, the certified sets,
the reconciliation and conservation invariants -- is held at the value the
stability experiment used.

So the completion search carries a certificate and a gap; the CE estimate
carries sampling uncertainty. The estimate is never called certified, and the
answer is for the next bid only -- it is not a maximum bid.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..auction.completion import CompletionSettings
from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionRuleError, AuctionState
from ..tactical.board import BoardSettings
from ..tactical.certified_frontier import gate_certified_offers
from ..tactical.certified_offers import build_certified_offers
from ..tactical.ensemble import balanced_schedule
from ..tactical.evalcontext import build_eval_context
from ..tactical.joint import clear_outcome_banks, register_outcome_bank
from ..tactical.midauction import PassPrice, PassPriceMode
from ..tactical.reconciled import check_agreement
from ..tactical.union_experiment import _uncertainty
from .roughce import RoughWorldCache, load_or_build_cache

__all__ = [
    "LIVE_LABEL", "LIVE_SUBLABEL", "LIVE_SCOPE", "FORBIDDEN_CLAIMS",
    "K_ROTATIONS", "MAX_WORLDS", "SELECTION_SEASONS", "HOLDOUT_SEASONS",
    "SELECTION_SEED", "HOLDOUT_SEED", "GAP_LIMIT",
    "OutcomeBanks", "LiveCEResult", "live_ce_next_bid", "prepare_offers",
]

LIVE_LABEL = "LIVE CE — RAPID SAMPLE"
LIVE_SUBLABEL = "CERTIFIED COMPLETION SETS"
LIVE_SCOPE = "NEXT BID ONLY — NOT A MAX BID"

#: Claims this path may never make. The completion search has a certificate;
#: the CE estimate has sampling uncertainty. Calling the estimate certified
#: would transfer a guarantee from the object that has one to the object that
#: does not.
FORBIDDEN_CLAIMS = ("CE AUDITED", "CERTIFIED CE", "CERTIFIED ESTIMATE",
                    "MAX BID", "MAXIMUM BID", "OPTIMAL BID", "GUARANTEED")

# --- predeclared sampling. Fixed before any result was observed. -----------
K_ROTATIONS = 11
MAX_WORLDS = 8
SELECTION_SEASONS = 200
HOLDOUT_SEASONS = 1_000
SELECTION_SEED = 20260904
HOLDOUT_SEED = 917_324_011
GAP_LIMIT = 0.10

#: The certified offer generator's own bounds, from the stability experiment.
OFFER_POOL_DEPTH = 40
OFFER_TARGET = 12
OFFER_BAND = 0.50


@dataclass
class OutcomeBanks:
    """The two independent samples, and the registration that installs them.

    Separate seeds, separate banks, separate sizes. A selection observation
    can never reach the reported number: the holdout bank is the only thing
    the interval is computed from, and a bank serves only an exact
    ``(sims, seed)`` request.
    """

    selection: RoughWorldCache
    holdout: RoughWorldCache

    def __post_init__(self) -> None:
        if self.selection.seed == self.holdout.seed:
            raise ValueError(
                "selection and holdout banks share a seed; the sample that "
                "chose a completion would then also report its advantage, "
                "which is the selection bias the holdout exists to remove")
        if self.selection.digest != self.holdout.digest:
            raise ValueError(
                "the two banks were built over different pools; they cannot "
                "describe the same league")

    @property
    def fingerprint(self) -> str:
        return f"sel[{self.selection.fingerprint}]+hold[{self.holdout.fingerprint}]"

    def install(self) -> None:
        clear_outcome_banks()
        register_outcome_bank(self.selection.seasons, self.selection.seed,
                              self.selection)
        register_outcome_bank(self.holdout.seasons, self.holdout.seed,
                              self.holdout)

    @classmethod
    def load(cls, pool, settings, *, selection_seasons: int = SELECTION_SEASONS,
             holdout_seasons: int = HOLDOUT_SEASONS,
             selection_seed: int = SELECTION_SEED,
             holdout_seed: int = HOLDOUT_SEED,
             root: Optional[Path] = None, progress=None) -> "OutcomeBanks":
        kw = {"root": root} if root is not None else {}
        return cls(
            selection=load_or_build_cache(
                pool, settings, seasons=selection_seasons,
                seed=selection_seed, scenario="selection",
                progress=progress, **kw),
            holdout=load_or_build_cache(
                pool, settings, seasons=holdout_seasons, seed=holdout_seed,
                scenario="holdout", progress=progress, **kw))


def _completion_settings() -> CompletionSettings:
    """Sampling only. The beam fields are inert because offers are supplied.

    They still have to be *legal* values for the dataclass, so they are left at
    the library defaults rather than set to something small that might read as
    a tuning choice. Nothing on this path calls the beam.
    """
    return CompletionSettings(
        selection_sims=SELECTION_SEASONS, selection_seed=SELECTION_SEED,
        evaluation_sims=HOLDOUT_SEASONS, evaluation_seed=HOLDOUT_SEED)


@dataclass(frozen=True)
class LiveCEResult:
    """One next-bid decision, its interval, and the certificate behind it."""

    candidate_id: int
    candidate_name: str
    p: int
    q: int
    recipient: str
    recipient_name: str
    focus_owner_id: str
    mean_delta: float
    ci_low: float
    ci_high: float
    between_sd: float
    within_se: float
    k: int
    max_worlds: int
    selection_seasons: int
    holdout_seasons: int
    max_solver_gap: float
    n_with_candidate: int
    n_without_candidate: int
    opportunity_fingerprint: str
    state_fingerprint: str
    banks_fingerprint: str
    conservation_ok: bool
    league_ce_sums_ok: bool
    all_draws_agree: bool
    max_residual: float
    timings: Dict[str, float]
    runtime_s: float
    stale: bool = False
    notes: str = ""

    @property
    def verdict(self) -> str:
        """The whole interval, not the point estimate.

        BID and STOP each require the entire 95% interval on one side of zero.
        An interval that straddles zero is UNRESOLVED and stays UNRESOLVED --
        the market answer never substitutes for it.
        """
        if self.stale:
            return "STALE — ROOM CHANGED"
        if not (self.conservation_ok and self.league_ce_sums_ok
                and self.all_draws_agree):
            return "REFUSED — INVARIANT FAILED"
        if self.max_solver_gap > GAP_LIMIT:
            return "REFUSED — SOLVER GAP TOO WIDE"
        if math.isnan(self.ci_low) or math.isnan(self.ci_high):
            return "UNRESOLVED"
        if self.ci_low > 0:
            return "BID"
        if self.ci_high < 0:
            return "STOP"
        return "UNRESOLVED"

    def headline(self) -> str:
        return (
            f"{LIVE_LABEL}\n\n"
            f"Bid ${self.p} versus stop now:\n"
            f"{self.recipient_name} receives {self.candidate_name} for ${self.q}\n\n"
            f"Δ championship equity: {self.mean_delta:+.5f}\n"
            f"95% interval: [{self.ci_low:+.5f}, {self.ci_high:+.5f}]\n"
            f"{self.verdict}\n\n"
            f"K={self.k} allocation futures\n"
            f"{self.selection_seasons} selection / {self.holdout_seasons} holdout\n"
            f"completion gap: {self.max_solver_gap:.4f}\n"
            f"runtime: {self.runtime_s:.1f}s\n"
            f"state fingerprint: {self.state_fingerprint}\n\n"
            f"{LIVE_SCOPE}\n"
            f"Reduced-sample result; rerun when the bid or leader changes.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": LIVE_LABEL, "sublabel": LIVE_SUBLABEL, "scope": LIVE_SCOPE,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "p": self.p, "q": self.q, "recipient": self.recipient,
            "recipient_name": self.recipient_name,
            "focus_owner_id": self.focus_owner_id,
            "verdict": self.verdict,
            "mean_delta": round(self.mean_delta, 6),
            "ci95": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "between_allocation_sd": round(self.between_sd, 6),
            "within_allocation_se": round(self.within_se, 6),
            "k": self.k, "max_worlds": self.max_worlds,
            "selection_seasons": self.selection_seasons,
            "holdout_seasons": self.holdout_seasons,
            "max_solver_gap": round(self.max_solver_gap, 6),
            "n_with_candidate": self.n_with_candidate,
            "n_without_candidate": self.n_without_candidate,
            "opportunity_fingerprint": self.opportunity_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "banks_fingerprint": self.banks_fingerprint,
            "conservation_ok": self.conservation_ok,
            "league_ce_sums_ok": self.league_ce_sums_ok,
            "all_draws_agree": self.all_draws_agree,
            "max_residual": self.max_residual,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "runtime_s": round(self.runtime_s, 2),
            "stale": self.stale, "notes": self.notes,
        }


def prepare_offers(state: AuctionState, costs, proxy: ProxyEvaluator, *,
                   candidate_id: int, p: int, owner_id: Optional[str] = None,
                   time_limit_s: Optional[float] = None):
    """The certified sets for one price. Safe to run the moment a player is
    nominated, before anyone presses the button."""
    return build_certified_offers(
        state, costs, proxy, candidate_id=candidate_id, prices=[int(p)],
        pool_depth=OFFER_POOL_DEPTH, target=OFFER_TARGET, band=OFFER_BAND,
        owner_id=owner_id, time_limit_s=time_limit_s)


def _max_gap(offers) -> float:
    rows = (offers.provenance or {}).get("rows") or []
    gaps = [float(r.get("max_gap", 0.0)) for r in rows if "max_gap" in r]
    return max(gaps) if gaps else 0.0


def live_ce_next_bid(
    state: AuctionState,
    cast,
    costs,
    banks: OutcomeBanks,
    candidate_id: int,
    q: int,
    recipient: str,
    *,
    proxy: Optional[ProxyEvaluator] = None,
    market=None,
    key_by_id: Optional[Dict[int, str]] = None,
    name_by_id: Optional[Dict[int, str]] = None,
    team_names: Optional[Dict[str, str]] = None,
    offers=None,
    p: Optional[int] = None,
    board_settings: Optional[BoardSettings] = None,
    k: int = K_ROTATIONS,
    max_worlds: int = MAX_WORLDS,
) -> LiveCEResult:
    """Bidding ``q+1`` versus stopping now and letting ``recipient`` have him.

    ``STOP_NOW``: passing is not the player evaporating. It is that owner
    taking him at the bid he has already made, which spends his money and
    fills his slot. Our bid is ``q+1`` because that is what the next bid costs.
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    focus = state.focus_owner_id
    p = int(p) if p is not None else int(q) + 1

    if not state.is_available(candidate_id):
        raise AuctionRuleError(f"player {candidate_id} is not available")
    if recipient == focus:
        raise AuctionRuleError(
            "the high bidder must be an opponent; there is no buy-versus-stop "
            "question while we are already the leader")
    if recipient not in {o.owner_id for o in state.owners}:
        raise AuctionRuleError(f"no owner {recipient!r} in this room")
    bad = state.purchase_shortfall(candidate_id, focus, p)
    if bad is not None:
        raise AuctionRuleError(f"we cannot legally bid ${p}: {bad}")
    bad = state.purchase_shortfall(candidate_id, recipient, int(q))
    if bad is not None:
        raise AuctionRuleError(
            f"{recipient} cannot legally hold him at ${q}: {bad}. The stop "
            f"branch would not be a legal auction and is refused.")

    if proxy is None:
        proxy = ProxyEvaluator(state.pool, state.settings, 16, 7)

    t = time.perf_counter()
    banks.install()
    timings["bank_install_s"] = time.perf_counter() - t

    t = time.perf_counter()
    if offers is None:
        offers = prepare_offers(state, costs, proxy,
                                candidate_id=candidate_id, p=p)
    timings["certified_offers_s"] = time.perf_counter() - t

    gate = gate_certified_offers(offers)
    gap = _max_gap(offers)

    board_settings = board_settings or BoardSettings(pool_depth=400,
                                                     max_allocations=200)
    completion = _completion_settings()
    pass_price = PassPrice(mode=PassPriceMode.STOP_NOW, increment=1,
                           standing_price=int(q))

    t = time.perf_counter()
    draws = balanced_schedule(len(state.owners), k)
    ctxs = [build_eval_context(
        state, cast, costs, candidate_id, p=p, pass_price=pass_price,
        recipient=recipient, q=int(q), draw=d, board=board_settings,
        completion=completion, market=market, key_by_id=key_by_id, proxy=proxy,
        with_candidate=offers.with_candidate,
        without_candidate=offers.without_candidate,
        max_worlds=max_worlds) for d in draws]
    timings["context_build_s"] = time.perf_counter() - t

    t = time.perf_counter()
    rep = check_agreement(ctxs, proxy=proxy)
    timings["ce_evaluation_s"] = time.perf_counter() - t

    unc = _uncertainty(rep)
    ci = unc.get("t_interval_95") or [float("nan"), float("nan")]
    names = name_by_id or {}
    teams = team_names or {}
    conservation = all(b.conservation_ok for d in rep.draws
                       for b in d.branches.values())
    ce_sums = all(abs(b.league_ce_sum - 1.0) < 1e-9 for d in rep.draws
                  for b in d.branches.values())

    return LiveCEResult(
        candidate_id=candidate_id,
        candidate_name=names.get(candidate_id, str(candidate_id)),
        p=p, q=int(q), recipient=recipient,
        recipient_name=teams.get(recipient, recipient), focus_owner_id=focus,
        mean_delta=float(unc.get("mean_delta", float("nan"))),
        ci_low=float(ci[0]), ci_high=float(ci[1]),
        between_sd=float(unc.get("between_allocation_sd", float("nan"))),
        within_se=float(unc.get("rms_within_allocation_se") or float("nan")),
        k=k, max_worlds=max_worlds,
        selection_seasons=banks.selection.seasons,
        holdout_seasons=banks.holdout.seasons,
        max_solver_gap=gap,
        n_with_candidate=len(offers.with_candidate),
        n_without_candidate=len(offers.without_candidate),
        opportunity_fingerprint=offers.opportunity_fingerprint(),
        state_fingerprint=state.fingerprint(),
        banks_fingerprint=banks.fingerprint,
        conservation_ok=conservation, league_ce_sums_ok=ce_sums,
        all_draws_agree=rep.all_agree, max_residual=rep.max_residual,
        timings=timings, runtime_s=time.perf_counter() - t0,
        notes=("offer gate: " + ("ok" if gate.ok else ",".join(gate.failures))))
