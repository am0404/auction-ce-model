"""The real opening board: every rosterable player, priced and capped.

This is Phase 1 of the draft-day delivery. It reuses the existing ingestion,
mapping and market layers verbatim through
:func:`ceauction.tactical.realpilot.load_real_board` -- there is no second
pipeline here and no re-derived projection.

What this module adds is the *board*: one row per rosterable player carrying
the market band, the exact opening legal maximum, the roster-fit and
lineup-improvement figures the existing proxy already computes, and the
provisional opening cap with the rail that bound it.

Everything it writes lands under ``local_data/draftday/``, which is gitignored.
Player-level output must never be committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.proxy import ProxyEvaluator
from ..auction.state import AuctionState
from ..league import DEFAULT_LEAGUE, LeagueSettings, Position
from ..market.live import MarketState
from ..market.prior import MarketPrior
from ..players import PlayerSpec
from . import DRAFTDAY_DIR
from .caps import (
    MARKET_LIVE,
    MARKET_PRIOR,
    CapRails,
    MarketBand,
    ProvisionalCap,
    provisional_cap,
)

__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_SLEEPER_CSV",
    "OWNER_IDS",
    "BoardRow",
    "DraftDayBoard",
    "load_board",
    "opening_rows",
    "write_opening_board",
    "market_band",
]

DEFAULT_CONTRACT = Path("local_data/real_player_contract_v1.json")
DEFAULT_SLEEPER_CSV = Path("local_data/sleeper_2qb_values_2026_clean.csv")

#: Twelve owners, reusing the ids the auction engine already builds rooms with
#: (``Team01``..``Team12``). ``Team01`` is us. The user renames them in the
#: dashboard, which changes the display name and never the id -- an id is what
#: every saved sale refers to.
from ..tactical.realpilot import OWNER_IDS  # noqa: E402

_CACHE = DRAFTDAY_DIR / "board_cache.pkl"
_CACHE_VERSION = 3


def _digest(*paths: Path) -> str:
    """Identity of the inputs, so a cache can never answer for other data."""
    h = hashlib.sha256()
    h.update(str(_CACHE_VERSION).encode())
    for p in paths:
        h.update(str(p).encode())
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Loading, with a cache so a crash recovery is seconds rather than a minute
# ---------------------------------------------------------------------------


@dataclass
class DraftDayBoard:
    """The opening room plus every lookup the dashboard needs by player id."""

    state: AuctionState
    prior: MarketPrior
    market: MarketState
    key_by_id: Dict[int, str]
    name_by_id: Dict[int, str]
    coverage: Dict[str, object]
    settings: LeagueSettings = DEFAULT_LEAGUE
    ppg_by_id: Dict[int, float] = field(default_factory=dict)
    points_by_id: Dict[int, float] = field(default_factory=dict)
    raw_anchor_by_id: Dict[int, Optional[float]] = field(default_factory=dict)
    display_anchor_by_id: Dict[int, Optional[int]] = field(default_factory=dict)
    _proxy: Optional[ProxyEvaluator] = None

    @property
    def focus_owner_id(self) -> str:
        return self.state.focus_owner_id

    @property
    def proxy(self) -> ProxyEvaluator:
        """Built once, lazily. Availability replicates are shared by every
        roster it scores, which is what makes the comparisons paired."""
        if self._proxy is None:
            self._proxy = ProxyEvaluator(self.state.pool, self.settings,
                                         n_reps=32, seed=20260906)
        return self._proxy

    def key_for(self, player_id: int) -> Optional[str]:
        return self.key_by_id.get(player_id)

    def id_for_key(self, key: str) -> Optional[int]:
        for pid, k in self.key_by_id.items():
            if k == key:
                return pid
        return None


def _build_board(contract: Path, sleeper_csv: Path,
                 pool_limit: Optional[int],
                 settings: LeagueSettings) -> DraftDayBoard:
    from ..tactical.realpilot import PilotInputs, load_real_board

    inputs = PilotInputs(contract=contract, sleeper_csv=sleeper_csv,
                         out_dir=DRAFTDAY_DIR,
                         pool_limit=pool_limit or 10_000)
    missing = inputs.missing()
    if missing:
        raise FileNotFoundError(
            "the real sources are not where the draft-day tool expects them: "
            + ", ".join(missing))
    rb = load_real_board(inputs, settings=settings)

    payload = json.loads(contract.read_text(encoding="utf-8"))
    by_key = {}
    for row in payload.get("players", []):
        k = row.get("player_key")
        if k:
            by_key[k] = row
    weeks = float(settings.regular_season_weeks)
    points, ppg = {}, {}
    for pid, key in rb.key_by_id.items():
        row = by_key.get(key) or {}
        pts = float((row.get("season_points") or {}).get("points") or 0.0)
        points[pid] = round(pts, 2)
        ppg[pid] = round(pts / 17.0, 2) if pts else 0.0

    raw, disp = {}, {}
    for pid, key in rb.key_by_id.items():
        pr = rb.prior.by_key.get(key)
        if pr is None:
            raw[pid], disp[pid] = None, None
            continue
        try:
            raw[pid] = float(pr.raw_value) if pr.raw_value is not None else None
        except (TypeError, ValueError):
            raw[pid] = None
        disp[pid] = pr.display_anchor

    return DraftDayBoard(state=rb.state, prior=rb.prior, market=rb.market,
                         key_by_id=dict(rb.key_by_id),
                         name_by_id=dict(rb.name_by_id),
                         coverage=dict(rb.coverage), settings=settings,
                         ppg_by_id=ppg, points_by_id=points,
                         raw_anchor_by_id=raw, display_anchor_by_id=disp)


def load_board(contract: Path = DEFAULT_CONTRACT,
               sleeper_csv: Path = DEFAULT_SLEEPER_CSV,
               *, pool_limit: Optional[int] = None,
               settings: LeagueSettings = DEFAULT_LEAGUE,
               use_cache: bool = True,
               verbose: bool = False) -> DraftDayBoard:
    """The opening board, from cache when the inputs are byte-identical.

    The cache key is a digest of the *contents* of both source files plus the
    pool limit, so editing a source invalidates it. A stale cache answering for
    changed data is the one failure mode a draft-day tool cannot survive, and a
    content digest removes it rather than mitigating it.
    """
    key = _digest(contract, sleeper_csv) + f"-{pool_limit}"
    if use_cache and _CACHE.exists():
        try:
            blob = pickle.loads(_CACHE.read_bytes())
            if blob.get("key") == key:
                if verbose:
                    print(f"opening board: cache hit ({_CACHE})")
                return blob["board"]
        except Exception:  # a damaged cache is rebuilt, never trusted
            pass
    t0 = time.perf_counter()
    if verbose:
        print("opening board: building from the real sources "
              "(this takes about a minute; the result is cached)...")
    board = _build_board(contract, sleeper_csv, pool_limit, settings)
    if use_cache:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_bytes(pickle.dumps({"key": key, "board": board}))
    if verbose:
        print(f"opening board: built in {time.perf_counter() - t0:.1f}s "
              f"({len(board.state.pool)} players)")
    return board


# ---------------------------------------------------------------------------
# The market band, at open and live
# ---------------------------------------------------------------------------


def market_band(board: DraftDayBoard, player_id: int,
                market: Optional[MarketState] = None) -> MarketBand:
    """Low / base / high for one player under the current market state.

    An unanchored player -- one the Sleeper list never priced -- comes back
    with ``anchored=False`` and no numbers at all. He is **not** given a $1
    band, because "nobody published a price" and "the market says one dollar"
    are different claims and only one of them is true.
    """
    key = board.key_for(player_id)
    prior = board.prior.by_key.get(key) if key else None
    if prior is None or not getattr(prior, "draftable", False):
        return MarketBand(None, None, None, anchored=False, basis=MARKET_PRIOR,
                          note="unanchored: no usable Sleeper anchor")
    state = market if market is not None else board.market
    moved = bool(state.observations) if state is not None else False
    low = base = high = None
    if state is not None:
        try:
            adj = state.adjusted(key)
            low, base, high = int(adj.low), int(adj.base), int(adj.high)
        except Exception:
            low = base = high = None
    if base is None:
        low, base, high = (int(prior.low_price), int(prior.base_price),
                           int(prior.high_price))
    return MarketBand(low=low, base=base, high=high, anchored=True,
                      basis=MARKET_LIVE if moved else MARKET_PRIOR)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class BoardRow:
    """One player's line on the board. Every field is displayable as-is."""

    player_id: int
    name: str
    position: str
    nfl_team: str
    bye_week: int
    season_points: float
    ppg: float
    sleeper_raw_value: Optional[float]
    sleeper_display_value: Optional[int]
    anchored: bool
    opening_low: Optional[int]
    opening_base: Optional[int]
    opening_high: Optional[int]
    live_low: Optional[int]
    live_base: Optional[int]
    live_high: Optional[int]
    roster_fit: Optional[float]
    lineup_improvement: Optional[float]
    legal_max: int
    opening_cap: int
    provisional_cap: int
    basis: str
    cap_label: str
    bound_by: str
    status: str
    sold: bool = False
    owner: str = ""
    sale_price: Optional[int] = None
    manual_adjustment: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


CSV_COLUMNS: Tuple[str, ...] = (
    "player_id", "name", "position", "nfl_team", "bye_week",
    "season_points", "ppg", "sleeper_raw_value", "sleeper_display_value",
    "anchored", "opening_low", "opening_base", "opening_high",
    "live_low", "live_base", "live_high", "roster_fit", "lineup_improvement",
    "legal_max", "opening_cap", "provisional_cap", "basis", "cap_label",
    "bound_by", "status", "sold", "owner", "sale_price", "manual_adjustment",
    "notes",
)


def _fit_and_improvement(board: DraftDayBoard, state: AuctionState,
                         ids: Sequence[int]) -> Tuple[Dict[int, float],
                                                      Dict[int, float]]:
    """Lineup improvement from adding each player at the minimum price.

    The improvement is the increase in the existing proxy's mean weekly
    starting-lineup projection when the player joins *our current roster*. At
    an empty opening room that is close to the player's own startable value; as
    the roster fills it correctly collapses toward zero for a position we can
    no longer start.

    ``roster_fit`` is the fraction of scoring weeks the player would actually
    occupy one of the eight seats on that roster. A fourth running back who can
    never start scores near zero here, with no positional rule anywhere.

    Neither number is value and neither is championship equity. They order a
    board; that is all they are for.
    """
    proxy = board.proxy
    ours = list(state.owner(state.focus_owner_id).player_ids)
    if not ids:
        return {}, {}
    base_strength = proxy.strength(ours) if ours else 0.0
    rosters = [ours + [pid] for pid in ids]
    strengths = proxy.strength_many(rosters)
    improvement = {int(pid): round(float(s) - base_strength, 3)
                   for pid, s in zip(ids, strengths)}
    fit: Dict[int, float] = {}
    for pid, roster in zip(ids, rosters):
        shares = proxy.lineup_shares(roster)
        fit[int(pid)] = round(float(shares.get(int(pid), 0.0)), 3)
    return fit, improvement


def opening_rows(board: DraftDayBoard, *,
                 state: Optional[AuctionState] = None,
                 market: Optional[MarketState] = None,
                 overrides: Optional[Dict[str, Dict[str, object]]] = None,
                 proxy_ceilings: Optional[Dict[int, int]] = None,
                 opening_caps: Optional[Dict[int, int]] = None,
                 with_fit: bool = True) -> List[BoardRow]:
    """Build every board row against a state and a market.

    Called once at open with the empty room, and again on every live refresh
    with the current room. The opening cap is passed back in through
    ``opening_caps`` so the live view can show it beside the live cap without
    the opening number silently mutating.
    """
    state = state or board.state
    market = market if market is not None else board.market
    overrides = overrides or {}
    proxy_ceilings = proxy_ceilings or {}
    spec_by_id = state.spec_by_id
    owner_of = state.owner_of
    price_of: Dict[int, int] = {t.player_id: t.price for t in state.transactions}
    focus = state.focus_owner_id
    our_max = state.owner(focus).max_bid

    ids = [s.player_id for s in state.pool]
    available = set(state.available_ids)
    if with_fit:
        fit, improvement = _fit_and_improvement(
            board, state, [p for p in ids if p in available])
    else:
        fit, improvement = {}, {}

    rows: List[BoardRow] = []
    for pid in ids:
        spec: PlayerSpec = spec_by_id[pid]
        key = board.key_for(pid) or ""
        band = market_band(board, pid, market)
        ov = overrides.get(key, {})
        adjustment = int(ov.get("dollar_adjustment") or 0)

        sold = pid in owner_of
        if sold:
            legal = 0
            status = "SOLD"
        elif pid in state.withdrawn:
            legal = 0
            status = "WITHDRAWN"
        else:
            # Exact arithmetic: our ceiling, and zero if this purchase could
            # not leave us a completable roster at any price.
            # Feasibility does not depend on price, and the money test is a
            # single comparison against max_bid, so the ceiling is max_bid
            # whenever the minimum bid is legal -- no descending scan needed.
            legal = (our_max if our_max > 0
                     and state.purchase_is_legal(pid, focus, our_max) else 0)
            status = "AVAILABLE" if legal > 0 else "INELIGIBLE"

        rails = CapRails(
            legal_max=legal, market=band,
            proxy_ceiling=proxy_ceilings.get(pid),
            proxy_status="cached" if pid in proxy_ceilings else "absent",
            manual_adjustment=adjustment,
            manual_note=str(ov.get("reasoning") or ""))
        cap: ProvisionalCap = provisional_cap(rails)

        notes = list(cap.notes)
        if ov.get("contingency_tag") and ov["contingency_tag"] != "none":
            notes.append(f"contingency: {ov['contingency_tag']}")
        elif spec.position == Position.RB:
            notes.append("HANDCUFF VALUE NOT MODELED")

        rows.append(BoardRow(
            player_id=pid, name=board.name_by_id.get(pid, spec.name),
            position=Position(int(spec.position)).name,
            nfl_team=spec.nfl_team, bye_week=int(spec.bye_week),
            season_points=board.points_by_id.get(pid, 0.0),
            ppg=board.ppg_by_id.get(pid, 0.0),
            sleeper_raw_value=board.raw_anchor_by_id.get(pid),
            sleeper_display_value=board.display_anchor_by_id.get(pid),
            anchored=band.anchored,
            opening_low=band.low if market is board.market else None,
            opening_base=band.base if market is board.market else None,
            opening_high=band.high if market is board.market else None,
            live_low=band.low, live_base=band.base, live_high=band.high,
            roster_fit=fit.get(pid), lineup_improvement=improvement.get(pid),
            legal_max=legal,
            opening_cap=int(opening_caps.get(pid, cap.cap))
            if opening_caps else cap.cap,
            provisional_cap=cap.cap, basis=cap.basis, cap_label=cap.label,
            bound_by=cap.bound_by, status=status, sold=sold,
            owner=owner_of.get(pid, ""), sale_price=price_of.get(pid),
            manual_adjustment=adjustment, notes="; ".join(notes)))

    rows.sort(key=lambda r: (-(r.opening_base or 0), -(r.season_points or 0.0)))
    return rows


def write_opening_board(rows: Sequence[BoardRow], out_dir: Path = DRAFTDAY_DIR,
                        *, coverage: Optional[Dict[str, object]] = None
                        ) -> Tuple[Path, Path]:
    """Write ``opening_board.csv`` and ``opening_board.json``.

    LOCAL ONLY. These carry real player names and must not be committed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "opening_board.csv"
    json_path = out_dir / "opening_board.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.to_dict()[k] for k in CSV_COLUMNS})

    anchored = sum(1 for r in rows if r.anchored)
    payload = {
        "generated_local_only": True,
        "warning": ("REAL PLAYER-LEVEL DATA. Gitignored. Never commit this "
                    "file."),
        "label": ("Provisional caps are market/proxy led unless the basis "
                  "column says CE AUDITED. They are not championship-equity "
                  "max bids."),
        "counts": {"players": len(rows), "anchored": anchored,
                   "unanchored": len(rows) - anchored},
        "coverage": coverage or {},
        "players": [r.to_dict() for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return csv_path, json_path
