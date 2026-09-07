"""The live draft session: sales, undo, persistence and manual overrides.

The auction engine and the market updater are both immutable, so this layer
does not mutate anything -- it keeps a stack of whole
``(AuctionState, MarketState)`` snapshots. Undo is therefore *exact* by
construction rather than by careful reversal: it pops back to a pair of objects
that were never touched, and there is no field a hand-written inverse could
forget.

Persistence is the same idea. ``draft_state.json`` records the sales in order,
never derived quantities, so a restart replays them through the same two
engines and lands on the same fingerprint. Nothing derived is stored, so
nothing derived can go stale.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..auction.state import AuctionRuleError, AuctionState
from ..league import Position
from ..market.live import MarketState, SaleObservation, tier_of
from . import DRAFTDAY_DIR
from .board import DraftDayBoard, OWNER_IDS, load_board, market_band

__all__ = [
    "STATE_PATH",
    "OVERRIDES_PATH",
    "CONTINGENCY_TAGS",
    "SaleRecord",
    "ManualOverride",
    "DraftSession",
]

STATE_PATH = DRAFTDAY_DIR / "draft_state.json"
OVERRIDES_PATH = DRAFTDAY_DIR / "manual_overrides.json"

#: The only contingency tags the override form accepts. "none" is the default
#: and means exactly that -- not "no contingency exists", but "none recorded".
CONTINGENCY_TAGS: Tuple[str, ...] = (
    "none", "full handcuff", "partial takeover", "committee", "ambiguous")

STATE_VERSION = 1


@dataclass(frozen=True)
class SaleRecord:
    """One completed sale, exactly as the user entered it."""

    sequence: int
    player_id: int
    owner_id: str
    price: int
    recorded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return {"sequence": self.sequence, "player_id": self.player_id,
                "owner_id": self.owner_id, "price": self.price,
                "recorded_at": self.recorded_at}


@dataclass
class ManualOverride:
    """A local, visible correction the user typed. Never a model output."""

    player_key: str
    dollar_adjustment: int = 0
    confidence_note: str = ""
    contingency_tag: str = "none"
    linked_starter: str = ""
    takeover_share: Optional[float] = None
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.contingency_tag not in CONTINGENCY_TAGS:
            raise ValueError(
                f"contingency tag must be one of {CONTINGENCY_TAGS}, got "
                f"{self.contingency_tag!r}")
        if self.takeover_share is not None:
            share = float(self.takeover_share)
            if not 0.0 <= share <= 1.0:
                raise ValueError("takeover share is a fraction between 0 and 1")
            self.takeover_share = share
        self.dollar_adjustment = int(self.dollar_adjustment)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DraftSession:
    """Everything the dashboard reads and writes, in one place.

    Team names live here rather than inside :class:`AuctionState` on purpose:
    renaming a team is not an auction event, and folding it into the state
    stack would make undo revert a rename, which is not what anyone means by
    "undo the last sale".
    """

    def __init__(self, board: DraftDayBoard, *,
                 state_path: Path = STATE_PATH,
                 overrides_path: Path = OVERRIDES_PATH,
                 autosave: bool = True):
        self.board = board
        self.state_path = Path(state_path)
        self.overrides_path = Path(overrides_path)
        self.autosave = autosave

        self._stack: List[Tuple[AuctionState, MarketState]] = [
            (board.state, board.market)]
        self.sales: List[SaleRecord] = []
        self.team_names: Dict[str, str] = {
            o.owner_id: o.team_name for o in board.state.owners}
        self.team_names[board.focus_owner_id] = "US (our team)"
        self.overrides: Dict[str, ManualOverride] = {}
        self.log: List[Dict[str, object]] = []
        self._opening_caps: Dict[int, int] = {}
        self._load_overrides()

    # --- current position --------------------------------------------------

    @property
    def state(self) -> AuctionState:
        return self._stack[-1][0]

    @property
    def market(self) -> MarketState:
        return self._stack[-1][1]

    @property
    def focus_owner_id(self) -> str:
        return self.board.focus_owner_id

    @property
    def can_undo(self) -> bool:
        return len(self._stack) > 1

    def fingerprint(self) -> str:
        """Identity of the whole auction position.

        Both engines contribute, plus the overrides, because an override
        changes a displayed cap and a cache keyed without it could serve a
        number the user has already corrected.
        """
        ov = json.dumps({k: v.to_dict() for k, v in sorted(self.overrides.items())},
                        sort_keys=True)
        import hashlib
        h = hashlib.sha256()
        h.update(self.state.fingerprint().encode())
        h.update(self.market.fingerprint().encode())
        h.update(ov.encode())
        return h.hexdigest()[:16]

    def team_name(self, owner_id: str) -> str:
        return self.team_names.get(owner_id, owner_id)

    # --- opening caps, frozen ---------------------------------------------

    def freeze_opening_caps(self, caps: Dict[int, int]) -> None:
        """Record the opening caps once. They must not silently mutate."""
        if not self._opening_caps:
            self._opening_caps = dict(caps)

    @property
    def opening_caps(self) -> Dict[int, int]:
        return dict(self._opening_caps)

    # --- sales -------------------------------------------------------------

    def check_sale(self, player_id: int, owner_id: str,
                   price: int) -> Optional[str]:
        """Why this sale would be rejected, or ``None``.

        Every rule is the auction engine's own: duplicate sale, unknown player
        or owner, full roster, unaffordable, reserve violation and
        candidate-specific roster infeasibility all come back from
        ``purchase_shortfall``. Nothing is re-implemented here, so the
        dashboard and the engine cannot disagree about legality.
        """
        if not isinstance(price, int) or isinstance(price, bool):
            return f"a price must be whole auction dollars; got {price!r}"
        if owner_id not in self.state.owner_by_id:
            return f"no owner {owner_id!r} in the room"
        return self.state.purchase_shortfall(player_id, owner_id, price)

    def record_sale(self, player_id: int, owner_id: str,
                    price: int) -> SaleRecord:
        """Apply a sale to both engines, or raise with the reason."""
        problem = self.check_sale(player_id, owner_id, price)
        if problem is not None:
            raise AuctionRuleError(problem)

        state, market = self._stack[-1]
        key = self.board.key_for(player_id) or f"player-{player_id}"
        spec = state.spec(player_id)
        band = market_band(self.board, player_id, market)

        # Pre-sale room state, so a late sale can be read as a late sale.
        room_budget = sum(o.budget_remaining for o in state.owners)
        slots_left = sum(o.open_slots for o in state.owners)

        new_state = state.apply_purchase(player_id, owner_id, int(price))

        new_market = market
        if band.anchored:
            # Only an anchored player carries evidence about the anchor. An
            # unanchored sale has no prior to be a residual against, so folding
            # it in would be inventing the residual it claims to measure.
            obs = SaleObservation(
                player_key=key,
                position=Position(int(spec.position)).name,
                final_price=int(price), buyer=owner_id,
                sequence=len(market.observations),
                prior_base=band.base,
                display_anchor=self.board.display_anchor_by_id.get(player_id),
                tier=tier_of(band.base or 0),
                room_budget_remaining=room_budget, slots_remaining=slots_left)
            new_market = market.observe(obs)

        self._stack.append((new_state, new_market))
        record = SaleRecord(sequence=len(self.sales), player_id=player_id,
                            owner_id=owner_id, price=int(price))
        self.sales.append(record)
        self._append_log("sale", {
            "player_id": player_id, "player": self.board.name_by_id.get(player_id, ""),
            "owner_id": owner_id, "team": self.team_name(owner_id),
            "price": int(price),
            "market_updated": band.anchored,
            "market_note": ("" if band.anchored else
                            "UNANCHORED sale: no prior to form a residual "
                            "against, so the market was not moved"),
        })
        self.save()
        return record

    def undo(self) -> Optional[SaleRecord]:
        """Pop back to the exact prior snapshot of both engines."""
        if not self.can_undo:
            return None
        self._stack.pop()
        record = self.sales.pop()
        self._append_log("undo", {
            "player_id": record.player_id,
            "player": self.board.name_by_id.get(record.player_id, ""),
            "owner_id": record.owner_id, "price": record.price})
        self.save()
        return record

    # --- team names --------------------------------------------------------

    def rename_team(self, owner_id: str, name: str) -> None:
        if owner_id not in self.state.owner_by_id:
            raise AuctionRuleError(f"no owner {owner_id!r} in the room")
        name = str(name).strip()
        if not name:
            raise AuctionRuleError("a team name cannot be blank")
        self.team_names[owner_id] = name[:48]
        self._append_log("rename", {"owner_id": owner_id, "name": name[:48]})
        self.save()

    # --- overrides ---------------------------------------------------------

    def set_override(self, player_id: int, **fields) -> ManualOverride:
        key = self.board.key_for(player_id)
        if key is None:
            raise AuctionRuleError(f"player {player_id} is not on this board")
        current = self.overrides.get(key)
        data = current.to_dict() if current else {"player_key": key}
        for k, v in fields.items():
            if k in ("player_key",):
                continue
            data[k] = v
        override = ManualOverride(**data)
        if (override.dollar_adjustment == 0 and override.contingency_tag == "none"
                and not override.reasoning and not override.confidence_note
                and not override.linked_starter
                and override.takeover_share is None):
            self.overrides.pop(key, None)
        else:
            self.overrides[key] = override
        self._save_overrides()
        self._append_log("override", {"player_id": player_id,
                                      "player_key": key, **override.to_dict()})
        self.save()
        return override

    def override_map(self) -> Dict[str, Dict[str, object]]:
        return {k: v.to_dict() for k, v in self.overrides.items()}

    def _load_overrides(self) -> None:
        if not self.overrides_path.exists():
            return
        try:
            data = json.loads(self.overrides_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in data.get("overrides", []):
            try:
                ov = ManualOverride(**row)
            except (TypeError, ValueError):
                continue
            self.overrides[ov.player_key] = ov

    def _save_overrides(self) -> None:
        if not self.autosave:
            return
        self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "warning": "LOCAL ONLY -- manual entries about real players.",
            "label": ("These are the user's own corrections. They are never a "
                      "model output and are labelled MANUAL OVERRIDE wherever "
                      "they change a number."),
            "overrides": [v.to_dict() for v in self.overrides.values()],
        }
        _atomic_write(self.overrides_path, json.dumps(payload, indent=1))

    # --- transaction log ---------------------------------------------------

    def _append_log(self, kind: str, payload: Dict[str, object]) -> None:
        self.log.append({"at": time.time(), "kind": kind, **payload})

    # --- persistence -------------------------------------------------------

    def to_payload(self) -> Dict[str, object]:
        return {
            "version": STATE_VERSION,
            "warning": ("LOCAL ONLY -- real player-level auction record. "
                        "Gitignored; never commit."),
            "saved_at": time.time(),
            "board_fingerprint": self.board.prior.fingerprint(),
            "pool_fingerprint": self.board.state.pool_fingerprint(),
            "state_fingerprint": self.fingerprint(),
            "focus_owner_id": self.focus_owner_id,
            "team_names": dict(self.team_names),
            "sales": [s.to_dict() for s in self.sales],
            "overrides": [v.to_dict() for v in self.overrides.values()],
            "opening_caps": {str(k): v for k, v in self._opening_caps.items()},
            "log": self.log[-500:],
        }

    def save(self, path: Optional[Path] = None) -> Optional[Path]:
        if not self.autosave and path is None:
            return None
        target = Path(path) if path is not None else self.state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, json.dumps(self.to_payload(), indent=1))
        return target

    def load_payload(self, payload: Dict[str, object], *,
                     strict: bool = True) -> None:
        """Replay a saved session onto a fresh opening room.

        The pool digest is checked first. Replaying sales onto a board built
        from different sources would silently rebind every player id, so a
        mismatch refuses rather than producing a plausible wrong room.
        """
        saved_pool = payload.get("pool_fingerprint")
        if strict and saved_pool and saved_pool != self.board.state.pool_fingerprint():
            raise AuctionRuleError(
                "this saved draft was recorded against a different player pool "
                f"({saved_pool} vs {self.board.state.pool_fingerprint()}); "
                "refusing to replay it onto this board")

        self._stack = [(self.board.state, self.board.market)]
        self.sales = []
        self.log = list(payload.get("log") or [])
        names = payload.get("team_names") or {}
        for oid, nm in names.items():
            if oid in self.board.state.owner_by_id:
                self.team_names[oid] = str(nm)

        self.overrides = {}
        for row in payload.get("overrides", []):
            try:
                ov = ManualOverride(**row)
            except (TypeError, ValueError):
                continue
            self.overrides[ov.player_key] = ov

        self._opening_caps = {int(k): int(v)
                              for k, v in (payload.get("opening_caps") or {}).items()}

        for row in sorted(payload.get("sales", []),
                          key=lambda r: r.get("sequence", 0)):
            self.record_sale_silent(int(row["player_id"]), str(row["owner_id"]),
                                    int(row["price"]))

    def record_sale_silent(self, player_id: int, owner_id: str,
                           price: int) -> None:
        """Replay a sale without logging or autosaving it again."""
        was, self.autosave = self.autosave, False
        try:
            self.record_sale(player_id, owner_id, price)
            self.log.pop()
        finally:
            self.autosave = was

    def restore(self, path: Optional[Path] = None, *,
                strict: bool = True) -> bool:
        target = Path(path) if path is not None else self.state_path
        if not target.exists():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.load_payload(payload, strict=strict)
        return True

    def export_snapshot(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(self.to_payload(), indent=1))
        return path

    def import_snapshot(self, path: Path, *, strict: bool = True) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.load_payload(payload, strict=strict)
        self.save()

    def reset(self) -> None:
        """Back to the opening room. Sales and log go; overrides stay.

        Overrides survive because they are the user's own research about
        players, not a record of this auction, and re-typing them under time
        pressure is exactly the failure this tool exists to prevent.
        """
        if self.state_path.exists():
            backup = self.state_path.with_suffix(
                f".{int(time.time())}.bak.json")
            shutil.copy2(self.state_path, backup)
        self._stack = [(self.board.state, self.board.market)]
        self.sales = []
        self._append_log("reset", {"note": "room reset to opening state"})
        self.save()


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so a crash mid-save cannot
    truncate the only record of the draft."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def open_session(*, restore: bool = True, autosave: bool = True,
                 verbose: bool = False, **board_kwargs) -> DraftSession:
    """Load the board, build a session, and replay the last saved draft."""
    board = load_board(verbose=verbose, **board_kwargs)
    session = DraftSession(board, autosave=autosave)
    if restore:
        try:
            if session.restore() and verbose:
                print(f"restored {len(session.sales)} sale(s) from "
                      f"{session.state_path}")
        except (AuctionRuleError, ValueError) as exc:
            if verbose:
                print(f"could not restore the saved draft: {exc}")
    return session
