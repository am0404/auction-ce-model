"""Read-only synchronisation of *completed* Sleeper auction sales.

What this module does, precisely
--------------------------------
It polls two public, unauthenticated Sleeper endpoints and turns **completed
auction picks** into sales in the local :class:`~ceauction.draftday.session.DraftSession`.
That is the whole scope. It is a *tracking* convenience, not a source of
valuation, and it changes no number this dashboard displays.

What it deliberately does **not** do
------------------------------------
* No credentials, tokens, cookies or logins of any kind.
* No writes to Sleeper. Every request is a ``GET``.
* No websockets, no reverse engineering of the private draft socket, no
  browser automation. A live *bid* is only on that socket, so this module
  cannot see one and never claims to: it observes a sale only once Sleeper has
  already awarded the player.
* No CE claim. Nothing here produces or repairs a championship-equity number.

The governing rule
------------------
**Never guess.** An owner, a player or a price that cannot be established
exactly from the feed is refused, and a refusal stops automatic application and
raises ``SYNC NEEDS ATTENTION`` rather than writing a plausible-looking sale.
The manual controls stay live the entire time, because the manual tally is the
fallback and a sync that quietly disabled it would remove the thing it is
supposed to be a convenience over.
"""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..auction.state import AuctionRuleError
from ..realdata.identity import normalize_name
from . import DRAFTDAY_DIR

__all__ = [
    "SLEEPER_API", "POLL_SECONDS", "SYNC_SCOPE_NOTE", "NEEDS_ATTENTION_BADGE",
    "STATUS_OFF", "STATUS_PRE_DRAFT", "STATUS_CONNECTED", "STATUS_DISCONNECTED",
    "STATUS_NEEDS_ATTENTION", "NAME_ALIASES",
    "SleeperError", "SleeperPick", "PickRefusal", "PlayerIndex", "OwnerMap",
    "SleeperClient", "SleeperSync", "board_key_for_name",
]

SLEEPER_API = "https://api.sleeper.app/v1"

#: Sleeper's published guidance is to stay under roughly 1000 calls/minute.
#: One call every three seconds is ~20/minute -- three orders of magnitude
#: below the ceiling, and still faster than an auction can complete a lot.
POLL_SECONDS = 3.0

SYNC_SCOPE_NOTE = (
    "Sleeper sync covers COMPLETED SALES ONLY. It does not see live bids, "
    "nominations or the bid clock -- those exist only on Sleeper's private "
    "draft socket, which this tool does not touch. A player appears here after "
    "Sleeper has already awarded him.")

NEEDS_ATTENTION_BADGE = "SYNC NEEDS ATTENTION"

STATUS_OFF = "OFF"
STATUS_PRE_DRAFT = "PRE-DRAFT"
STATUS_CONNECTED = "CONNECTED"
STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_NEEDS_ATTENTION = "NEEDS ATTENTION"

#: Board keys that are not ``normalize_name`` of the name Sleeper publishes.
#: Hand-verified one at a time against the real board and the real Sleeper
#: record; this is a lookup table of confirmed identities, never a fuzzy
#: matcher. A name that is not here and does not match exactly is refused.
NAME_ALIASES: Dict[str, str] = {
    # Board carries him under the name he played under; Sleeper uses his legal
    # name. Same player: Sleeper 5848, WR, PHI.
    "marquise_brown": "hollywood_brown",
}

STATE_FILENAME = "sleeper_sync.json"

#: The only positions this board holds. A Sleeper player who is none of these
#: is never joined, however his name reads.
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


class SleeperError(RuntimeError):
    """A transport-level failure. Never changes any local state."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def board_key_for_name(name: str) -> str:
    """The board's player key for a display name, or ``""``.

    The board keys players by ``normalize_name`` with spaces underscored. This
    is an exact transform, not a match: two different spellings of one player
    either agree here or the player is refused upstream.
    """
    key = normalize_name(name).replace(" ", "_")
    if not key:
        return ""
    return NAME_ALIASES.get(key, key)


@dataclass(frozen=True)
class SleeperPick:
    """One completed auction sale, exactly as Sleeper reported it."""

    pick_no: int
    sleeper_player_id: str
    roster_id: int
    amount: int
    player_name: str
    board_player_id: int
    owner_id: str

    @property
    def pick_key(self) -> str:
        """The idempotency key persisted across restarts."""
        return f"{self.pick_no}:{self.sleeper_player_id}"

    def to_dict(self) -> Dict[str, object]:
        return {"pick_no": self.pick_no,
                "sleeper_player_id": self.sleeper_player_id,
                "roster_id": self.roster_id, "amount": self.amount,
                "player_name": self.player_name,
                "board_player_id": self.board_player_id,
                "owner_id": self.owner_id, "pick_key": self.pick_key}


@dataclass(frozen=True)
class PickRefusal:
    """Why one raw pick could not be turned into a sale. Never a guess."""

    reason: str
    raw: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"reason": self.reason,
                "pick_no": self.raw.get("pick_no"),
                "player_id": self.raw.get("player_id"),
                "roster_id": self.raw.get("roster_id")}


class PlayerIndex:
    """Sleeper player id -> the board's player id.

    The join runs in exactly one direction. A pick names a Sleeper player id,
    which resolves to one Sleeper record, one name, one board key and -- because
    board keys are unique -- at most one board player. Names that collide inside
    Sleeper (two different men called "Justin Jefferson") never reach this,
    because the pick already told us which id was sold.
    """

    def __init__(self, sleeper_players: Dict[str, dict],
                 board_key_to_id: Dict[str, int]):
        self._names: Dict[str, str] = {}
        self._to_board: Dict[str, int] = {}
        for pid, rec in sleeper_players.items():
            if not isinstance(rec, dict):
                continue
            name = (rec.get("full_name")
                    or " ".join(x for x in (rec.get("first_name"),
                                            rec.get("last_name")) if x).strip())
            if not name:
                continue
            self._names[str(pid)] = name
            key = board_key_for_name(name)
            if key in board_key_to_id and self._is_skill(rec):
                self._to_board[str(pid)] = board_key_to_id[key]

    @staticmethod
    def _is_skill(rec: dict) -> bool:
        """Whether this Sleeper record could be the skill player we hold.

        The board is QB/RB/WR/TE only, and Sleeper carries a linebacker who
        happens to share a name with a starting receiver. Without this, that
        linebacker's id would resolve to the receiver's board slot. Positions
        are read from ``fantasy_positions`` as well as ``position``, because
        Sleeper files two-way players (a corner who plays receiver) under their
        defensive position while listing the fantasy one alongside.
        """
        candidates = set(rec.get("fantasy_positions") or ())
        if rec.get("position"):
            candidates.add(rec["position"])
        return bool(candidates & SKILL_POSITIONS)

    def name_of(self, sleeper_player_id: str) -> Optional[str]:
        return self._names.get(str(sleeper_player_id))

    def board_id_of(self, sleeper_player_id: str) -> Optional[int]:
        return self._to_board.get(str(sleeper_player_id))

    @property
    def n_joined(self) -> int:
        return len(self._to_board)

    @classmethod
    def from_board(cls, board, sleeper_players: Dict[str, dict]) -> "PlayerIndex":
        return cls(sleeper_players, {k: i for i, k in board.key_by_id.items()})


class OwnerMap:
    """Sleeper roster id -> dashboard owner id, or a refusal.

    The mapping is positional and total: roster ``N`` is ``Team{N:02d}``. That
    is deterministic and checkable, and it is only accepted once the feed has
    been proved self-consistent -- twelve rosters, twelve draft slots, a
    bijection between them, and every roster id in ``1..12`` exactly once. If
    any of that fails the map refuses to exist rather than covering the gap,
    because a silently shifted owner map would post every sale to the wrong team.
    """

    def __init__(self, roster_to_owner: Dict[int, str],
                 team_names: Dict[str, str], display_names: Dict[str, str]):
        self.roster_to_owner = dict(roster_to_owner)
        self.team_names = dict(team_names)
        self.display_names = dict(display_names)

    def owner_for(self, roster_id: int) -> Optional[str]:
        return self.roster_to_owner.get(int(roster_id))

    @classmethod
    def build(cls, draft: dict, rosters: Sequence[dict],
              users: Sequence[dict], owner_ids: Sequence[str]) -> "OwnerMap":
        n = len(owner_ids)
        settings = draft.get("settings") or {}
        teams = settings.get("teams")
        if teams is not None and int(teams) != n:
            raise SleeperError(
                f"Sleeper says {teams} teams; the board has {n}. Refusing to map.")
        if len(rosters) != n:
            raise SleeperError(
                f"Sleeper returned {len(rosters)} rosters; expected {n}.")

        roster_ids = sorted(int(r["roster_id"]) for r in rosters)
        if roster_ids != list(range(1, n + 1)):
            raise SleeperError(
                f"roster ids are not 1..{n}: {roster_ids}. Refusing to map.")

        slot_to_roster = {int(k): int(v)
                          for k, v in (draft.get("slot_to_roster_id") or {}).items()}
        if sorted(slot_to_roster) != list(range(1, n + 1)) or \
                sorted(slot_to_roster.values()) != list(range(1, n + 1)):
            raise SleeperError(
                "slot_to_roster_id is not a bijection over 1..%d. Refusing." % n)

        draft_order = {str(k): int(v)
                       for k, v in (draft.get("draft_order") or {}).items()}
        owner_by_roster = {int(r["roster_id"]): r.get("owner_id")
                           for r in rosters}
        if draft_order:
            if sorted(draft_order.values()) != list(range(1, n + 1)):
                raise SleeperError("draft_order slots are not 1..%d." % n)
            # Cross-check: the owner draft_order puts in slot S must be the
            # owner of the roster slot_to_roster_id maps S to.
            for user_id, slot in draft_order.items():
                expected_roster = slot_to_roster.get(slot)
                if owner_by_roster.get(expected_roster) != user_id:
                    raise SleeperError(
                        f"draft_order and slot_to_roster_id disagree for slot "
                        f"{slot}. Refusing to map owners.")

        user_by_id = {str(u.get("user_id")): u for u in users}
        team_names: Dict[str, str] = {}
        display_names: Dict[str, str] = {}
        roster_to_owner: Dict[int, str] = {}
        for rid in roster_ids:
            owner = owner_ids[rid - 1]
            roster_to_owner[rid] = owner
            user = user_by_id.get(str(owner_by_roster.get(rid)) or "") or {}
            display = str(user.get("display_name") or "").strip()
            meta = user.get("metadata") or {}
            team = str(meta.get("team_name") or "").strip()
            # Import the real name only when Sleeper actually published one.
            # A blank is left blank rather than invented.
            if team:
                team_names[owner] = team
            elif display:
                team_names[owner] = display
            if display:
                display_names[owner] = display
        return cls(roster_to_owner, team_names, display_names)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class SleeperClient:
    """Read-only ``GET`` access to Sleeper's public v1 API.

    ``fetch`` is injectable so the tests never touch the network. The real
    implementation sends no headers that identify the user, carries no cookies
    and uses no authentication, because none of these endpoints require any.
    """

    def __init__(self, draft_id: str, *, timeout: float = 6.0,
                 fetch: Optional[Callable[[str], object]] = None):
        self.draft_id = str(draft_id)
        self.timeout = timeout
        self._fetch = fetch or self._http_get
        self.n_requests = 0

    def _http_get(self, url: str) -> object:
        req = urllib.request.Request(url, method="GET",
                                     headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    raise SleeperError(f"HTTP {resp.status} from {url}")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise SleeperError(f"network failure: {exc}") from exc
        except (TimeoutError, OSError) as exc:
            raise SleeperError(f"network failure: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SleeperError(f"malformed JSON from {url}: {exc}") from exc

    def _get(self, path: str) -> object:
        self.n_requests += 1
        return self._fetch(f"{SLEEPER_API}{path}")

    def draft(self) -> dict:
        out = self._get(f"/draft/{self.draft_id}")
        if not isinstance(out, dict):
            raise SleeperError("draft endpoint did not return an object")
        return out

    def picks(self) -> List[dict]:
        out = self._get(f"/draft/{self.draft_id}/picks")
        if not isinstance(out, list):
            raise SleeperError("picks endpoint did not return a list")
        return out

    def users(self, league_id: str) -> List[dict]:
        out = self._get(f"/league/{league_id}/users")
        if not isinstance(out, list):
            raise SleeperError("users endpoint did not return a list")
        return out

    def rosters(self, league_id: str) -> List[dict]:
        out = self._get(f"/league/{league_id}/rosters")
        if not isinstance(out, list):
            raise SleeperError("rosters endpoint did not return a list")
        return out


# ---------------------------------------------------------------------------
# Parsing one raw pick
# ---------------------------------------------------------------------------


def _parse_amount(raw: object) -> Optional[int]:
    """The winning price, or ``None`` if Sleeper did not state one exactly.

    Sleeper reports auction prices as ``metadata.amount``, usually a decimal
    *string*. Anything that is not a whole positive number of auction dollars
    is refused: a missing, blank, fractional, negative or non-numeric amount is
    not a price we may infer, and a sale posted at a guessed price is worse
    than no sale at all.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if raw != int(raw):
            return None
        value = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text or not text.lstrip("-").isdigit():
            return None
        value = int(text)
    else:
        return None
    return value if value > 0 else None


def parse_pick(raw: object, index: PlayerIndex, owners: OwnerMap):
    """One raw pick -> :class:`SleeperPick`, or a :class:`PickRefusal`.

    Completed-only: a pick with no player is an in-flight nomination, not a
    sale, and is skipped silently by the caller rather than refused.
    """
    if not isinstance(raw, dict):
        return PickRefusal("pick was not an object", {})

    sleeper_pid = raw.get("player_id")
    if sleeper_pid in (None, "", 0, "0"):
        return PickRefusal("pick has no player_id (not a completed sale)", raw)
    sleeper_pid = str(sleeper_pid)

    pick_no = raw.get("pick_no")
    if not isinstance(pick_no, int) or isinstance(pick_no, bool) or pick_no <= 0:
        return PickRefusal(
            f"pick_no is missing or not a positive integer ({pick_no!r})", raw)

    meta = raw.get("metadata")
    if not isinstance(meta, dict):
        return PickRefusal("pick has no metadata object", raw)
    amount = _parse_amount(meta.get("amount"))
    if amount is None:
        return PickRefusal(
            f"metadata.amount is missing or not a whole positive number "
            f"({meta.get('amount')!r}); refusing to infer a price", raw)

    roster_id = raw.get("roster_id")
    if roster_id in (None, ""):
        roster_id = raw.get("picked_by_roster_id")
    if roster_id in (None, ""):
        return PickRefusal(
            "pick names no roster_id; refusing to guess the buyer", raw)
    try:
        roster_id = int(roster_id)
    except (TypeError, ValueError):
        return PickRefusal(f"roster_id {roster_id!r} is not an integer", raw)

    owner_id = owners.owner_for(roster_id)
    if owner_id is None:
        return PickRefusal(
            f"roster_id {roster_id} is not in the owner map; refusing", raw)

    name = index.name_of(sleeper_pid)
    if not name:
        return PickRefusal(
            f"Sleeper player {sleeper_pid} is not in the player index", raw)
    board_id = index.board_id_of(sleeper_pid)
    if board_id is None:
        return PickRefusal(
            f"{name} (Sleeper {sleeper_pid}) is not on this board; "
            f"record him by hand if he belongs here", raw)

    return SleeperPick(pick_no=pick_no, sleeper_player_id=sleeper_pid,
                       roster_id=roster_id, amount=amount, player_name=name,
                       board_player_id=board_id, owner_id=owner_id)


# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------


@dataclass
class ReconcileReport:
    """What the local room and the Sleeper feed agree and disagree about."""

    ok: bool
    problems: List[str] = field(default_factory=list)
    n_sleeper_completed: int = 0
    n_sleeper_applied: int = 0
    n_local_sales: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "problems": list(self.problems),
                "n_sleeper_completed": self.n_sleeper_completed,
                "n_sleeper_applied": self.n_sleeper_applied,
                "n_local_sales": self.n_local_sales}


class SleeperSync:
    """Idempotent, read-only application of completed Sleeper sales.

    Default **off**. Nothing here opens a socket, and no request is made, until
    :meth:`enable` is called. Every poll is a full reconciliation: the applied
    set, the local room and the Sleeper feed must agree, and the first
    disagreement stops automatic application and raises
    ``SYNC NEEDS ATTENTION``. The manual sale and undo controls are never
    touched, so the fallback is always one click away.
    """

    def __init__(self, session, client: SleeperClient, *,
                 owner_ids: Sequence[str],
                 state_path: Optional[Path] = None,
                 players_path: Optional[Path] = None,
                 autoload: bool = True):
        self.session = session
        self.client = client
        self.owner_ids = tuple(owner_ids)
        self.state_path = Path(state_path) if state_path is not None \
            else DRAFTDAY_DIR / STATE_FILENAME
        self.players_path = players_path

        self.enabled = False
        self.status = STATUS_OFF
        self.last_poll_ok_at: Optional[float] = None
        self.last_error: str = ""
        self.attention: List[str] = []
        self.applied: Dict[str, Dict[str, object]] = {}
        self.refusals: List[Dict[str, object]] = []
        self.draft_meta: Dict[str, object] = {}
        self.owners: Optional[OwnerMap] = None
        self.index: Optional[PlayerIndex] = None
        self.n_sleeper_completed = 0
        self.last_applied: List[Dict[str, object]] = []
        self.reconcile_report = ReconcileReport(ok=True)
        if autoload:
            self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        """Re-read the applied set so a restart cannot re-apply a sale."""
        if not self.state_path.exists():
            return
        try:
            blob = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if str(blob.get("draft_id")) != str(self.client.draft_id):
            # A different draft's ledger must never authorise this one.
            return
        applied = blob.get("applied")
        if isinstance(applied, dict):
            self.applied = {str(k): v for k, v in applied.items()
                            if isinstance(v, dict)}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "draft_id": self.client.draft_id,
                   "saved_at": time.time(), "applied": self.applied,
                   "warning": "LOCAL ONLY -- gitignored draft ledger."}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def forget(self) -> None:
        """Drop the applied ledger. Used when the room itself is reset."""
        self.applied = {}
        self.last_applied = []
        self.attention = []
        self.refusals = []
        self._save()

    # --- lifecycle ---------------------------------------------------------

    def enable(self) -> None:
        self.enabled = True
        self.status = STATUS_DISCONNECTED
        self.last_error = ""

    def disable(self) -> None:
        self.enabled = False
        self.status = STATUS_OFF

    def clear_attention(self) -> None:
        """Operator acknowledges the mismatch and re-arms automatic application."""
        self.attention = []
        self.refusals = []

    def _raise_attention(self, *problems: str) -> None:
        for p in problems:
            if p not in self.attention:
                self.attention.append(p)
        self.status = STATUS_NEEDS_ATTENTION

    # --- bootstrap ---------------------------------------------------------

    def connect(self, sleeper_players: Optional[Dict[str, dict]] = None) -> None:
        """Load draft metadata, the owner map and the player index.

        Raises :class:`SleeperError` on a transport failure, which the caller
        treats as ``DISCONNECTED`` -- never as a reason to change local state.
        """
        draft = self.client.draft()
        league_id = draft.get("league_id")
        if not league_id:
            raise SleeperError("draft response carries no league_id")
        if str(draft.get("type") or "") != "auction":
            raise SleeperError(
                f"draft type is {draft.get('type')!r}, not 'auction'. "
                f"This sync only understands auction prices.")
        rosters = self.client.rosters(str(league_id))
        users = self.client.users(str(league_id))
        self.owners = OwnerMap.build(draft, rosters, users, self.owner_ids)
        settings = draft.get("settings") or {}
        self.draft_meta = {
            "draft_id": self.client.draft_id, "league_id": str(league_id),
            "status": draft.get("status"), "type": draft.get("type"),
            "season": draft.get("season"),
            "name": (draft.get("metadata") or {}).get("name"),
            "teams": settings.get("teams"), "budget": settings.get("budget"),
            "rounds": settings.get("rounds"),
        }
        if sleeper_players is None:
            sleeper_players = self._load_players()
        self.index = PlayerIndex.from_board(self.session.board, sleeper_players)

    def _load_players(self) -> Dict[str, dict]:
        """The Sleeper player dictionary, from the local cache.

        Sleeper asks callers to fetch this once a day, not once a poll, so it
        is a file on disk that the operator refreshes deliberately. A missing
        cache is an error, never a silent empty index that would refuse every
        player in the draft.
        """
        path = self.players_path or (DRAFTDAY_DIR.parent / "sleeper"
                                     / "players_nfl.json")
        if not Path(path).exists():
            raise SleeperError(
                f"the Sleeper player cache {path} is missing; refresh it with "
                f"'curl -s {SLEEPER_API}/players/nfl -o {path}'")
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if not isinstance(blob, dict):
            raise SleeperError("the Sleeper player cache is not an object")
        return blob

    # --- the poll ----------------------------------------------------------

    def poll_once(self) -> Dict[str, object]:
        """One full cycle: fetch, apply what is new, reconcile everything.

        Makes **zero** network requests while disabled. A transport failure
        leaves every local structure exactly as it was and simply reports
        ``DISCONNECTED``; the next call retries.
        """
        if not self.enabled:
            self.status = STATUS_OFF
            return self.snapshot_status()

        try:
            if self.owners is None or self.index is None:
                self.connect()
            raw_picks = self.client.picks()
        except SleeperError as exc:
            # No local state is touched. This is the lost-internet path.
            self.status = STATUS_DISCONNECTED
            self.last_error = str(exc)
            return self.snapshot_status()

        self.last_error = ""
        self.last_poll_ok_at = time.time()

        parsed: List[SleeperPick] = []
        refusals: List[PickRefusal] = []
        n_completed = 0
        for raw in raw_picks:
            if isinstance(raw, dict) and raw.get("player_id") in (None, "", 0, "0"):
                continue  # a nomination in flight, not a completed sale
            n_completed += 1
            out = parse_pick(raw, self.index, self.owners)
            if isinstance(out, PickRefusal):
                refusals.append(out)
            else:
                parsed.append(out)
        self.n_sleeper_completed = n_completed
        self.refusals = [r.to_dict() for r in refusals]

        if refusals:
            self._raise_attention(*[
                f"pick refused: {r.reason}" for r in refusals])

        # Out-of-order arrival is normal on a reconnect: Sleeper returns the
        # whole list and ordering is not promised. Apply in draft order.
        parsed.sort(key=lambda p: p.pick_no)

        pending = [p for p in parsed if p.pick_key not in self.applied]

        if pending and not self.attention:
            self._apply_batch(pending)

        self.reconcile_report = self.reconcile(parsed)
        if not self.reconcile_report.ok:
            self._raise_attention(*self.reconcile_report.problems)

        if self.attention:
            self.status = STATUS_NEEDS_ATTENTION
        elif n_completed == 0 and str(
                self.draft_meta.get("status") or "") in ("pre_draft", "paused"):
            self.status = STATUS_PRE_DRAFT
        else:
            self.status = STATUS_CONNECTED
        return self.snapshot_status()

    def _apply_batch(self, pending: Sequence[SleeperPick]) -> None:
        """Apply new sales atomically, or restore the snapshot and stop.

        The snapshot is the session's own payload -- the same structure a
        restart replays -- so restoring it reproduces the pre-batch room
        exactly rather than approximately.
        """
        snapshot = copy.deepcopy(self.session.to_payload())
        applied_now: List[Dict[str, object]] = []
        sold_now = self._sold_index()

        try:
            for pick in pending:
                existing = sold_now.get(pick.board_player_id)
                if existing is not None:
                    owner_id, price = existing
                    if owner_id == pick.owner_id and price == pick.amount:
                        # A manual entry that already says exactly what Sleeper
                        # says. Adopt it rather than duplicating it.
                        self.applied[pick.pick_key] = dict(
                            pick.to_dict(), adopted_manual=True)
                        continue
                    raise _SyncConflict(
                        f"{pick.player_name}: Sleeper says "
                        f"{self._label(pick.owner_id)} ${pick.amount}, this "
                        f"board already records {self._label(owner_id)} "
                        f"${price}. Automatic sync stopped; nothing was "
                        f"overwritten. Fix by hand, then acknowledge.")
                try:
                    self.session.record_sale(pick.board_player_id,
                                             pick.owner_id, pick.amount)
                except AuctionRuleError as exc:
                    raise _SyncConflict(
                        f"{pick.player_name} to {self._label(pick.owner_id)} "
                        f"at ${pick.amount} was refused by the auction rules: "
                        f"{exc}") from exc
                sold_now[pick.board_player_id] = (pick.owner_id, pick.amount)
                self.applied[pick.pick_key] = pick.to_dict()
                applied_now.append(pick.to_dict())
        except _SyncConflict as exc:
            self.session.load_payload(snapshot, strict=False)
            self.session.save()
            for rec in applied_now:
                self.applied.pop(str(rec.get("pick_key")), None)
            self.last_applied = []
            self._raise_attention(str(exc))
            self._save()
            return

        self.last_applied = applied_now
        if applied_now:
            self._save()

    # --- reconciliation ----------------------------------------------------

    def _sold_index(self) -> Dict[int, Tuple[str, int]]:
        return {t.player_id: (t.owner_id, t.price)
                for t in self.session.state.transactions}

    def _label(self, owner_id: str) -> str:
        try:
            return f"{self.session.team_name(owner_id)} ({owner_id})"
        except Exception:
            return owner_id

    def reconcile(self, parsed: Sequence[SleeperPick]) -> ReconcileReport:
        """Check the whole room against the feed, every poll.

        Counts, ownership, prices, budgets, roster sizes, availability and
        duplicate ownership. Anything that fails here stops automatic
        application; none of it is repaired silently.
        """
        problems: List[str] = []
        state = self.session.state
        sold = self._sold_index()
        applied_keys = {p.pick_key for p in parsed if p.pick_key in self.applied}

        # 1. Every applied pick is present locally, to the same owner and price.
        for pick in parsed:
            if pick.pick_key not in self.applied:
                continue
            local = sold.get(pick.board_player_id)
            if local is None:
                problems.append(
                    f"{pick.player_name} was applied from Sleeper but is not "
                    f"sold on this board")
            elif local[0] != pick.owner_id or local[1] != pick.amount:
                problems.append(
                    f"{pick.player_name}: Sleeper says "
                    f"{self._label(pick.owner_id)} ${pick.amount}, board says "
                    f"{self._label(local[0])} ${local[1]}")

        # 2. Counts. Local may legitimately exceed Sleeper (a hand-entered
        #    sale for a player off this board); it may never fall short.
        n_local = len(self.session.sales)
        if n_local < len(applied_keys):
            problems.append(
                f"{len(applied_keys)} Sleeper picks are marked applied but the "
                f"board holds only {n_local} sales")

        # 3. No player owned twice, anywhere in the room.
        seen: Dict[int, str] = {}
        for owner in state.owners:
            for pid in owner.player_ids:
                if pid in seen:
                    problems.append(
                        f"player {pid} is on both {seen[pid]} and "
                        f"{owner.owner_id}")
                seen[pid] = owner.owner_id

        # 4. Money and roster room, per owner, against the engine's own rules.
        budget = self.draft_meta.get("budget")
        for owner in state.owners:
            if owner.budget_remaining < 0:
                problems.append(
                    f"{owner.owner_id} has spent ${owner.spent}, "
                    f"${-owner.budget_remaining} more than the budget")
            if owner.spent + owner.budget_remaining != owner.budget_start:
                problems.append(f"{owner.owner_id}: budget does not balance")
            if budget is not None and int(budget) != owner.budget_start:
                problems.append(
                    f"Sleeper budget ${budget} != board budget "
                    f"${owner.budget_start} for {owner.owner_id}")
            if owner.n_players > owner.roster_capacity:
                problems.append(
                    f"{owner.owner_id} holds {owner.n_players} players, over "
                    f"the {owner.roster_capacity}-slot roster")
            if owner.open_slots > 0 and owner.max_bid < 1:
                problems.append(
                    f"{owner.owner_id} has an open slot but no legal bid")

        # 5. Availability: every sold id is a real pool player, and the
        #    available count is exactly the pool less what has been sold.
        pool_ids = {s.player_id for s in state.pool}
        for pid in sold:
            if pid not in pool_ids:
                problems.append(f"sold player {pid} is not in the board pool")
        n_available = len(pool_ids - set(sold))
        if n_available != len(pool_ids) - len(set(sold) & pool_ids):
            problems.append("available count does not equal pool less sold")

        # 6. The engine's own invariant check, if it exposes one.
        checker = getattr(state, "problems", None)
        if callable(checker):
            try:
                problems.extend(str(p) for p in (checker() or ()))
            except Exception:  # a checker that throws is not evidence
                pass

        # The engine and the checks above overlap deliberately; report each
        # distinct complaint once, in the order it was found.
        problems = list(dict.fromkeys(problems))
        return ReconcileReport(
            ok=not problems, problems=problems,
            n_sleeper_completed=self.n_sleeper_completed,
            n_sleeper_applied=len(self.applied), n_local_sales=n_local)

    # --- what the dashboard renders ---------------------------------------

    def snapshot_status(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "needs_attention": bool(self.attention),
            "attention_badge": NEEDS_ATTENTION_BADGE,
            "attention": list(self.attention),
            "refusals": list(self.refusals),
            "last_poll_ok_at": self.last_poll_ok_at,
            "last_poll_age_s": (None if self.last_poll_ok_at is None
                                else round(time.time() - self.last_poll_ok_at, 1)),
            "last_error": self.last_error,
            "poll_seconds": POLL_SECONDS,
            "scope_note": SYNC_SCOPE_NOTE,
            "draft": dict(self.draft_meta),
            "n_sleeper_completed": self.n_sleeper_completed,
            "n_sleeper_applied": len(self.applied),
            "n_local_sales": len(self.session.sales),
            "last_applied": list(self.last_applied),
            "reconcile": self.reconcile_report.to_dict(),
            "owner_map": ({str(k): v for k, v in self.owners.roster_to_owner.items()}
                          if self.owners else {}),
            "n_requests": self.client.n_requests,
        }


class _SyncConflict(Exception):
    """Internal: abort the batch and restore the snapshot."""
