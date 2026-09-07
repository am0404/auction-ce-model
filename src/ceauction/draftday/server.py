"""The local draft-day dashboard: stdlib HTTP, one page, no dependencies.

No framework, no Node, no database and no deployment. ``http.server`` serves a
single HTML page and a small JSON API from the same process that holds the
:class:`~ceauction.draftday.session.DraftSession`, so the page and the engine
can never disagree about the room.

Latency discipline, which is the whole point on a ten-second clock:

* board rows are recomputed **once per auction state** and cached under the
  session fingerprint, so search and sort are pure client-side work;
* the nomination panel reads only cached and exact quantities;
* the seven-second tactical proxy runs on a worker thread and the page shows
  ``CALCULATING`` until it lands -- it is never awaited on a request.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..auction.state import AuctionRuleError
from . import DRAFTDAY_DIR
from .board import opening_rows, write_opening_board
from .caps import BASES, DISAGREEMENT_LABEL, MARKET_LED_LABEL
from .panels import RECIPIENT_WARNING, nomination_panel, owner_table, pid, qb_panel
from .proxycache import ProxyCeilingCache
from .session import CONTINGENCY_TAGS, DraftSession, open_session
from ..tactical.realpilot import OWNER_IDS

from .board import DRAFTDAY_DIR
from .ceboard import (CEBOARD_FILENAME, ROUGH_DISCLOSURE, ROUGH_LABEL,
                       OpeningCEBoard, board_fingerprint, live_value)
from .sleepersync import (
    NEEDS_ATTENTION_BADGE,
    POLL_SECONDS,
    STATUS_OFF,
    SYNC_SCOPE_NOTE,
    SleeperClient,
    SleeperError,
    SleeperSync,
)

__all__ = ["DraftDayServer", "serve", "PAGE", "MANUAL_TRACKING_BADGE",
           "MANUAL_TRACKING_NOTE", "SYNC_BADGE", "DEFAULT_DRAFT_ID"]

#: The real draft this dashboard tracks. Read-only, public, no credentials.
DEFAULT_DRAFT_ID = "1386497531533357056"

SYNC_BADGE = "LIVE SLEEPER SYNC"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: What the header, the JSON state and the console all say about connectivity.
#: The dashboard can now read Sleeper's public feed, but only for *completed
#: sales*, only while the operator has switched it on, and only read-only. A
#: tool that silently lagged the real draft would be worse than no tool -- and
#: one that implied it could see live bidding would be worse still.
MANUAL_TRACKING_BADGE = "MANUAL TRACKING -- SLEEPER SYNC OFF"
MANUAL_TRACKING_NOTE = (
    "Live Sleeper sync is OFF by default and covers COMPLETED SALES ONLY. It "
    "reads Sleeper's public draft feed, holds no credentials, writes nothing "
    "and never sees a live bid, a nomination or the bid clock -- those are on "
    "Sleeper's private draft socket, which this tool does not touch. While "
    "sync is off the board reflects exactly what has been typed in. Manual "
    "sale entry and undo remain available at all times, including while sync "
    "is running.")


class DraftDayServer:
    """Session, caches and request handling, with one lock over the engines."""

    def __init__(self, session: DraftSession, *, enable_proxy: bool = True,
                 draft_id: str = DEFAULT_DRAFT_ID):
        self.session = session
        self.lock = threading.RLock()
        self.proxy = ProxyCeilingCache(session, enabled=enable_proxy)
        self._rows_cache: Tuple[str, List[Dict[str, object]]] = ("", [])
        self._opening_rows: List[Dict[str, object]] = []
        self.started_at = time.time()
        # Constructed, deliberately not enabled. Nothing reaches out to Sleeper
        # until the operator turns the toggle on.
        self.sync = SleeperSync(session, SleeperClient(draft_id),
                                owner_ids=OWNER_IDS)
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_stop = threading.Event()
        # The rough CE board, or None. Loaded once, only if its fingerprint
        # matches this room; a stale board is never shown, the dashboard says
        # CE PRECOMPUTE REQUIRED instead.
        self.ceboard: Optional[OpeningCEBoard] = None
        self.ceboard_status = "CE PRECOMPUTE REQUIRED"

    # --- the sync poll loop ------------------------------------------------

    def _sync_loop(self) -> None:
        while not self._sync_stop.wait(POLL_SECONDS):
            if not self.sync.enabled:
                return
            try:
                with self.lock:
                    self.sync.poll_once()
            except Exception as exc:  # a poller must never kill the dashboard
                self.sync.last_error = f"poll crashed: {exc}"

    def start_sync(self) -> Dict[str, object]:
        with self.lock:
            self.sync.enable()
            out = self.sync.poll_once()
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_stop.clear()
            self._sync_thread = threading.Thread(
                target=self._sync_loop, name="sleeper-sync", daemon=True)
            self._sync_thread.start()
        return out

    def stop_sync(self) -> Dict[str, object]:
        self._sync_stop.set()
        with self.lock:
            self.sync.disable()
            return self.sync.snapshot_status()

    # --- the rough CE board ------------------------------------------------

    def load_ceboard(self) -> str:
        """Load the precomputed board, but only for THIS room.

        The fingerprint covers the opening state, the pool, the league
        settings, the contexts, the seeds and the base-price distribution. A
        mismatch is not repaired and not partially used: the board is dropped
        and the dashboard reports that a precompute is required, because a
        price computed for a different pool is worse than no price at all.
        """
        from .board import market_band
        from .roughce import DEFAULT_WORLD_SEED
        from .ceboard import RANK_SEASONS, RANK_SEED
        board = self.session.board
        opening = board.state
        anchors = {p: (board.display_anchor_by_id.get(p)
                       or board.raw_anchor_by_id.get(p))
                   for p in board.key_by_id}
        covered = [p for p in board.key_by_id if (anchors.get(p) or 0) > 0]
        base = {p: market_band(board, p).base for p in covered}
        try:
            from .roughce import _pool_digest
            digest = _pool_digest(opening.pool, opening.settings)
        except Exception:
            digest = ""
        fp = board_fingerprint(opening, digest, base_prices=base,
                               seasons=RANK_SEASONS, seed=RANK_SEED)
        path = DRAFTDAY_DIR / CEBOARD_FILENAME
        loaded = OpeningCEBoard.load(path, fp)
        self.ceboard = loaded
        self.ceboard_status = ("LOADED" if loaded is not None
                               else "CE PRECOMPUTE REQUIRED")
        return self.ceboard_status

    def ce_for(self, player_id: int) -> Dict[str, object]:
        """Opening and live rough CE for one player, or MARKET ONLY.

        A player with no computed score gets no number invented for him: the
        status says MARKET ONLY and every CE field is absent.
        """
        if self.ceboard is None:
            return {"status": "CE PRECOMPUTE REQUIRED"}
        entry = self.ceboard.get(player_id)
        if entry is None:
            return {"status": "MARKET ONLY"}
        state = self.session.state
        legal = self._candidate_legal_max(player_id)
        guard = None
        for r in self.rows():
            if r.get("player_id") == str(player_id) or r.get("player_id") == player_id:
                guard = r.get("guardrail")
                break
        lv = live_value(entry, market=self.session.market,
                        position=entry.position, legal_max=legal,
                        guardrail=guard,
                        reconcile_multiplier=self._reconcile_multiplier())
        out = {"status": "OK", "label": ROUGH_LABEL}
        out.update(entry.to_dict())
        live = lv.to_dict()
        # The entry owns the suppression wording: a structurally-disagreeing
        # row must say so, not fall back to the generic noise message that
        # LiveValue emits for every LOW row.
        live.pop("message", None)
        out.update(live)
        return out

    def _ce_row(self, player_id: int) -> Dict[str, object]:
        """The CE columns for one board row.

        A LOW row carries no ``rough_ce_max``: the field is absent, not zero
        and not the raw consensus, so nothing downstream can render a noisy
        number as an actionable maximum.
        """
        from .ceboard import NOISY_MESSAGE, STRUCTURAL_MESSAGE
        if self.ceboard is None:
            return {"rough_ce_status": "CE PRECOMPUTE REQUIRED"}
        e = self.ceboard.get(player_id)
        if e is None:
            return {"rough_ce_status": "MARKET ONLY"}
        low = e.confidence == "LOW"
        return {
            "rough_ce_status": "OK",
            "rough_ce_confidence": e.confidence,
            "rough_ce_max": None if low else e.center,
            "rough_ce_low": None if low else e.low,
            "rough_ce_high": None if low else e.high,
            "rough_ce_message": (
                (STRUCTURAL_MESSAGE if e.structural_disagreement
                 else NOISY_MESSAGE) if low else ""),
            "rough_ce_suppression": (
                "STRUCTURAL DISAGREEMENT" if e.structural_disagreement
                else ("SEED/CONTEXT NOISE" if low else "")),
            "rough_ce_lean": e.lean if low else "",
        }

    def _candidate_legal_max(self, player_id: int) -> int:
        """Our exact legal maximum for this player, from the auction state."""
        state = self.session.state
        owner = state.owner(state.focus_owner_id)
        try:
            return int(owner.max_bid)
        except Exception:
            return int(max(1, owner.budget_remaining - owner.open_slots + 1))

    def _reconcile_multiplier(self) -> float:
        """Dollars still unspent, against dollars still needed to fill rosters.

        One ratio, computed from the room -- not a tuned coefficient. If the
        room has spent ahead of schedule the remaining players are chasing
        fewer dollars and this falls below one; if the room is holding money
        it rises above one.
        """
        state = self.session.state
        remaining = sum(o.budget_remaining for o in state.owners)
        slots = sum(o.open_slots for o in state.owners)
        if slots <= 0:
            return 1.0
        started = state.settings.budget * state.settings.n_teams
        total_slots = state.settings.roster_size * state.settings.n_teams
        expected = (started / total_slots) * slots
        if expected <= 0:
            return 1.0
        return max(0.25, min(4.0, remaining / expected))

    # --- board rows, cached per state --------------------------------------

    def rows(self) -> List[Dict[str, object]]:
        """Every board row for the current room. One recomputation per state."""
        fp = self.session.fingerprint()
        if self._rows_cache[0] == fp:
            return self._rows_cache[1]
        t0 = time.perf_counter()
        rows = opening_rows(self.session.board, state=self.session.state,
                            market=self.session.market,
                            overrides=self.session.override_map(),
                            opening_caps=self.session.opening_caps)
        out = []
        for r in rows:
            d = r.to_dict()
            d["player_id"] = pid(r.player_id)
            d["owner_team"] = (self.session.team_name(r.owner)
                               if r.owner else "")
            ce = self._ce_row(r.player_id)
            d.update(ce)
            out.append(d)
        self._rows_cache = (fp, out)
        self._last_rows_ms = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def bootstrap_opening(self) -> None:
        """Freeze the opening caps and write the opening board files."""
        rows = opening_rows(self.session.board, state=self.session.board.state,
                            market=self.session.board.market,
                            overrides=self.session.override_map())
        self.session.freeze_opening_caps(
            {r.player_id: r.provisional_cap for r in rows})
        write_opening_board(rows, coverage=self.session.board.coverage)
        # Build the proxy evaluator now. It is lazy, costs a few hundred
        # milliseconds over a 549-player pool, and the one place it must never
        # be built is the sale path -- which is exactly where it landed before
        # this line existed.
        self.session.board.proxy
        self.load_ceboard()
        self._opening_rows = []
        for r in rows:
            d = r.to_dict()
            d["player_id"] = pid(r.player_id)
            self._opening_rows.append(d)

    def opening(self) -> List[Dict[str, object]]:
        if not self._opening_rows:
            self.bootstrap_opening()
        return self._opening_rows

    def _resolve(self, raw: Optional[str]) -> Optional[int]:
        if raw in (None, "", "null"):
            return None
        try:
            value = int(str(raw))
        except (TypeError, ValueError):
            return None
        return value if value in self.session.state.spec_by_id else None

    # --- API ---------------------------------------------------------------

    def handle(self, method: str, path: str, query: Dict[str, List[str]],
               body: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        def one(name: str, default=None):
            vals = query.get(name)
            return vals[0] if vals else default

        with self.lock:
            if path == "/api/state" and method == "GET":
                return 200, self.state_payload()

            if path == "/api/board" and method == "GET":
                mode = one("mode", "live")
                rows = self.opening() if mode == "opening" else self.rows()
                return 200, {"rows": rows, "mode": mode,
                             "bases": list(BASES),
                             "market_led_label": MARKET_LED_LABEL}

            if path == "/api/nomination" and method == "GET":
                player_id = self._resolve(one("player_id"))
                if player_id is None:
                    return 400, {"error": "unknown or missing player_id"}
                bid = _int_or_none(one("bid"))
                leader = one("leader") or None
                by = one("by") or None
                ceiling, status = self.proxy.get(player_id, current_price=bid)
                panel = nomination_panel(
                    self.session, player_id, current_bid=bid,
                    high_bidder=leader, nominated_by=by,
                    proxy_ceiling=ceiling, proxy_status=status)
                panel["owners"] = owner_table(
                    self.session, candidate_id=player_id,
                    next_bid=panel["next_legal_bid"])
                panel["qb"] = qb_panel(
                    self.session, candidate_id=player_id,
                    next_bid=panel["next_legal_bid"],
                    provisional=panel["provisional_cap"])
                panel["proxy_detail"] = self.proxy.detail(player_id)
                return 200, panel

            if path == "/api/proxy" and method == "POST":
                player_id = self._resolve(body.get("player_id"))
                if player_id is None:
                    return 400, {"error": "unknown or missing player_id"}
                status = self.proxy.request(
                    player_id, current_price=_int_or_none(body.get("bid")))
                return 200, {"status": status}

            if path == "/api/ceboard" and method == "GET":
                who = one("player_id")
                if who:
                    resolved = self._resolve(who)
                    if resolved is None:
                        return 400, {"error": "unknown player_id"}
                    return 200, self.ce_for(resolved)
                return 200, {
                    "status": self.ceboard_status,
                    "label": ROUGH_LABEL, "disclosure": ROUGH_DISCLOSURE,
                    "coverage": (self.ceboard.coverage if self.ceboard else {}),
                    "fingerprint": (self.ceboard.fingerprint
                                    if self.ceboard else ""),
                    "seasons": (self.ceboard.seasons if self.ceboard else 0),
                    "reconcile_multiplier": round(self._reconcile_multiplier(), 4),
                }

            if path == "/api/sleeper" and method == "GET":
                return 200, self.sync.snapshot_status()

            if path == "/api/sleeper/enable" and method == "POST":
                try:
                    return 200, self.start_sync()
                except SleeperError as exc:
                    self.sync.last_error = str(exc)
                    return 200, self.sync.snapshot_status()

            if path == "/api/sleeper/disable" and method == "POST":
                return 200, self.stop_sync()

            if path == "/api/sleeper/now" and method == "POST":
                if not self.sync.enabled:
                    return 400, {"error": "turn LIVE SLEEPER SYNC on first"}
                return 200, self.sync.poll_once()

            if path == "/api/sleeper/ack" and method == "POST":
                # The operator has looked at the mismatch and fixed or accepted
                # it. Nothing is repaired here; automatic application re-arms.
                self.sync.clear_attention()
                return 200, self.sync.snapshot_status()

            if path == "/api/sale" and method == "POST":
                return self._sale(body)

            if path == "/api/undo" and method == "POST":
                record = self.session.undo()
                if record is None:
                    return 400, {"error": "there is no sale to undo"}
                return 200, {"undone": record.to_dict(),
                             "fingerprint": self.session.fingerprint()}

            if path == "/api/rename" and method == "POST":
                try:
                    self.session.rename_team(str(body.get("owner_id") or ""),
                                             str(body.get("name") or ""))
                except AuctionRuleError as exc:
                    return 400, {"error": str(exc)}
                return 200, {"ok": True}

            if path == "/api/override" and method == "POST":
                return self._override(body)

            if path == "/api/reset" and method == "POST":
                if not body.get("confirm"):
                    return 400, {"error": "reset needs an explicit confirmation"}
                self.session.reset()
                # The ledger authorises "already applied"; a reset room must
                # not inherit it, or the re-run would apply nothing.
                self.sync.forget()
                return 200, {"ok": True,
                             "fingerprint": self.session.fingerprint()}

            if path == "/api/export" and method == "GET":
                target = DRAFTDAY_DIR / f"snapshot_{int(time.time())}.json"
                self.session.export_snapshot(target)
                return 200, {"path": str(target),
                             "payload": self.session.to_payload()}

            if path == "/api/import" and method == "POST":
                return self._import(body)

            if path == "/api/log" and method == "GET":
                return 200, {"log": self.session.log[-200:][::-1]}

            if path == "/api/qb" and method == "GET":
                player_id = self._resolve(one("player_id"))
                return 200, qb_panel(self.session, candidate_id=player_id,
                                     next_bid=_int_or_none(one("bid")) or 1)

        return 404, {"error": f"no route {method} {path}"}

    # --- individual actions ------------------------------------------------

    def _sale(self, body: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        player_id = self._resolve(body.get("player_id"))
        if player_id is None:
            return 400, {"error": "unknown or missing player"}
        owner_id = str(body.get("owner_id") or "")
        raw = body.get("price")
        try:
            price = int(raw)
        except (TypeError, ValueError):
            return 400, {"error": f"a price must be a whole number; got {raw!r}"}
        problem = self.session.check_sale(player_id, owner_id, price)
        if problem is not None:
            return 400, {"error": problem, "rejected": True}
        t0 = time.perf_counter()
        record = self.session.record_sale(player_id, owner_id, price)
        rows_t = time.perf_counter()
        self.rows()  # warm the cache the page is about to ask for
        return 200, {
            "sale": record.to_dict(),
            "fingerprint": self.session.fingerprint(),
            "apply_ms": round((rows_t - t0) * 1000, 1),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def _override(self, body: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        player_id = self._resolve(body.get("player_id"))
        if player_id is None:
            return 400, {"error": "unknown or missing player"}
        fields: Dict[str, object] = {}
        if "dollar_adjustment" in body:
            try:
                fields["dollar_adjustment"] = int(body.get("dollar_adjustment") or 0)
            except (TypeError, ValueError):
                return 400, {"error": "the dollar adjustment must be a whole number"}
        for name in ("confidence_note", "contingency_tag", "linked_starter",
                     "reasoning"):
            if name in body:
                fields[name] = str(body.get(name) or "")
        if "takeover_share" in body:
            raw = body.get("takeover_share")
            if raw in (None, ""):
                fields["takeover_share"] = None
            else:
                try:
                    fields["takeover_share"] = float(raw)
                except (TypeError, ValueError):
                    return 400, {"error": "takeover share must be a fraction 0-1"}
        try:
            override = self.session.set_override(player_id, **fields)
        except (AuctionRuleError, ValueError) as exc:
            return 400, {"error": str(exc)}
        return 200, {"override": override.to_dict()}

    def _import(self, body: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        payload = body.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError as exc:
                return 400, {"error": f"that is not valid JSON: {exc}"}
        if not isinstance(payload, dict):
            path = body.get("path")
            if not path:
                return 400, {"error": "import needs a payload or a path"}
            try:
                payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return 400, {"error": f"could not read that snapshot: {exc}"}
        try:
            self.session.load_payload(payload)
            self.session.save()
        except (AuctionRuleError, ValueError, KeyError, TypeError) as exc:
            return 400, {"error": f"refused to import: {exc}"}
        return 200, {"ok": True, "sales": len(self.session.sales),
                     "fingerprint": self.session.fingerprint()}

    # --- the whole-room payload -------------------------------------------

    def state_payload(self) -> Dict[str, object]:
        session = self.session
        state = session.state
        board = session.board
        rows = self.rows()
        return {
            "fingerprint": session.fingerprint(),
            "auction_fingerprint": state.fingerprint(),
            "market_fingerprint": session.market.fingerprint(),
            "focus_owner_id": session.focus_owner_id,
            "owners": owner_table(session),
            "can_undo": session.can_undo,
            "sales": [s.to_dict() for s in session.sales][::-1][:40],
            "n_sales": len(session.sales),
            "counts": {
                "players": len(rows),
                "available": sum(1 for r in rows if r["status"] == "AVAILABLE"),
                "sold": sum(1 for r in rows if r["sold"]),
                "anchored": sum(1 for r in rows if r["anchored"]),
                "unanchored": sum(1 for r in rows if not r["anchored"]),
            },
            "market_observations": len(session.market.observations),
            "positional_levels": {
                k: {"n": v.n, "multiplier": round(v.multiplier, 4),
                    "additive": round(v.additive, 3)}
                for k, v in session.market.position_levels.items()},
            "room_level": {
                "n": session.market.room_effect.n,
                "multiplier": round(session.market.room_effect.multiplier, 4)},
            "overrides": session.override_map(),
            "contingency_tags": list(CONTINGENCY_TAGS),
            "bases": list(BASES),
            "recipient_warning": RECIPIENT_WARNING,
            "manual_tracking": True,
            "manual_tracking_badge": MANUAL_TRACKING_BADGE,
            "manual_tracking_note": MANUAL_TRACKING_NOTE,
            "live_sync_implemented": True,
            "sync_badge": SYNC_BADGE,
            "sync_scope_note": SYNC_SCOPE_NOTE,
            "sleeper": self.sync.snapshot_status(),
            "disagreement_label": DISAGREEMENT_LABEL,
            "proxy": self.proxy.stats(),
            "board_rows_ms": getattr(self, "_last_rows_ms", None),
            "coverage": board.coverage,
            "state_path": str(session.state_path),
            "uptime_s": round(time.time() - self.started_at, 1),
        }


def _int_or_none(raw) -> Optional[int]:
    if raw in (None, "", "null"):
        return None
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "ceauction-draftday/1.0"
    app: DraftDayServer = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:  # keep the console usable
        return

    def _send(self, code: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_page()
            return
        try:
            code, payload = self.app.handle(
                "GET", parsed.path, parse_qs(parsed.query), {})
        except Exception as exc:  # never take the tool down mid-auction
            code, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        self._send(code, payload)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
        except ValueError as exc:
            self._send(400, {"error": f"malformed request body: {exc}"})
            return
        try:
            code, payload = self.app.handle(
                "POST", parsed.path, parse_qs(parsed.query), body)
        except Exception as exc:
            code, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        self._send(code, payload)


def serve(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          open_browser: bool = True, enable_proxy: bool = True,
          session: Optional[DraftSession] = None,
          verbose: bool = True) -> None:
    """Start the dashboard and block. One command, one URL, no deployment."""
    if session is None:
        session = open_session(verbose=verbose)
    app = DraftDayServer(session, enable_proxy=enable_proxy)
    if verbose:
        print("freezing the opening board (written to "
              f"{DRAFTDAY_DIR}/opening_board.csv|json)...")
    app.bootstrap_opening()
    app.rows()

    handler = type("_BoundHandler", (_Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print("=" * 72)
    print("DRAFT DAY DASHBOARD")
    print("=" * 72)
    print(f"  open:      {url}")
    print(f"  players:   {len(app.rows())}  "
          f"({sum(1 for r in app.rows() if r['anchored'])} anchored, "
          f"{sum(1 for r in app.rows() if not r['anchored'])} unanchored)")
    print(f"  saved to:  {session.state_path}")
    print(f"  restored:  {len(session.sales)} sale(s)")
    print(f"  {MANUAL_TRACKING_BADGE}")
    print(f"  sleeper:   sync is OFF. draft {app.sync.client.draft_id}")
    print("             turn it on in the dashboard when the draft starts.")
    print("             read-only, no credentials; COMPLETED SALES ONLY, never live bids.")
    print("             manual entry and undo stay available the whole time.")
    print("  guardrails are MARKET/PROXY provisional unless a row says CE AUDITED.")
    print("  stop with Ctrl-C; the draft is saved after every accepted change.")
    print("=" * 72)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping. the draft is saved at", session.state_path)
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Draft Day</title>
<style>
 :root{
   --bg:#0f1216; --panel:#171b22; --panel2:#1e242e; --line:#2c3542;
   --ink:#e9eef5; --dim:#94a3b8; --dimmer:#64748b;
   --ok:#3fb950; --warn:#d9a441; --stop:#f0554b; --acc:#5aa2f0;
   --gray:#5b6675;
   --pad:14px;
 }
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
     "Helvetica Neue",Arial,sans-serif;
   -webkit-font-smoothing:antialiased}
 .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

 /* ---- header ---------------------------------------------------------- */
 header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
   padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);
   position:sticky;top:0;z-index:40}
 h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.01em;white-space:nowrap}
 .grow{flex:1 1 auto}
 .badge{display:inline-flex;align-items:center;gap:7px;
   background:#3a2a10;border:1px solid var(--warn);color:#f5cf85;
   padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:650;
   letter-spacing:.03em;white-space:nowrap}
 .badge .dot{width:8px;height:8px;border-radius:50%;background:var(--warn)}

 .badge.live{background:#0d2a1a;border-color:var(--ok);color:#9fe7bd}
 .badge.live .dot{background:var(--ok)}
 .badge.att{background:#3a1414;border-color:var(--stop);color:#ffb3ae}
 .badge.att .dot{background:var(--stop)}
 .badge.offb{background:var(--panel2);border-color:var(--line);color:var(--dim)}
 .badge.offb .dot{background:var(--dim)}
 #syncPanel.att{border-color:var(--stop);box-shadow:0 0 0 1px var(--stop) inset}
 .attbar{background:#3a1414;border:1px solid var(--stop);color:#ffd9d5;
   border-radius:8px;padding:10px 13px;margin-bottom:10px;font-weight:600}
 .attbar ul{margin:7px 0 0 18px;padding:0;font-weight:400;font-size:13px}
 .syncgrid{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-end}
 .syncgrid .k{color:var(--dim);font-size:11.5px;letter-spacing:.05em;
   text-transform:uppercase}
 .syncgrid .v{font-size:19px;font-weight:650;margin-top:3px}

 /* ---- controls -------------------------------------------------------- */
 button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
   border-radius:7px;padding:7px 13px;cursor:pointer;font:inherit;font-size:14px}
 button:hover{border-color:var(--acc)}
 button:disabled{opacity:.38;cursor:default}
 button.on{background:var(--acc);color:#08111f;border-color:var(--acc);font-weight:600}
 button.primary{background:var(--acc);color:#08111f;border-color:var(--acc);
   font-weight:650;font-size:15px;padding:9px 18px}
 button.primary:hover{filter:brightness(1.08)}
 button.quiet{background:transparent;color:var(--dim);border-color:transparent;
   padding:6px 9px;font-size:13px}
 button.quiet:hover{color:var(--ink);border-color:var(--line)}
 button.danger:hover{border-color:var(--stop);color:#ffb3ae}
 input,select,textarea{background:#0c1016;color:var(--ink);
   border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit;
   font-size:14px}
 input:focus,select:focus,textarea:focus{outline:none;border-color:var(--acc)}
 input.big{font-size:20px;font-weight:650;padding:8px 11px;width:110px;
   text-align:right}
 label{display:block;font-size:12px;color:var(--dim);margin:0 0 4px;
   letter-spacing:.02em}
 .field{display:flex;flex-direction:column}

 /* ---- layout ---------------------------------------------------------- */
 main{padding:var(--pad);display:grid;gap:var(--pad);
   grid-template-columns:1fr;align-items:start}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
   padding:var(--pad);min-width:0}
 .panel h2{margin:0 0 10px;font-size:12px;letter-spacing:.09em;color:var(--dim);
   text-transform:uppercase;font-weight:650}
 .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 .span{grid-column:1/-1}
 @media(min-width:1240px){
   main{grid-template-columns:minmax(0,2.7fr) minmax(280px,.9fr)}
 }

 /* ---- tables ---------------------------------------------------------- */
 .tablewrap{overflow:auto;border:1px solid var(--line);border-radius:8px;
   max-height:calc(100vh - 300px);min-height:220px}
 table{border-collapse:separate;border-spacing:0;width:100%;font-size:13.5px}
 th,td{padding:7px 7px;text-align:right;white-space:nowrap;
   border-bottom:1px solid #232b36}
 th{color:var(--dim);font-weight:600;font-size:11.5px;letter-spacing:.05em;
   text-transform:uppercase;position:sticky;top:0;z-index:2;
   background:var(--panel2);cursor:pointer;user-select:none;
   border-bottom:1px solid var(--line)}
 th:hover{color:var(--ink)}
 td.l,th.l{text-align:left}
 tbody tr{cursor:pointer}
 tbody tr:hover{background:#212a37}
 tbody tr.sel{background:#1d3350;box-shadow:inset 3px 0 0 var(--acc)}
 tbody tr.sel:hover{background:#22406a}
 tr.sold{opacity:.45}
 tr.us td{background:#152436}
 tr.us:hover td{background:#1b2e45}
 .name{font-weight:600}
 .num{font-variant-numeric:tabular-nums}

 /* ---- semantic colour ------------------------------------------------- */
 .c-below{color:var(--ok)} .c-in{color:var(--warn)} .c-above{color:var(--stop)}
 .c-none{color:var(--gray)}
 .BID{color:var(--ok)} .CAUTION{color:var(--warn)} .STOP{color:var(--stop)}
 .muted{color:var(--dim)} .dimmer{color:var(--dimmer)}
 .warn{color:var(--warn)} .bad{color:var(--stop)} .good{color:var(--ok)}
 .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;
   border:1px solid var(--line);color:var(--dim);vertical-align:middle}
 .pill.gray{color:var(--gray);border-color:#333c48}

 /* ---- cards ----------------------------------------------------------- */
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
   gap:10px}
 .card{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
   padding:10px 12px;min-width:0}
 .card .k{color:var(--dim);font-size:11.5px;letter-spacing:.03em;
   text-transform:uppercase}
 .card .v{font-size:22px;font-weight:650;margin-top:2px;
   font-variant-numeric:tabular-nums}
 .card .s{color:var(--dimmer);font-size:11.5px;margin-top:2px;
   white-space:normal;line-height:1.35}
 .card.hero .v{font-size:30px}

 .verdict{font-size:30px;font-weight:750;letter-spacing:.01em;line-height:1.15}
 .selname{font-size:24px;font-weight:700;line-height:1.2}

 .cebox{border:1px solid var(--line);border-radius:8px;padding:10px 12px;
   background:var(--panel2);margin-top:12px}
 .cebox.medium{border-color:var(--warn);background:#2b2210}
 .cebox.low{border-color:#4a3c44;background:#241a1e}
 .celabel{font-size:11.5px;font-weight:700;letter-spacing:.06em;
   text-transform:uppercase;line-height:1.35}
 .cebox.medium .celabel{color:#f5cf85}
 .cebox.low .celabel{color:#f0a99f}
 .cenum{font-size:28px;font-weight:700;margin-top:4px}
 .cediag{margin-top:8px;font-size:11.5px;color:var(--dimmer);
   display:flex;flex-wrap:wrap;gap:4px 16px}
 .cediag b{color:var(--fg);font-weight:600}
 .banner{background:#332510;border:1px solid var(--warn);color:#f2cf8b;
   padding:9px 12px;border-radius:8px;margin-bottom:10px;font-size:13.5px}
 .err{background:#331618;border:1px solid var(--stop);color:#ffb3ae;
   padding:9px 12px;border-radius:8px;margin-bottom:10px;font-size:13.5px}
 .ok{background:#14280f;border:1px solid var(--ok);color:#95e8a1;
   padding:9px 12px;border-radius:8px;font-size:13.5px}
 .note{color:var(--dim);font-size:12.5px;line-height:1.5}

 details.diag{background:var(--panel);border:1px solid var(--line);
   border-radius:10px;padding:10px var(--pad)}
 details.diag>summary{cursor:pointer;color:var(--dim);font-size:12px;
   letter-spacing:.09em;text-transform:uppercase;font-weight:650;
   list-style:none;padding:2px 0}
 details.diag>summary::-webkit-details-marker{display:none}
 details.diag>summary:before{content:"\25B8 ";color:var(--dimmer)}
 details.diag[open]>summary:before{content:"\25BE "}
 details.diag .body{margin-top:10px}

 dialog{background:var(--panel);color:var(--ink);border:1px solid var(--line);
   border-radius:12px;max-width:560px;width:92%;padding:0}
 dialog::backdrop{background:rgba(0,0,0,.55)}
 hr{border:none;border-top:1px solid var(--line);margin:12px 0}
</style></head><body>

<header>
 <h1>Draft Day</h1>
 <span class="badge offb" id="syncBadge"
   title="Completed Sleeper sales only. Never live bids.">
   <span class="dot"></span>SLEEPER SYNC: OFF &mdash; MANUAL TRACKING</span>
 <span class="grow"></span>
 <button id="mLive" class="on">Live</button>
 <button id="mOpen">Opening</button>
 <button id="undo">Undo last sale</button>
 <button id="exp" class="quiet">Export</button>
 <button id="imp" class="quiet">Import</button>
 <button id="rst" class="quiet danger">Reset</button>
</header>

<main>

 <!-- ======================= live sleeper sync ======================= -->
 <section class="panel span" id="syncPanel">
  <h2>Live Sleeper sync <span class="s" style="font-weight:400;color:var(--dim)">
    &mdash; completed sales only</span></h2>
  <div id="syncAtt"></div>
  <div class="syncgrid">
   <div>
    <button id="syncToggle">Turn LIVE SLEEPER SYNC on</button>
   </div>
   <div>
    <div class="k">Status</div>
    <div class="v" id="syncStatus">OFF</div>
   </div>
   <div>
    <div class="k">Last successful poll</div>
    <div class="v" id="syncPoll">never</div>
   </div>
   <div>
    <div class="k">Sleeper picks</div>
    <div class="v" id="syncPicks">&mdash;</div>
   </div>
   <div>
    <div class="k">Sales on this board</div>
    <div class="v" id="syncLocal">&mdash;</div>
   </div>
   <div>
    <button id="syncNow" class="quiet">Sync now</button>
    <button id="syncAck" class="quiet" style="display:none">Acknowledge &amp; re-arm</button>
   </div>
  </div>
  <div class="s" style="margin-top:10px;color:var(--dim);max-width:900px"
       id="syncNote"></div>
  <div class="s" style="margin-top:5px;color:var(--dim)" id="syncDraft"></div>
 </section>

 <!-- ======================= nomination + sale ======================= -->
 <section class="panel span" id="nomPanel">
  <h2>Current nomination &amp; sale</h2>
  <div class="row" style="gap:14px;align-items:flex-end;margin-bottom:12px">
   <div class="field" style="flex:1 1 260px;min-width:220px">
    <label for="nomSearch">Find a player (or click a row on the board)</label>
    <input id="nomSearch" placeholder="type a name..." autocomplete="off">
   </div>
   <div class="field" style="flex:0 1 260px">
    <label for="nomPick">Matches</label>
    <select id="nomPick"><option value="">(no search yet)</option></select>
   </div>
   <div class="field">
    <label for="nomBid">Current bid</label>
    <input id="nomBid" class="big" type="number" min="0" placeholder="0">
   </div>
   <div class="field" style="flex:0 1 180px">
    <label for="nomLeader">High bidder</label>
    <select id="nomLeader"><option value="">(nobody yet)</option></select>
   </div>
   <div class="field" style="flex:0 1 180px">
    <label for="nomBy">Nominated by</label>
    <select id="nomBy"><option value="">(not recorded)</option></select>
   </div>
   <button id="nomProxy">Run proxy ceiling</button>
  </div>

  <div id="nomOut" class="note">Click a player on the board to nominate him.</div>

  <hr>

  <div class="row" style="gap:14px;align-items:flex-end">
   <div style="flex:1 1 240px;min-width:200px">
    <div class="k muted" style="font-size:11.5px;letter-spacing:.03em;
      text-transform:uppercase">Recording a sale for</div>
    <div id="saleWho" class="selname c-none">no player selected</div>
   </div>
   <div class="field" style="flex:0 1 220px">
    <label for="saleOwner">Winning team</label>
    <select id="saleOwner"></select>
   </div>
   <div class="field">
    <label for="salePrice">Final price</label>
    <input id="salePrice" class="big" type="number" min="1" placeholder="$">
   </div>
   <button id="saleGo" class="primary">Record sale</button>
   <div id="saleMsg" class="note" style="flex:1 1 200px"></div>
  </div>
 </section>

 <!-- ============================ board ============================= -->
 <section class="panel" id="boardPanel">
  <h2>Player board</h2>
  <div class="row" style="margin-bottom:10px">
   <input id="q" placeholder="search the board" style="flex:1 1 180px;min-width:150px">
   <select id="fpos"><option value="">All positions</option>
     <option>QB</option><option>RB</option><option>WR</option><option>TE</option></select>
   <select id="fstat"><option value="available">Available</option>
     <option value="sold">Sold</option><option value="">All</option></select>
   <select id="fanch"><option value="">Priced + unpriced</option>
     <option value="1">Priced only</option><option value="0">Unpriced only</option></select>
   <span id="bcount" class="note"></span>
  </div>
  <div class="tablewrap"><table id="board">
   <thead><tr>
    <th class="l" data-s="name" title="Player name. Click any row to nominate him.">Player</th>
    <th class="l" data-s="position" title="Position">Pos</th>
    <th class="l" data-s="nfl_team" title="NFL team">Team</th>
    <th data-s="bye_week" title="Bye week">Bye</th>
    <th data-s="ppg" title="Projected points per game">PPG</th>
    <th data-s="sleeper_display_value"
        title="The price Sleeper displays, exactly as supplied. Not adjusted by this tool. This is the number the room is anchored on.">Sleeper</th>
    <th data-s="live_base"
        title="This model's format-adjusted expected clearing range (low-high) for THIS league. A different number from the Sleeper price, and never blended with it.">Model Range</th>
    <th data-s="lineup_improvement" style="max-width:90px"
        title="Weekly Lineup Gain: the increase in our mean weekly starting-lineup projection if he joins our roster. A proxy, not championship equity.">Lineup Gain</th>
    <th data-s="guardrail"
        title="Draft Guardrail: the working dollar figure to bid against. Where no converged tactical result exists this is max(Sleeper price, adjusted model high), clamped to our legal maximum -- the USER MARKET-ANCHOR POLICY. It is not a max bid.">Guardrail</th>
    <th class="l" data-s="guardrail_basis"
        title="Guardrail Basis: what evidence stands behind that number.">Basis</th>
    <th data-s="rough_ce_max"
        title="ROUGH CE MAX -- a heuristic starting point, NOT an audited CE result and not a max bid. CE ranks the players; the market price curve supplies the dollar scale. A LOW-confidence row shows no number on purpose.">Rough CE</th>
    <th class="l" data-s="rough_ce_confidence"
        title="How far the five ranking seeds and three roster contexts disagree. MEDIUM = a usable heuristic number. LOW = the spread is too wide, or CE and the market disagree structurally; use the market guardrail instead.">CE Conf</th>
    <th class="l" data-s="status" title="Availability, or who bought him and for how much.">Status</th>
   </tr></thead><tbody></tbody></table></div>
  <div class="note" style="margin-top:8px">
   <b>Sleeper</b> is the raw displayed price, untouched. <b>Model Range</b> is this
   league's format-adjusted range. They are different numbers and are never combined
   except by the guardrail policy, which says so where it applies.</div>
 </section>

 <!-- ======================= our money + QB ========================= -->
 <div style="display:grid;gap:var(--pad);align-content:start;min-width:0">
  <section class="panel">
   <h2>Our team</h2>
   <div class="cards" id="usCards"></div>
  </section>
  <section class="panel">
   <h2>QB scarcity (superflex, skill fallback permitted)</h2>
   <div id="qbOut" class="note">Select a player to see candidate-specific figures.</div>
  </section>
 </div>

 <!-- =========================== owners ============================= -->
 <section class="panel span">
  <h2>All twelve owners</h2>
  <div class="tablewrap" style="max-height:420px"><table id="owners"><thead><tr>
   <th class="l" title="Owner name. Click edit to rename; the underlying id never changes.">Team</th>
   <th title="Budget remaining">Budget</th>
   <th title="Dollars already spent">Spent</th>
   <th title="Players rostered">Rostered</th>
   <th title="Roster slots still open">Open slots</th>
   <th title="General legal maximum: the most this owner can bid on ANY player.">General max</th>
   <th title="Nominated-player legal maximum: the most this owner can bid on the player currently selected.">Max on this player</th>
   <th>QB</th><th>RB</th><th>WR</th><th>TE</th>
   <th class="l" title="Whether this owner can legally make the next bid on the selected player.">Can bid next?</th>
  </tr></thead><tbody></tbody></table></div>
 </section>

 <!-- ========================= diagnostics ========================== -->
 <details class="diag span" id="diag"><summary>Diagnostics &amp; transaction log</summary>
  <div class="body">
   <div class="banner" style="margin-bottom:12px">
    <b>Sleeper sync covers completed sales only &mdash; never live bids.</b>
    When LIVE SLEEPER SYNC is on, this dashboard polls Sleeper's public draft feed
    read-only, with no credentials and no writes, and records a sale after Sleeper has
    already awarded the player. Nominations, live bids and the bid clock exist only on
    Sleeper's private draft socket, which this tool does not touch: those are still
    entered by hand. With sync off, the board reflects exactly what has been typed in and
    nothing else.</div>
   <div class="cards" style="margin-bottom:12px" id="diagCards"></div>
   <div class="tablewrap" style="max-height:260px"><table id="log"><tbody></tbody></table></div>
  </div>
 </details>

</main>

<dialog id="ovDlg"><form method="dialog" id="ovForm" style="padding:18px">
 <h3 style="margin:0 0 6px" id="ovName"></h3>
 <div class="note" style="margin-bottom:8px">A manual override is your own note. It is
  labelled MANUAL OVERRIDE wherever it changes a number, and is never a model output.</div>
 <label>Dollar adjustment (signed, applied to the guardrail)</label>
 <input id="ovAdj" type="number" style="width:100%">
 <label style="margin-top:8px">Contingency tag</label><select id="ovTag" style="width:100%"></select>
 <label style="margin-top:8px">Linked starter</label><input id="ovStarter" style="width:100%">
 <label style="margin-top:8px">Estimated takeover share (0-1, blank if unknown)</label>
 <input id="ovShare" type="number" step="0.05" min="0" max="1" style="width:100%">
 <label style="margin-top:8px">Confidence note</label><input id="ovConf" style="width:100%">
 <label style="margin-top:8px">Reasoning</label><textarea id="ovWhy" rows="3" style="width:100%"></textarea>
 <div class="row" style="margin-top:14px">
  <button value="save" id="ovSave" class="primary">Save</button>
  <button value="cancel">Cancel</button>
  <span class="grow"></span><span id="ovMsg" class="note"></span>
 </div>
</form></dialog>

<script>
const $=s=>document.querySelector(s), fmt=v=>(v===null||v===undefined||v==='')?'-':v;
let S={}, ROWS=[], MODE='live', SEL=null, NOM=null, CE=null, sortKey='guardrail', sortDir=-1;

async function api(path,opts){const r=await fetch(path,opts||{});
  let j={}; try{j=await r.json()}catch(e){j={error:'unreadable response'}}
  if(!r.ok) j.__err=j.error||('HTTP '+r.status); return j;}
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b||{})});

const esc=t=>String(t===null||t===undefined?'':t).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function money(v){return (v===null||v===undefined)
  ?'<span class="c-none">&mdash;</span>':'$'+v;}
const BASIS_SHORT={
  'EXACT FINANCIAL/ROSTER ARITHMETIC':'Legal max',
  'MARKET PRIOR':'Market prior',
  'MARKET + LIVE SALES':'Market + sales',
  'PROXY/HEURISTIC':'Proxy',
  'CE AUDITED':'CE audited',
  'CE UNDERPOWERED':'CE underpowered',
  'SEARCH UNDERCONVERGED':'Search unconverged',
  'MANUAL OVERRIDE':'Manual'};
const shortBasis=b=>BASIS_SHORT[b]||b;

// The four states the colour vocabulary is allowed to express.
function decisionClass(d){
  if(d==='BELOW MARKET'||d==='BID') return 'c-below';
  if(d==='IN MARKET RANGE'||d==='CAUTION') return 'c-in';
  if(d==='ABOVE MARKET'||d==='STOP'||d==='OVER LEGAL MAXIMUM') return 'c-above';
  return 'c-none';
}

async function refresh(){
  S=await api('/api/state');
  const b=await api('/api/board?mode='+MODE); ROWS=b.rows||[];
  $('#undo').disabled=!S.can_undo;
  drawOwners(S.owners,null); drawBoard(); drawUs(); drawLog(); drawDiag();
  const sel=$('#saleOwner'), keep=sel.value;
  sel.innerHTML=S.owners.map(o=>`<option value="${o.owner_id}">${esc(o.team_name)}</option>`).join('');
  sel.value=keep||S.focus_owner_id;
  for(const id of ['#nomLeader','#nomBy']){const e=$(id),k=e.value;
    e.innerHTML='<option value="">'+(id==='#nomLeader'?'(nobody yet)':'(not recorded)')+'</option>'
      +S.owners.map(o=>`<option value="${o.owner_id}">${esc(o.team_name)}</option>`).join(''); e.value=k;}
  $('#ovTag').innerHTML=(S.contingency_tags||[]).map(t=>`<option>${esc(t)}</option>`).join('');
  drawSync(S.sleeper||{});
}

/* ---- live Sleeper sync -------------------------------------------------
   The server does the polling, so the sync keeps working with the browser
   closed. This loop only mirrors what the server already knows, and pulls a
   full refresh when the room actually changed -- so a completed Sleeper sale
   lands on the board with no reload. */
let SYNC={}, LAST_FP=null;

function drawSync(k){
  SYNC=k||{};
  const st=k.status||'OFF', att=!!k.needs_attention;
  $('#syncStatus').textContent=st;
  $('#syncStatus').className='v '+(att?'bad':(st==='CONNECTED'?'good':''));
  $('#syncPoll').textContent=k.last_poll_ok_at
    ? (new Date(k.last_poll_ok_at*1000).toLocaleTimeString()
       +' ('+k.last_poll_age_s+'s ago)') : 'never';
  $('#syncPicks').textContent=(k.n_sleeper_completed==null?'\u2014'
    :k.n_sleeper_completed+' completed, '+(k.n_sleeper_applied||0)+' applied');
  $('#syncLocal').textContent=(k.n_local_sales==null?'\u2014':k.n_local_sales);
  $('#syncToggle').textContent=k.enabled
    ? 'Turn LIVE SLEEPER SYNC off' : 'Turn LIVE SLEEPER SYNC on';
  $('#syncToggle').className=k.enabled?'on':'';
  $('#syncNow').disabled=!k.enabled;
  $('#syncAck').style.display=att?'':'none';
  $('#syncPanel').classList.toggle('att',att);

  const b=$('#syncBadge');
  b.className='badge '+(att?'att':(k.enabled?'live':'offb'));
  b.innerHTML='<span class="dot"></span>'+esc(att
    ? 'SYNC NEEDS ATTENTION \u2014 MANUAL CONTROLS STILL LIVE'
    : (k.enabled ? 'SLEEPER SYNC: '+st+' \u2014 COMPLETED SALES ONLY'
                 : 'SLEEPER SYNC: OFF \u2014 MANUAL TRACKING'));

  let note=esc(k.scope_note||'');
  if(k.last_error) note+=' <span class="bad">Last error: '+esc(k.last_error)+'</span>';
  $('#syncNote').innerHTML=note;
  const d=k.draft||{};
  $('#syncDraft').textContent=d.draft_id
    ? `draft ${d.draft_id} \u00b7 ${d.name||''} \u00b7 ${d.status||''} `
      +`\u00b7 ${d.type||''} \u00b7 ${d.teams||'?'} teams \u00b7 $${d.budget||'?'} `
      +`\u00b7 polling every ${k.poll_seconds}s (read-only, no credentials)`
    : 'not connected to Sleeper';

  const att_html=(k.attention||[]).length
    ? '<div class="attbar">SYNC NEEDS ATTENTION \u2014 automatic application is '
      +'STOPPED. Nothing was overwritten. Record sales by hand until this is '
      +'cleared.<ul>'+(k.attention||[]).map(a=>'<li>'+esc(a)+'</li>').join('')
      +'</ul></div>'
    : '';
  $('#syncAtt').innerHTML=att_html;
}

async function syncTick(){
  const k=await api('/api/sleeper');
  if(k.__err) return;
  drawSync(k);
  // The server may have applied a sale since the last look.
  if(k.enabled && k.n_local_sales!==undefined){
    const st=await api('/api/state');
    if(st.fingerprint && st.fingerprint!==LAST_FP){
      LAST_FP=st.fingerprint; await refresh(); if(SEL) selectPlayer(SEL);
    }
  }
}

function drawUs(){
  const us=(S.owners||[]).find(o=>o.is_us); if(!us) return;
  $('#usCards').innerHTML=`
   <div class="card hero"><div class="k">Our budget</div>
     <div class="v good">$${us.budget_remaining}</div>
     <div class="s">$${us.spent} spent</div></div>
   <div class="card hero"><div class="k">Open slots</div>
     <div class="v">${us.open_slots}</div>
     <div class="s">${us.roster_size} rostered</div></div>
   <div class="card"><div class="k">Our general max</div>
     <div class="v">$${us.general_max}</div>
     <div class="s">most we may bid on any player</div></div>
   <div class="card"><div class="k">Our roster</div>
     <div class="v" style="font-size:16px">QB ${us.QB} &middot; RB ${us.RB} &middot; WR ${us.WR} &middot; TE ${us.TE}</div>
     <div class="s">${S.counts.sold} of ${S.counts.players} players sold</div></div>`;
}

function drawBoard(){
  const q=$('#q').value.toLowerCase(), p=$('#fpos').value, st=$('#fstat').value,
        an=$('#fanch').value;
  let rows=ROWS.filter(r=>{
    if(q && r.name.toLowerCase().indexOf(q)<0) return false;
    if(p && r.position!==p) return false;
    if(st==='available' && r.sold) return false;
    if(st==='sold' && !r.sold) return false;
    if(an==='1' && !r.anchored) return false;
    if(an==='0' && r.anchored) return false;
    return true;});
  rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
    if(x===y)return 0; if(x===null||x===undefined)return 1; if(y===null||y===undefined)return -1;
    return (x>y?1:-1)*sortDir;});
  $('#bcount').textContent=`${rows.length} shown of ${ROWS.length}`;
  $('#board tbody').innerHTML=rows.slice(0,400).map(r=>{
    // An unpriced player gets no Sleeper number and no invented $1 band.
    const sleeper = r.sleeper_display_value===null||r.sleeper_display_value===undefined
      ? '<span class="c-none">&mdash;</span>'
      : '<b class="num">$'+r.sleeper_display_value+'</b>';
    const range = r.anchored
      ? `<span class="num">$${r.live_low}&ndash;$${r.live_high}</span>`
      : '<span class="pill gray">unpriced</span>';
    return `
   <tr class="${r.sold?'sold':''}${SEL===r.player_id?' sel':''}" data-id="${r.player_id}">
    <td class="l name">${esc(r.name)}</td>
    <td class="l">${r.position}</td><td class="l">${esc(r.nfl_team)||'-'}</td>
    <td class="num">${r.bye_week||'-'}</td>
    <td class="num">${r.ppg}</td>
    <td>${sleeper}</td>
    <td class="muted">${range}</td>
    <td class="num">${r.lineup_improvement===null||r.lineup_improvement===undefined
      ?'-':r.lineup_improvement.toFixed(1)}</td>
    <td><b class="num ${decisionClass(r.recommendation)}">${money(r.guardrail)}</b></td>
    <td class="l muted" style="font-size:12px" title="${esc(r.guardrail_basis)}${
      r.ce_audited?'':' -- CE NOT AUDITED'}">${esc(shortBasis(r.guardrail_basis))}${
      r.ce_audited?'':' <span class="pill gray">no CE</span>'}</td>
    <td>${ceCell(r)}</td>
    <td class="l" style="font-size:11.5px">${ceConfCell(r)}</td>
    <td class="l" style="font-size:12.5px">${r.sold
      ?('SOLD &middot; '+esc(r.owner_team)+' $'+r.sale_price)
      :(r.status==='AVAILABLE'?'<span class="muted">available</span>'
        :'<span class="c-none">'+r.status.toLowerCase()+'</span>')}</td>
   </tr>`;}).join('');
  document.querySelectorAll('#board tbody tr').forEach(tr=>{
    tr.onclick=()=>{selectPlayer(tr.dataset.id);};
    tr.ondblclick=e=>{e.preventDefault();openOverride(tr.dataset.id);};});
}

function drawOwners(owners,cand){
  $('#owners tbody').innerHTML=owners.map(o=>`<tr class="${o.is_us?'us':''}">
   <td class="l">${o.is_us?'<b>'+esc(o.team_name)+'</b> <span class="pill">us</span>'
     :esc(o.team_name)}
     <button class="quiet" data-ren="${o.owner_id}" style="padding:1px 6px;font-size:11px">edit</button></td>
   <td class="num">$${o.budget_remaining}</td><td class="num">$${o.spent}</td>
   <td class="num">${o.roster_size}</td>
   <td class="num">${o.open_slots}</td><td class="num">$${o.general_max}</td>
   <td class="num">${cand===null?'<span class="c-none">-</span>':'$'+o.candidate_max}</td>
   <td class="num">${o.QB}</td><td class="num">${o.RB}</td><td class="num">${o.WR}</td>
   <td class="num">${o.TE}</td>
   <td class="l" style="font-size:12.5px">${
     cand===null?'<span class="c-none">no player selected</span>'
     :(o.can_bid_candidate?'<span class="good">yes</span>'
       :'<span class="c-none">'+esc(o.blocked_reason||'no')+'</span>')}</td></tr>`).join('');
  document.querySelectorAll('[data-ren]').forEach(b=>b.onclick=async e=>{
    e.stopPropagation();
    const cur=S.owners.find(o=>o.owner_id===b.dataset.ren);
    const name=prompt('Team name for '+b.dataset.ren, cur?cur.team_name:'');
    if(name){await post('/api/rename',{owner_id:b.dataset.ren,name});
      await refresh(); if(SEL) selectPlayer(SEL);}});
}

function drawDiag(){
  $('#diagCards').innerHTML=`
   <div class="card"><div class="k">Session fingerprint</div>
     <div class="v mono" style="font-size:13px">${esc(S.fingerprint)}</div></div>
   <div class="card"><div class="k">Board recompute</div>
     <div class="v" style="font-size:16px">${S.board_rows_ms||0} ms</div></div>
   <div class="card"><div class="k">Market observations</div>
     <div class="v">${S.market_observations}</div>
     <div class="s">manually entered sales that moved a price level</div></div>
   <div class="card"><div class="k">Sales recorded</div>
     <div class="v">${S.n_sales}</div></div>
   <div class="card"><div class="k">Priced / unpriced</div>
     <div class="v" style="font-size:16px">${S.counts.anchored} / ${S.counts.unanchored}</div></div>
   <div class="card"><div class="k">Saved to</div>
     <div class="v mono" style="font-size:11px;word-break:break-all">${esc(S.state_path)}</div></div>`;
}

function drawLog(){
  api('/api/log').then(j=>{$('#log tbody').innerHTML=(j.log||[]).map(e=>{
    const t=new Date(e.at*1000).toLocaleTimeString();
    let d=e.kind.toUpperCase()+' ';
    if(e.kind==='sale') d+=`${esc(e.player)} -> ${esc(e.team)} $${e.price}`+(e.market_updated?'':' (market NOT moved: unpriced)');
    else if(e.kind==='undo') d+=`${esc(e.player)} $${e.price} reversed`;
    else if(e.kind==='rename') d+=`${esc(e.owner_id)} = ${esc(e.name)}`;
    else if(e.kind==='override') d+=`${esc(e.player_key)} adj ${e.dollar_adjustment}`;
    else d+=esc(JSON.stringify(e));
    return `<tr><td class="l muted" style="width:100px">${t}</td><td class="l">${d}</td></tr>`;}).join('');});
}

async function selectPlayer(id){
  SEL=id;
  const bid=$('#nomBid').value, leader=$('#nomLeader').value, by=$('#nomBy').value;
  const u=`/api/nomination?player_id=${id}`+(bid!==''?`&bid=${bid}`:'')
    +(leader?`&leader=${leader}`:'')+(by?`&by=${by}`:'');
  const n=await api(u);
  if(n.__err){$('#nomOut').innerHTML=`<div class="err">${esc(n.__err)}</div>`;return;}
  // The rough CE board is a separate, already-computed cache. Nothing is
  // recomputed by asking for it; a failure here must not blank the panel.
  const ce=await api(`/api/ceboard?player_id=${id}`);
  CE = ce && !ce.__err ? ce : null;
  NOM=n; drawNom(n); drawOwners(n.owners,id); drawQB(n.qb); drawBoard();
  $('#saleWho').innerHTML=`${esc(n.name)} <span class="pill">${n.position} ${esc(n.nfl_team)}</span>`;
  $('#saleWho').classList.remove('c-none');
  const opt=[...$('#nomPick').options].find(o=>o.value===id);
  if(!opt){const o=document.createElement('option');o.value=id;o.textContent=n.name;
    $('#nomPick').prepend(o);} $('#nomPick').value=id;
  if(!$('#salePrice').value) $('#salePrice').value=n.next_legal_bid;
  const tr=document.querySelector(`#board tbody tr[data-id="${id}"]`);
  if(tr) tr.scrollIntoView({block:'nearest'});
}

// --- rough CE rendering ------------------------------------------------
// One rule governs every one of these: a LOW row NEVER renders a dollar
// figure. The server already withholds it (rough_ce_max is null, and
// live_rough_ce_shown is null), and nothing here reaches around that to
// print the raw consensus. LOW rows get the message and the market
// guardrail instead, which is the whole point of the confidence gate.
function ceCell(r){
  if(r.rough_ce_status!=='OK')
    return '<span class="c-none" title="'+esc(r.rough_ce_status||'')+'">&mdash;</span>';
  if(r.rough_ce_max===null||r.rough_ce_max===undefined)
    return '<span class="c-none" title="'+esc(r.rough_ce_message||'')
      .replace(/\n/g,' ')+'">&mdash;</span>';
  return '<b class="num warn" title="ROUGH CE WORKING MAX -- HEURISTIC, NOT AUDITED">$'
    +r.rough_ce_max+'</b>';
}
function ceConfCell(r){
  if(r.rough_ce_status!=='OK')
    return '<span class="c-none">'+esc((r.rough_ce_status||'').toLowerCase())+'</span>';
  if(r.rough_ce_confidence==='LOW')
    return '<span class="pill gray" title="'+esc(r.rough_ce_message||'')
      .replace(/\n/g,' ')+'">LOW &middot; '+esc(r.rough_ce_suppression||'')+'</span>';
  return '<span class="pill warn" title="ROUGH CE WORKING MAX -- HEURISTIC, NOT AUDITED">'
    +esc(r.rough_ce_confidence||'')+'</span>';
}

// The selected-player rough CE block. This is the mandatory surface: the
// board columns are a convenience, this is where the number, its label and
// the diagnostics that justify it all appear together.
function ceBlock(ce){
  if(!ce) return '';
  if(ce.status!=='OK'){
    const why = ce.status==='MARKET ONLY'
      ? 'No rough CE was computed for this player. The market guardrail is the only number here.'
      : 'The rough CE board is not loaded for this room.';
    return '<div class="cebox"><div class="celabel">Rough CE &mdash; '
      +esc(ce.status)+'</div><div class="s" style="margin-top:4px">'+esc(why)+'</div></div>';
  }
  const d = ce.diagnostics||{};
  const diag = '<div class="cediag">'
    +'<span>consensus median <b>'+money(d.consensus_median)+'</b></span>'
    +'<span>seed/context range <b>'+money(d.p20)+'&ndash;'+money(d.p80)+'</b></span>'
    +'<span>spread <b>$'+d.spread+'</b></span>'
    +'<span>seeds agreeing <b>'+d.seed_agreement+'/5</b></span>'
    +'<span>observations <b>'+d.n_observations+'</b></span>'
    +'<span>market base <b>'+money(d.market_base)+'</b></span>'
    +'<span>CE/market ratio <b>'+d.market_ratio+'</b></span>'
    +'<span>vs market <b>'+esc(d.direction||'')+'</b></span>'
    +'</div>';

  if(ce.confidence==='LOW'){
    // No actionable maximum is exposed. Not the consensus, not the live
    // value, not a midpoint -- the operator is sent to the market guardrail.
    const msg = (ce.message||'').split('\n').map(esc).join('<br>');
    return '<div class="cebox low">'
      +'<div class="celabel bad">'+msg+'</div>'
      +'<div class="s" style="margin-top:6px">No rough CE maximum is shown for this '
      +'player. Suppressed: <b>'+esc(ce.suppression_reason||'')+'</b>'
      +(ce.lean?' &middot; '+esc(ce.lean):'')+'. '
      +'Bid against the market guardrail above.</div>'
      +diag+'</div>';
  }
  const live = ce.live_rough_ce_shown;
  return '<div class="cebox medium">'
    +'<div class="celabel">ROUGH CE WORKING MAX &mdash; HEURISTIC, NOT AUDITED</div>'
    +'<div class="row" style="gap:22px;align-items:flex-end;margin-top:4px">'
    +'<div><div class="k">Live-adjusted</div><div class="cenum warn">'+money(live)+'</div>'
    +'<div class="s">'+money(ce.live_low)+'&ndash;'+money(ce.live_high)
    +(ce.clamped?' &middot; '+esc(ce.clamp_label||'clamped'):'')+'</div></div>'
    +'<div><div class="k">Opening</div><div class="cenum">'+money(ce.opening_rough_ce_max)+'</div>'
    +'<div class="s">'+money(ce.opening_low)+'&ndash;'+money(ce.opening_high)+'</div></div>'
    +'<div><div class="k">Confidence</div><div class="cenum">'+esc(ce.confidence)+'</div>'
    +'<div class="s">of MEDIUM / LOW</div></div>'
    +'</div>'
    +'<div class="s" style="margin-top:6px">CE ranks the players; the market price '
    +'curve supplies the dollar scale. This is not an audited CE result, not a '
    +'certified maximum and not a reservation price.</div>'
    +diag+'</div>';
}

function drawNom(n){
  const c=n.cap, r=c.rails, m=n.market, rec=n.recommendation||{};
  const sleeper = n.sleeper_display_anchor===null||n.sleeper_display_anchor===undefined
    ? '<span class="c-none">none</span>' : '$'+n.sleeper_display_anchor;
  const rails=[
   ['Our exact legal maximum', money(r.legal_max), 'EXACT FINANCIAL/ROSTER ARITHMETIC'],
   ['Sleeper psychological anchor', sleeper, 'RAW SLEEPER DISPLAYED PRICE -- UNADJUSTED'],
   ['Format-adjusted market range', m.anchored?`${money(m.low)} / ${money(m.base)} / ${money(m.high)}`:'UNPRICED', m.basis],
   ['Proxy permissive ceiling', r.proxy_status==='cached'?money(r.proxy_ceiling):r.proxy_status.toUpperCase(),'PROXY/HEURISTIC'],
   ['Cached CE bracket', r.ce_bracket?r.ce_bracket.join('-'):'none', r.ce_status==='none'?'no CE result for this state':r.ce_status],
   ['Manual adjustment', r.manual_adjustment?(r.manual_adjustment>0?'+':'')+r.manual_adjustment:'none','MANUAL OVERRIDE'],
  ];
  $('#nomOut').innerHTML=`
   ${n.disagreement?`<div class="banner">${esc(n.disagreement_label)} &mdash; adjusted market base ${money(m.base)} vs proxy ceiling ${money(r.proxy_ceiling)}</div>`:''}
   <div class="row" style="gap:14px;align-items:stretch">
    <div class="card" style="flex:1 1 260px">
     <div class="selname">${esc(n.name)}</div>
     <div class="s">${n.position} &middot; ${esc(n.nfl_team)} &middot; bye ${n.bye_week||'-'} &middot; ${n.ppg} ppg</div>
     <div class="verdict ${decisionClass(rec.decision)}" style="margin-top:8px">${esc(rec.decision)}</div>
     <div class="s">${esc(rec.detail||'')}</div>
     ${rec.ce_audited?'':'<div class="s warn" style="margin-top:5px"><b>CE NOT AUDITED</b> &mdash; no championship-equity result stands behind this. It is a market position, not advice to bid or stop.</div>'}
    </div>
    <div class="card"><div class="k">Next legal bid</div><div class="v">${money(n.next_legal_bid)}</div></div>
    <div class="card"><div class="k">Sleeper anchor</div><div class="v">${sleeper}</div>
      <div class="s">what the room sees. Untouched.</div></div>
    <div class="card"><div class="k">Model range</div>
      <div class="v" style="font-size:18px">${m.anchored?money(m.low)+'&ndash;'+money(m.high):'unpriced'}</div>
      <div class="s">format-adjusted for this league</div></div>
    <div class="card"><div class="k">${esc(rec.number_label||'Draft guardrail')}</div>
      <div class="v ${decisionClass(rec.decision)}">${money(rec.number)}</div>
      <div class="s">${esc(n.guardrail_label)}</div></div>
    <div class="card"><div class="k">Our legal max</div><div class="v">${money(n.our_legal_max)}</div>
      <div class="s">exact auction arithmetic</div></div>
    <div class="card"><div class="k">Weekly lineup gain / roster fit</div>
      <div class="v" style="font-size:17px">${fmt(n.lineup_improvement)} / ${fmt(n.roster_fit)}</div>
      <div class="s">PROXY/HEURISTIC</div></div>
   </div>
   ${ceBlock(CE)}
   ${(c.notes||[]).length?`<div class="banner" style="margin-top:10px">${c.notes.map(esc).join(' &middot; ')}</div>`:''}
   <div class="cards" style="margin-top:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">
    <div class="card"><div class="k">Guardrail basis</div>
     <table style="margin-top:6px;font-size:12.5px">${rails.map(x=>`<tr>
      <td class="l muted" style="white-space:normal">${x[0]}</td>
      <td><b>${x[1]}</b></td>
      <td class="l dimmer" style="font-size:11px;white-space:normal">${esc(x[2])}</td></tr>`).join('')}</table></div>
    <div class="card"><div class="k">Opponents able to bid ${money(n.next_legal_bid)}</div>
     <div class="v">${n.n_capable_at_next_bid}</div>
     <div class="s">at the guardrail ${money(rec.number)}: ${n.n_capable_at_cap}</div>
     <div class="s">${esc(n.capable_now.map(o=>o.team).join(', ')||'nobody')}</div></div>
    <div class="card"><div class="k">Most plausible recipients</div>
     <table style="margin-top:6px;font-size:12.5px">${(n.recipients||[]).map(b=>`<tr>
       <td class="l">${esc(b.team)}</td><td>${money(b.price)}</td>
       <td class="muted">w ${b.weight}</td></tr>`).join('')||'<tr><td class="muted">none</td></tr>'}</table>
     <div class="s warn" style="margin-top:6px">${esc(S.recipient_warning||'')}</div></div>
   </div>`;
}

function drawQB(q){
  if(!q){return;}
  const f=q.superflex_fallback;
  $('#qbOut').innerHTML=`
   ${q.replacement_warning?`<div class="banner">${esc(q.replacement_warning)}</div>`:''}
   <div class="cards">
    <div class="card"><div class="k">Startable QBs left</div>
      <div class="v">${q.startable_remaining}</div>
      <div class="s">of ${q.total_qbs_remaining} QBs remaining</div></div>
    <div class="card"><div class="k">Teams by QB count</div>
      <div class="v" style="font-size:15px">0:${q.counts_by_bucket['0']} &nbsp;1:${q.counts_by_bucket['1']}
        &nbsp;2:${q.counts_by_bucket['2']} &nbsp;3+:${q.counts_by_bucket['3+']}</div></div>
    <div class="card"><div class="k">Highest opposing max</div>
      <div class="v">${money(q.highest_opposing_candidate_max)}</div></div>
    <div class="card"><div class="k">Superflex skill fallback</div>
      ${f?`<div class="v" style="font-size:14px">QB +${f.best_available_qb.improvement}
        vs skill +${f.best_available_skill.improvement}</div>
        <div class="s">gap ${f.qb_minus_skill_improvement} &middot; PROXY/HEURISTIC</div>`:'<div class="v">-</div>'}</div>
   </div>
   <div class="note" style="margin-top:8px">${esc(q.startable_definition)}</div>`;
}

function openOverride(id){
  const r=ROWS.find(x=>x.player_id===id); if(!r) return;
  $('#ovName').textContent=r.name; $('#ovMsg').textContent='';
  $('#ovAdj').value=r.manual_adjustment||0; $('#ovDlg').dataset.id=id;
  $('#ovDlg').showModal();
}

$('#ovForm').onsubmit=async e=>{
  if(e.submitter && e.submitter.value!=='save') return;
  const j=await post('/api/override',{player_id:$('#ovDlg').dataset.id,
    dollar_adjustment:$('#ovAdj').value||0, contingency_tag:$('#ovTag').value,
    linked_starter:$('#ovStarter').value, takeover_share:$('#ovShare').value,
    confidence_note:$('#ovConf').value, reasoning:$('#ovWhy').value});
  if(j.__err){$('#ovMsg').textContent=j.__err;return;}
  await refresh(); if(SEL) selectPlayer(SEL);
};

// Record Sale always uses the player currently selected on the board. There is
// no second, unlabelled player dropdown to get out of step with it.
$('#saleGo').onclick=async()=>{
  if(!SEL){$('#saleMsg').innerHTML='<span class="bad">Click a player on the board first.</span>';return;}
  const price=parseInt($('#salePrice').value,10);
  if(!(price>=0)){$('#saleMsg').innerHTML='<span class="bad">Enter a final price.</span>';return;}
  const j=await post('/api/sale',{player_id:SEL,owner_id:$('#saleOwner').value,price});
  if(j.__err){$('#saleMsg').innerHTML=`<span class="bad">REJECTED: ${esc(j.__err)}</span>`;return;}
  $('#saleMsg').innerHTML='<span class="ok">Sale recorded. Board and all twelve owners updated.</span>';
  $('#nomBid').value=''; $('#salePrice').value=''; SEL=null; NOM=null;
  $('#nomOut').innerHTML='<span class="note">Click a player on the board to nominate him.</span>';
  $('#saleWho').textContent='no player selected'; $('#saleWho').classList.add('c-none');
  await refresh();};

$('#undo').onclick=async()=>{const j=await post('/api/undo');
  $('#saleMsg').innerHTML=j.__err?`<span class="bad">${esc(j.__err)}</span>`
    :'<span class="ok">Last sale undone.</span>';
  await refresh(); if(SEL) selectPlayer(SEL);};
$('#rst').onclick=async()=>{
  if(!confirm('Reset the room to the OPENING state? Every recorded sale is cleared. '
    +'A backup of the current file is kept. Manual overrides are preserved.')) return;
  await post('/api/reset',{confirm:true}); SEL=null;
  $('#nomOut').innerHTML='<span class="note">Click a player on the board to nominate him.</span>';
  $('#saleWho').textContent='no player selected'; $('#saleWho').classList.add('c-none');
  await refresh();};
$('#exp').onclick=async()=>{const j=await api('/api/export');
  alert(j.__err?j.__err:('snapshot written to\n'+j.path));};
$('#imp').onclick=async()=>{const p=prompt('Path to a snapshot JSON file:');
  if(!p) return; const j=await post('/api/import',{path:p});
  alert(j.__err?('refused: '+j.__err):('imported '+j.sales+' sale(s)')); refresh();};
$('#nomProxy').onclick=async()=>{
  if(!SEL) return; await post('/api/proxy',{player_id:SEL,bid:$('#nomBid').value||null});
  const tick=setInterval(async()=>{const n=await api(`/api/nomination?player_id=${SEL}`
    +($('#nomBid').value!==''?`&bid=${$('#nomBid').value}`:''));
    if(n.cap && n.cap.rails.proxy_status!=='calculating'){clearInterval(tick);
      NOM=n;drawNom(n);drawQB(n.qb);} },1500);
  if(NOM){NOM.cap.rails.proxy_status='calculating';drawNom(NOM);}};

$('#nomPick').onchange=()=>{if($('#nomPick').value) selectPlayer($('#nomPick').value);};
$('#nomBid').oninput=()=>{if(SEL)selectPlayer(SEL);};
$('#nomLeader').onchange=()=>{if(SEL)selectPlayer(SEL);};
$('#nomBy').onchange=()=>{if(SEL)selectPlayer(SEL);};
$('#nomSearch').oninput=()=>{
  const q=$('#nomSearch').value.toLowerCase();
  const hits=q.length<2?[]:ROWS.filter(r=>!r.sold&&r.name.toLowerCase().indexOf(q)>=0).slice(0,40);
  $('#nomPick').innerHTML=hits.length
    ?hits.map(r=>`<option value="${r.player_id}">${esc(r.name)} (${r.position} ${esc(r.nfl_team)})</option>`).join('')
    :'<option value="">(no match)</option>';
  if(hits.length===1) selectPlayer(hits[0].player_id);};
['#q','#fpos','#fstat','#fanch'].forEach(s=>{$(s).oninput=drawBoard;$(s).onchange=drawBoard;});
document.querySelectorAll('#board th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.s; if(!k)return;
  if(sortKey===k) sortDir=-sortDir; else {sortKey=k;sortDir=-1;} drawBoard();});
$('#mLive').onclick=()=>{MODE='live';$('#mLive').classList.add('on');
  $('#mOpen').classList.remove('on');refresh();};
$('#mOpen').onclick=()=>{MODE='opening';$('#mOpen').classList.add('on');
  $('#mLive').classList.remove('on');refresh();};
$('#syncToggle').onclick=async()=>{
  const on=SYNC.enabled;
  if(!on && !confirm('Turn LIVE SLEEPER SYNC on?\n\nThis polls Sleeper\'s public '
      +'draft feed every 3 seconds, read-only, and applies COMPLETED SALES only. '
      +'It never sees live bids. Manual entry and undo stay available.')) return;
  const k=await api(on?'/api/sleeper/disable':'/api/sleeper/enable',{method:'POST'});
  if(k.__err){alert('sync: '+k.__err); return;}
  drawSync(k); await refresh();};
$('#syncNow').onclick=async()=>{
  const k=await api('/api/sleeper/now',{method:'POST'});
  if(k.__err){alert('sync: '+k.__err); return;}
  drawSync(k); await refresh(); if(SEL) selectPlayer(SEL);};
$('#syncAck').onclick=async()=>{
  if(!confirm('Acknowledge the mismatch and re-arm automatic application?\n\n'
     +'Nothing is repaired by this. Fix the board by hand FIRST.')) return;
  drawSync(await api('/api/sleeper/ack',{method:'POST'}));};
setInterval(syncTick, 2000);
refresh();
</script></body></html>
"""
