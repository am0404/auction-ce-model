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

__all__ = ["DraftDayServer", "serve", "PAGE"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class DraftDayServer:
    """Session, caches and request handling, with one lock over the engines."""

    def __init__(self, session: DraftSession, *, enable_proxy: bool = True):
        self.session = session
        self.lock = threading.RLock()
        self.proxy = ProxyCeilingCache(session, enabled=enable_proxy)
        self._rows_cache: Tuple[str, List[Dict[str, object]]] = ("", [])
        self._opening_rows: List[Dict[str, object]] = []
        self.started_at = time.time()

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
    print("  caps are MARKET/PROXY provisional unless a row says CE AUDITED.")
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
 :root{--bg:#11141a;--panel:#1a1f29;--line:#2b3444;--ink:#e6ebf2;--dim:#8d9bb0;
   --ok:#3fb950;--warn:#d29922;--stop:#f85149;--acc:#58a6ff;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
 header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
   padding:8px 12px;background:var(--panel);border-bottom:1px solid var(--line);
   position:sticky;top:0;z-index:20}
 h1{font-size:14px;margin:0;letter-spacing:.06em}
 .grow{flex:1}
 button{background:#222b38;color:var(--ink);border:1px solid var(--line);
   border-radius:4px;padding:4px 9px;cursor:pointer;font:inherit}
 button:hover{border-color:var(--acc)} button:disabled{opacity:.4;cursor:default}
 button.on{background:var(--acc);color:#08111f;border-color:var(--acc)}
 input,select,textarea{background:#0e1218;color:var(--ink);border:1px solid var(--line);
   border-radius:4px;padding:4px 6px;font:inherit}
 main{padding:10px;display:grid;gap:10px;grid-template-columns:1fr}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}
 .panel h2{margin:0 0 8px;font-size:12px;letter-spacing:.08em;color:var(--dim);
   text-transform:uppercase}
 .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{padding:3px 6px;text-align:right;border-bottom:1px solid #222a36;white-space:nowrap}
 th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--panel);
   cursor:pointer;user-select:none}
 td.l,th.l{text-align:left}
 tbody tr:hover{background:#20283a;cursor:pointer}
 tr.sold{opacity:.42}
 .scroll{max-height:60vh;overflow:auto;border:1px solid var(--line);border-radius:4px}
 .wide{overflow-x:auto}
 .pill{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px;
   border:1px solid var(--line);color:var(--dim)}
 .big{font-size:26px;font-weight:700;letter-spacing:.02em}
 .BID{color:var(--ok)} .CAUTION{color:var(--warn)} .STOP{color:var(--stop)}
 .muted{color:var(--dim)} .warn{color:var(--warn)} .bad{color:var(--stop)}
 .grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
 .card{background:#141922;border:1px solid var(--line);border-radius:5px;padding:7px 9px}
 .card .k{color:var(--dim);font-size:11px} .card .v{font-size:17px;font-weight:600}
 .banner{background:#2a1f10;border:1px solid var(--warn);color:#f0c674;
   padding:6px 9px;border-radius:4px;margin-bottom:8px}
 .err{background:#2b1416;border:1px solid var(--stop);color:#ffb3ae;
   padding:6px 9px;border-radius:4px;margin-bottom:8px}
 .ok{background:#12240f;border:1px solid var(--ok);color:#8ee79a;
   padding:6px 9px;border-radius:4px;margin-bottom:8px}
 dialog{background:var(--panel);color:var(--ink);border:1px solid var(--line);
   border-radius:8px;max-width:560px;width:92%}
 label{display:block;font-size:11px;color:var(--dim);margin:6px 0 2px}
 @media(min-width:1200px){main{grid-template-columns:1.15fr .85fr}
   .span{grid-column:1/-1}}
</style></head><body>
<header>
 <h1>DRAFT DAY</h1>
 <button id="mLive" class="on">LIVE VIEW</button>
 <button id="mOpen">OPENING VIEW</button>
 <span class="grow"></span>
 <span id="hdr" class="muted"></span>
 <button id="undo">UNDO LAST SALE</button>
 <button id="exp">EXPORT</button>
 <button id="imp">IMPORT</button>
 <button id="rst">RESET</button>
</header>
<main>
 <section class="panel span" id="nomPanel"><h2>Current nomination</h2>
  <div class="row">
   <input id="nomSearch" placeholder="type a player name..." style="min-width:230px">
   <select id="nomPick" style="min-width:230px"></select>
   <label style="margin:0">bid <input id="nomBid" type="number" min="0" style="width:74px"></label>
   <select id="nomLeader"><option value="">(high bidder)</option></select>
   <select id="nomBy"><option value="">(nominated by)</option></select>
   <button id="nomGo">SHOW</button>
   <button id="nomProxy">RUN PROXY CEILING</button>
  </div>
  <div id="nomOut" class="muted" style="margin-top:9px">Pick a player.</div>
 </section>

 <section class="panel span" id="salePanel"><h2>Record sale</h2>
  <div class="row">
   <select id="saleOwner" style="min-width:200px"></select>
   <label style="margin:0">price <input id="salePrice" type="number" min="1" style="width:80px"></label>
   <button id="saleGo">RECORD SALE</button>
   <span id="saleMsg" class="muted"></span>
  </div>
 </section>

 <section class="panel"><h2>Player board</h2>
  <div class="row" style="margin-bottom:7px">
   <input id="q" placeholder="search" style="min-width:170px">
   <select id="fpos"><option value="">all positions</option>
     <option>QB</option><option>RB</option><option>WR</option><option>TE</option></select>
   <select id="fstat"><option value="available">available</option>
     <option value="sold">sold</option><option value="">all</option></select>
   <select id="fanch"><option value="">anchored + unanchored</option>
     <option value="1">anchored only</option><option value="0">unanchored only</option></select>
   <span id="bcount" class="muted"></span>
  </div>
  <div class="scroll wide"><table id="board">
   <thead><tr>
    <th class="l" data-s="name">player</th><th data-s="position">pos</th>
    <th data-s="nfl_team">tm</th><th data-s="bye_week">bye</th>
    <th data-s="ppg">ppg</th><th data-s="live_base">mkt</th>
    <th data-s="live_low">band</th><th data-s="roster_fit">fit</th>
    <th data-s="lineup_improvement">+lineup</th><th data-s="legal_max">legal</th>
    <th data-s="provisional_cap">cap</th><th class="l" data-s="basis">basis</th>
    <th class="l" data-s="status">status</th>
   </tr></thead><tbody></tbody></table></div>
 </section>

 <section class="panel"><h2>Owners</h2>
  <div class="wide"><table id="owners"><thead><tr>
   <th class="l">team</th><th>budget</th><th>spent</th><th>ros</th><th>open</th>
   <th>gen max</th><th>cand max</th><th>QB</th><th>RB</th><th>WR</th><th>TE</th>
   <th class="l">can bid?</th></tr></thead><tbody></tbody></table></div>
 </section>

 <section class="panel span"><h2>QB scarcity (superflex, skill-position fallback permitted)</h2>
  <div id="qbOut" class="muted">Select a nomination to see candidate-specific figures.</div>
 </section>

 <section class="panel span"><h2>Transaction log</h2>
  <div class="scroll" style="max-height:200px"><table id="log"><tbody></tbody></table></div>
 </section>
</main>

<dialog id="ovDlg"><form method="dialog" id="ovForm" style="padding:14px">
 <h3 style="margin:0 0 4px" id="ovName"></h3>
 <div class="muted" style="font-size:11px">A manual override is your own note. It is
  labelled MANUAL OVERRIDE wherever it changes a number, and is never a model output.</div>
 <label>dollar adjustment (signed, applied to the provisional cap)</label>
 <input id="ovAdj" type="number" style="width:100%">
 <label>contingency tag</label><select id="ovTag" style="width:100%"></select>
 <label>linked starter</label><input id="ovStarter" style="width:100%">
 <label>estimated takeover share (0-1, blank if unknown)</label>
 <input id="ovShare" type="number" step="0.05" min="0" max="1" style="width:100%">
 <label>confidence note</label><input id="ovConf" style="width:100%">
 <label>reasoning</label><textarea id="ovWhy" rows="3" style="width:100%"></textarea>
 <div class="row" style="margin-top:10px">
  <button value="save" id="ovSave">SAVE</button>
  <button value="cancel">CANCEL</button>
  <span class="grow"></span><span id="ovMsg" class="muted"></span>
 </div>
</form></dialog>

<script>
const $=s=>document.querySelector(s), fmt=v=>(v===null||v===undefined||v==='')?'-':v;
let S={}, ROWS=[], MODE='live', SEL=null, NOM=null, sortKey='live_base', sortDir=-1;

async function api(path,opts){const r=await fetch(path,opts||{});
  let j={}; try{j=await r.json()}catch(e){j={error:'unreadable response'}}
  if(!r.ok) j.__err=j.error||('HTTP '+r.status); return j;}
const post=(p,b)=>api(p,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b||{})});

function money(v){return (v===null||v===undefined)?'-':'$'+v;}
function basisClass(b){return b&&b.indexOf('CE AUDITED')===0?'ok':'muted';}

async function refresh(){
  S=await api('/api/state');
  const b=await api('/api/board?mode='+MODE); ROWS=b.rows||[];
  $('#hdr').textContent=`${S.n_sales} sold | fp ${S.fingerprint} | mkt obs ${S.market_observations} | board ${S.board_rows_ms||0}ms`;
  $('#undo').disabled=!S.can_undo;
  drawOwners(S.owners,null); drawBoard(); drawLog();
  const sel=$('#saleOwner'), keep=sel.value;
  sel.innerHTML=S.owners.map(o=>`<option value="${o.owner_id}">${o.team_name}</option>`).join('');
  sel.value=keep||S.focus_owner_id;
  for(const id of ['#nomLeader','#nomBy']){const e=$(id),k=e.value;
    e.innerHTML='<option value="">'+(id==='#nomLeader'?'(high bidder)':'(nominated by)')+'</option>'
      +S.owners.map(o=>`<option value="${o.owner_id}">${o.team_name}</option>`).join(''); e.value=k;}
  $('#ovTag').innerHTML=(S.contingency_tags||[]).map(t=>`<option>${t}</option>`).join('');
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
  $('#bcount').textContent=`${rows.length} shown / ${ROWS.length} total`;
  $('#board tbody').innerHTML=rows.slice(0,400).map(r=>`
   <tr class="${r.sold?'sold':''}" data-id="${r.player_id}">
    <td class="l">${r.name}${r.anchored?'':' <span class="pill">unanchored</span>'}</td>
    <td>${r.position}</td><td>${r.nfl_team}</td><td>${r.bye_week||'-'}</td>
    <td>${r.ppg}</td><td>${money(r.live_base)}</td>
    <td class="muted">${r.anchored?money(r.live_low)+'-'+money(r.live_high):'-'}</td>
    <td>${fmt(r.roster_fit)}</td><td>${fmt(r.lineup_improvement)}</td>
    <td>${money(r.legal_max)}</td><td><b>${money(r.provisional_cap)}</b></td>
    <td class="l ${basisClass(r.basis)}" style="font-size:11px">${r.basis}</td>
    <td class="l">${r.sold?('SOLD '+r.owner_team+' $'+r.sale_price):r.status}</td>
   </tr>`).join('');
  document.querySelectorAll('#board tbody tr').forEach(tr=>{
    tr.onclick=()=>{selectPlayer(tr.dataset.id);};
    tr.ondblclick=e=>{e.preventDefault();openOverride(tr.dataset.id);};});
}

function drawOwners(owners,cand){
  $('#owners tbody').innerHTML=owners.map(o=>`<tr>
   <td class="l">${o.is_us?'<b>'+o.team_name+'</b>':o.team_name}
     <button data-ren="${o.owner_id}" style="padding:0 4px;font-size:10px">edit</button></td>
   <td>$${o.budget_remaining}</td><td>$${o.spent}</td><td>${o.roster_size}</td>
   <td>${o.open_slots}</td><td>$${o.general_max}</td>
   <td>${cand===null?'-':'$'+o.candidate_max}</td>
   <td>${o.QB}</td><td>${o.RB}</td><td>${o.WR}</td><td>${o.TE}</td>
   <td class="l ${o.can_bid_candidate?'':'muted'}" style="font-size:11px">${
     cand===null?'-':(o.can_bid_candidate?'yes':(o.blocked_reason||'no'))}</td></tr>`).join('');
  document.querySelectorAll('[data-ren]').forEach(b=>b.onclick=async e=>{
    e.stopPropagation();
    const cur=S.owners.find(o=>o.owner_id===b.dataset.ren);
    const name=prompt('Team name for '+b.dataset.ren, cur?cur.team_name:'');
    if(name){await post('/api/rename',{owner_id:b.dataset.ren,name});refresh();}});
}

function drawLog(){
  api('/api/log').then(j=>{$('#log tbody').innerHTML=(j.log||[]).map(e=>{
    const t=new Date(e.at*1000).toLocaleTimeString();
    let d=e.kind.toUpperCase()+' ';
    if(e.kind==='sale') d+=`${e.player} -> ${e.team} $${e.price}`+(e.market_updated?'':' (market NOT moved: unanchored)');
    else if(e.kind==='undo') d+=`${e.player} $${e.price} reversed`;
    else if(e.kind==='rename') d+=`${e.owner_id} = ${e.name}`;
    else if(e.kind==='override') d+=`${e.player_key} adj ${e.dollar_adjustment}`;
    else d+=JSON.stringify(e);
    return `<tr><td class="l muted" style="width:90px">${t}</td><td class="l">${d}</td></tr>`;}).join('');});
}

async function selectPlayer(id){
  SEL=id;
  const bid=$('#nomBid').value, leader=$('#nomLeader').value, by=$('#nomBy').value;
  const u=`/api/nomination?player_id=${id}`+(bid!==''?`&bid=${bid}`:'')
    +(leader?`&leader=${leader}`:'')+(by?`&by=${by}`:'');
  const n=await api(u); if(n.__err){$('#nomOut').innerHTML=`<div class="err">${n.__err}</div>`;return;}
  NOM=n; drawNom(n); drawOwners(n.owners,id); drawQB(n.qb);
  const opt=[...$('#nomPick').options].find(o=>o.value===id);
  if(!opt){const o=document.createElement('option');o.value=id;o.textContent=n.name;
    $('#nomPick').prepend(o);} $('#nomPick').value=id;
  if(!$('#salePrice').value) $('#salePrice').value=n.next_legal_bid;
}

function drawNom(n){
  const c=n.cap, r=c.rails, m=n.market;
  const rails=[
   ['exact legal maximum', money(r.legal_max), 'EXACT FINANCIAL/ROSTER ARITHMETIC'],
   ['market low / base / high', m.anchored?`${money(m.low)} / ${money(m.base)} / ${money(m.high)}`:'UNANCHORED', m.basis],
   ['proxy permissive ceiling', r.proxy_status==='cached'?money(r.proxy_ceiling):r.proxy_status.toUpperCase(),'PROXY/HEURISTIC'],
   ['cached CE bracket', r.ce_bracket?r.ce_bracket.join('-'):'none', r.ce_status==='none'?'no CE result for this state':r.ce_status],
   ['manual adjustment', r.manual_adjustment?(r.manual_adjustment>0?'+':'')+r.manual_adjustment:'none','MANUAL OVERRIDE'],
  ];
  $('#nomOut').innerHTML=`
   ${n.disagreement?`<div class="banner">${n.disagreement_label} -- market base ${money(m.base)} vs proxy ceiling ${money(r.proxy_ceiling)}</div>`:''}
   <div class="row" style="gap:18px;align-items:flex-end">
    <div><div class="muted">${n.name} &middot; ${n.position} ${n.nfl_team} &middot; bye ${n.bye_week||'-'} &middot; ${n.ppg} ppg</div>
     <div class="big ${n.verdict}">${n.verdict}</div></div>
    <div class="card"><div class="k">next legal bid</div><div class="v">${money(n.next_legal_bid)}</div></div>
    <div class="card"><div class="k">our exact legal max</div><div class="v">${money(n.our_legal_max)}</div></div>
    <div class="card"><div class="k">provisional cap</div><div class="v">${money(n.provisional_cap)}</div>
      <div class="k">${c.label}</div></div>
    <div class="card"><div class="k">opening cap</div><div class="v">${money(n.opening_cap)}</div>
      <div class="k">frozen at open</div></div>
    <div class="card"><div class="k">basis</div><div class="v" style="font-size:12px">${c.basis}</div>
      <div class="k">bound by ${c.bound_by}</div></div>
    <div class="card"><div class="k">roster fit / +lineup</div>
      <div class="v">${fmt(n.roster_fit)} / ${fmt(n.lineup_improvement)}</div>
      <div class="k">PROXY/HEURISTIC</div></div>
   </div>
   ${c.notes.length?`<div class="banner" style="margin-top:8px">${c.notes.join(' &middot; ')}</div>`:''}
   <div class="grid4" style="margin-top:9px">
    <div class="card"><div class="k">rails behind the cap</div>
     <table style="margin-top:4px">${rails.map(x=>`<tr><td class="l muted">${x[0]}</td>
      <td><b>${x[1]}</b></td><td class="l muted" style="font-size:10px">${x[2]}</td></tr>`).join('')}</table></div>
    <div class="card"><div class="k">opponents legally able to bid ${money(n.next_legal_bid)}</div>
     <div class="v">${n.n_capable_at_next_bid}</div>
     <div class="k">at the cap ${money(n.provisional_cap)}: ${n.n_capable_at_cap}</div>
     <div class="muted" style="font-size:11px;margin-top:4px">${
       n.capable_now.map(o=>o.team).join(', ')||'nobody'}</div></div>
    <div class="card"><div class="k">most plausible recipients</div>
     <table style="margin-top:4px">${(n.recipients||[]).map(b=>`<tr>
       <td class="l">${b.team}</td><td>${money(b.price)}</td>
       <td class="muted">w ${b.weight}</td></tr>`).join('')||'<tr><td class="muted">none</td></tr>'}</table>
     <div class="warn" style="font-size:10px;margin-top:5px">${S.recipient_warning}</div></div>
   </div>`;
}

function drawQB(q){
  if(!q){return;}
  const f=q.superflex_fallback;
  $('#qbOut').innerHTML=`
   ${q.replacement_warning?`<div class="banner">${q.replacement_warning}</div>`:''}
   <div class="grid4">
    <div class="card"><div class="k">startable QBs remaining</div>
      <div class="v">${q.startable_remaining}</div>
      <div class="k">of ${q.total_qbs_remaining} QBs left</div></div>
    <div class="card"><div class="k">teams by QB count</div>
      <div class="v" style="font-size:14px">0:${q.counts_by_bucket['0']} &nbsp;1:${q.counts_by_bucket['1']}
        &nbsp;2:${q.counts_by_bucket['2']} &nbsp;3+:${q.counts_by_bucket['3+']}</div></div>
    <div class="card"><div class="k">highest opposing candidate max</div>
      <div class="v">${money(q.highest_opposing_candidate_max)}</div></div>
    <div class="card"><div class="k">opponents able at next bid / at cap</div>
      <div class="v">${q.n_capable_at_next_bid} / ${q.n_capable_at_provisional_cap}</div></div>
    <div class="card"><div class="k">superflex skill fallback</div>
      ${f?`<div class="v" style="font-size:13px">QB +${f.best_available_qb.improvement}
        vs skill +${f.best_available_skill.improvement}</div>
        <div class="k">premium ${f.qb_premium} &middot; PROXY/HEURISTIC</div>`:'<div class="v">-</div>'}</div>
    <div class="card"><div class="k">QB3 insurance</div>
      <div class="muted" style="font-size:11px">${q.qb3_insurance}</div></div>
   </div>
   <div class="muted" style="font-size:11px;margin-top:6px">${q.startable_definition}</div>
   <div class="muted" style="font-size:11px">teams with money and room for this QB:
     ${(q.capable_opponents||[]).map(o=>o.team+' ($'+o.candidate_max+')').join(', ')||'none'}</div>`;
}

function openOverride(id){
  const r=ROWS.find(x=>x.player_id===id); if(!r) return;
  $('#ovName').textContent=r.name; $('#ovMsg').textContent='';
  const ov=(S.overrides||{})[r.notes&&''] || null;
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

$('#saleGo').onclick=async()=>{
  if(!SEL){$('#saleMsg').innerHTML='<span class="bad">pick a player first</span>';return;}
  const t0=performance.now();
  const j=await post('/api/sale',{player_id:SEL,owner_id:$('#saleOwner').value,
    price:parseInt($('#salePrice').value,10)});
  if(j.__err){$('#saleMsg').innerHTML=`<span class="bad">REJECTED: ${j.__err}</span>`;return;}
  $('#saleMsg').innerHTML=`<span class="ok">recorded in ${Math.round(performance.now()-t0)}ms (server ${j.total_ms}ms)</span>`;
  $('#nomBid').value=''; $('#salePrice').value=''; SEL=null; NOM=null;
  $('#nomOut').textContent='Pick a player.'; await refresh();};

$('#undo').onclick=async()=>{const j=await post('/api/undo');
  $('#saleMsg').innerHTML=j.__err?`<span class="bad">${j.__err}</span>`:'<span class="ok">undone</span>';
  await refresh();};
$('#rst').onclick=async()=>{
  if(!confirm('Reset the room to the OPENING state? Every recorded sale is cleared. '
    +'A backup of the current file is kept. Manual overrides are preserved.')) return;
  await post('/api/reset',{confirm:true}); SEL=null; $('#nomOut').textContent='Pick a player.';
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

$('#nomGo').onclick=()=>{if($('#nomPick').value) selectPlayer($('#nomPick').value);};
$('#nomPick').onchange=()=>selectPlayer($('#nomPick').value);
$('#nomBid').onchange=()=>{if(SEL)selectPlayer(SEL);};
$('#nomLeader').onchange=()=>{if(SEL)selectPlayer(SEL);};
$('#nomSearch').oninput=()=>{
  const q=$('#nomSearch').value.toLowerCase();
  const hits=q.length<2?[]:ROWS.filter(r=>!r.sold&&r.name.toLowerCase().indexOf(q)>=0).slice(0,40);
  $('#nomPick').innerHTML=hits.map(r=>`<option value="${r.player_id}">${r.name} (${r.position} ${r.nfl_team}) $${fmt(r.live_base)}</option>`).join('');
  if(hits.length===1) selectPlayer(hits[0].player_id);};
['#q','#fpos','#fstat','#fanch'].forEach(s=>{$(s).oninput=drawBoard;$(s).onchange=drawBoard;});
document.querySelectorAll('#board th').forEach(th=>th.onclick=()=>{
  const k=th.dataset.s; if(!k)return;
  if(sortKey===k) sortDir=-sortDir; else {sortKey=k;sortDir=-1;} drawBoard();});
$('#mLive').onclick=()=>{MODE='live';$('#mLive').classList.add('on');
  $('#mOpen').classList.remove('on');refresh();};
$('#mOpen').onclick=()=>{MODE='opening';$('#mOpen').classList.add('on');
  $('#mLive').classList.remove('on');refresh();};
refresh();
</script></body></html>
"""
