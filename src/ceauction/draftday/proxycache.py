"""Background proxy ceilings, keyed by the whole auction state.

The immediate tactical proxy costs about seven seconds on the real 549-player
board. That is far too long for a ten-second bidding clock, so it never runs on
the request path: the dashboard asks for a ceiling, gets ``CALCULATING``
immediately, and a worker fills it in.

The worker is a **separate process**, not a thread, and that is not incidental.
The completion search is pure Python and holds the GIL, so running it in a
thread inside the server pushed sale entry from 127ms to 940ms and sometimes
past a second -- the tool's one hard latency rule, broken by the very
computation that was supposed to stay out of the way. In its own process it
cannot contend at all, and the measured cost of sale entry is unchanged whether
a proxy is running or not.

The worker loads the board once from the same content-addressed cache the
server used, then rebuilds each requested position by replaying the sales.
Only the sales, the player and the price cross the process boundary.

The cache key is the **complete** session fingerprint -- both engines plus the
manual overrides -- so a result computed under one room can never be served
under another. That is the rule this module exists to enforce: a stale ceiling
displayed beside a changed room is worse than no ceiling at all, because it
looks like an answer.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import threading
import time
from typing import Dict, List, Optional, Tuple

from pathlib import Path

from ..auction.completion import ComparisonCast, CompletionSettings
from ..market.costbook import cost_book_from_prior
from ..tactical.maxbid import TacticalSettings, immediate_max_bid
from ..tactical.realpilot import OWNER_IDS
from .board import DEFAULT_CONTRACT, DEFAULT_SLEEPER_CSV
from .session import DraftSession

__all__ = ["ProxyCeilingCache", "LEAN_SETTINGS"]

#: Deliberately smaller than the tactical default. The board is 549 players
#: deep and a live tool needs an answer this decade; the ladder is shorter and
#: the completion beam narrower, which is a real reduction in what was searched
#: and is why the result stays labelled PROXY/HEURISTIC.
LEAN_SETTINGS = TacticalSettings(
    completion=CompletionSettings(beam_width=16, candidate_pool=18,
                                  proxy_candidates=16, finalists=1,
                                  proxy_reps=16, max_candidates=64),
    max_prices=5, max_recipients=2, proxy_reps=16)


# --- the worker process ----------------------------------------------------
#
# Module-level so it can be pickled by name. The board is loaded once per
# worker and kept, because loading it is 0.1s and the search is 7s.

_WORKER: Dict[str, object] = {}


def _worker_init(contract: str, sleeper_csv: str) -> None:
    from .board import load_board
    _WORKER["board"] = load_board(contract=Path(contract),
                                  sleeper_csv=Path(sleeper_csv))


def _worker_run(sales: List[Tuple[int, str, int]], player_id: int,
                current_price: Optional[int]) -> Dict[str, object]:
    """Replay the room, then run one immediate tactical evaluation."""
    from .session import DraftSession
    board = _WORKER["board"]
    session = DraftSession(board, autosave=False)
    for pid_, owner, price in sales:
        session.record_sale_silent(int(pid_), str(owner), int(price))
    cb = cost_book_from_prior(board.prior, "base")
    gaps = {p: 1 for p in board.key_by_id if p not in cb}
    if gaps:
        cb = cb.with_costs(gaps)
    cast = ComparisonCast(0, tuple(() for _ in OWNER_IDS), OWNER_IDS)
    result = immediate_max_bid(
        session.state, cast, cb, int(player_id), market=session.market,
        key_by_id=board.key_by_id, current_price=current_price,
        settings=LEAN_SETTINGS)
    return {
        "permissive_ceiling": result.permissive_ceiling,
        "base_tactical_max": result.base_tactical_max,
        "robust_tactical_max": result.robust_tactical_max,
        "legal_max": result.legal_max,
        "result_kind": result.result_kind,
    }


class ProxyCeilingCache:
    """One worker process, one result per (state fingerprint, player).

    Results are never evicted by age -- only by the fingerprint changing --
    because a room that has not moved has not invalidated anything.
    """

    def __init__(self, session: DraftSession, *,
                 settings: TacticalSettings = LEAN_SETTINGS,
                 enabled: bool = True,
                 contract: Path = DEFAULT_CONTRACT,
                 sleeper_csv: Path = DEFAULT_SLEEPER_CSV):
        self.session = session
        self.settings = settings
        self.enabled = enabled
        self.contract = Path(contract)
        self.sleeper_csv = Path(sleeper_csv)
        self._lock = threading.Lock()
        self._done: Dict[Tuple[str, int], Dict[str, object]] = {}
        self._pending: set = set()
        self._pool: Optional[concurrent.futures.ProcessPoolExecutor] = None

    def _executor(self) -> concurrent.futures.ProcessPoolExecutor:
        """One worker process, started on first use and then kept.

        A single worker is deliberate: two concurrent seven-second searches
        would compete for cores with nothing else on the machine that matters,
        and the user only ever asks about the player on the block.
        """
        if self._pool is None:
            # "forkserver" where it exists, because the default on macOS is
            # "spawn", and spawn re-imports the parent's __main__ in the child.
            # For a tool a user may well start from their own small script,
            # that would re-run the script -- and the first thing it tries is
            # binding the dashboard's port. forkserver starts from a clean
            # process and imports only this module's dependencies.
            ctx = None
            for name in ("forkserver", "spawn"):
                try:
                    ctx = multiprocessing.get_context(name)
                    break
                except ValueError:
                    continue
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=ctx, initializer=_worker_init,
                initargs=(str(self.contract), str(self.sleeper_csv)))
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # --- reading -----------------------------------------------------------

    def get(self, player_id: int, *,
            current_price: Optional[int] = None) -> Tuple[Optional[int], str]:
        """``(ceiling, status)``. Never blocks, never crosses a fingerprint.

        ``status`` is one of ``absent`` / ``calculating`` / ``cached`` /
        ``failed`` / ``disabled``, and the dashboard prints it verbatim.
        """
        if not self.enabled:
            return None, "disabled"
        key = (self.session.fingerprint(), int(player_id))
        with self._lock:
            hit = self._done.get(key)
            if hit is not None:
                if hit.get("error"):
                    return None, "failed"
                return hit.get("permissive_ceiling"), "cached"
            if key in self._pending:
                return None, "calculating"
        return None, "absent"

    def request(self, player_id: int, *,
                current_price: Optional[int] = None) -> str:
        """Start a computation if this exact state has none. Returns the status."""
        if not self.enabled:
            return "disabled"
        value, status = self.get(player_id, current_price=current_price)
        if status in ("cached", "calculating", "failed"):
            return status
        key = (self.session.fingerprint(), int(player_id))
        with self._lock:
            if key in self._pending or key in self._done:
                return "calculating"
            self._pending.add(key)
        sales = [(s.player_id, s.owner_id, s.price)
                 for s in self.session.sales]
        watcher = threading.Thread(
            target=self._work, args=(key, sales, current_price),
            daemon=True, name=f"proxy-{player_id}")
        watcher.start()
        return "calculating"

    def detail(self, player_id: int) -> Optional[Dict[str, object]]:
        key = (self.session.fingerprint(), int(player_id))
        with self._lock:
            return dict(self._done[key]) if key in self._done else None

    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {"cached": len(self._done), "calculating": len(self._pending),
                    "enabled": self.enabled,
                    "current_fingerprint": self.session.fingerprint()}

    # --- the worker --------------------------------------------------------

    def _work(self, key: Tuple[str, int],
              sales: List[Tuple[int, str, int]],
              current_price: Optional[int]) -> None:
        """Wait on the worker process. This thread only blocks; it never computes."""
        fingerprint, player_id = key
        t0 = time.perf_counter()
        record: Dict[str, object]
        try:
            future = self._executor().submit(_worker_run, sales, player_id,
                                             current_price)
            record = dict(future.result())
            record["basis"] = "PROXY/HEURISTIC"
            record["label"] = (
                "Proxy ordering of branches under a REDUCED search. NOT "
                "championship equity, NOT a reservation price, and it carries "
                "no interval.")
            record["error"] = ""
        except Exception as exc:
            record = {"permissive_ceiling": None,
                      "error": f"{type(exc).__name__}: {exc}",
                      "basis": "PROXY/HEURISTIC"}
            # A dead pool must not poison every later request.
            self.shutdown()
        record["seconds"] = round(time.perf_counter() - t0, 2)
        record["fingerprint"] = fingerprint
        with self._lock:
            self._pending.discard(key)
            self._done[key] = record
