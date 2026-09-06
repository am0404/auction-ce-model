"""Sleeper's projected prices, loaded as an anchor and nothing more.

What this file reads is the response of

    https://api.sleeper.com/players/nfl/values/regular/2026/2qb

which is Sleeper's **generic 2026 superflex ("2qb") projected value list**. It is
not this league's market, and four different quantities have to stay apart:

``sleeper_display_anchor``
    what a manager sees on Sleeper. A number on a screen that some of them will
    treat as a reference point.
``expected_clearing_price``
    what *this* room may actually pay. Built in
    :mod:`ceauction.market.prior`, from the anchor plus a format adjustment, a
    budget reconciliation and whatever the room has already done.
``ce_reservation_range``
    what our own championship-equity engine says we can afford. A different
    question with a different answer, computed by
    :mod:`ceauction.auction.reservation`.
``tactical_max_bid``
    what to actually bid, given who else can bid and how their money is
    committed. Not built anywhere yet.

Collapsing any two of them is the failure this module is arranged to prevent, so
the raw value, the displayed anchor and every transformation stay separately
inspectable all the way through.

**Two label mismatches travel with the data and are recorded, not corrected
away.** The endpoint is `2qb`, and this league is *superflex*: a team may start
a running back in the flexible slot and never draft a second quarterback, so
generic two-quarterback demand overstates the floor under quarterbacks here. And
the list prices tight ends for a format with a required TE slot, which this
league does not have. Both are represented as *credibility*, not as a
hand-coded discount -- see :mod:`ceauction.market.prior`.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..league import Position
from ..realdata.identity import canonical_player_key, normalize_name

__all__ = [
    "SLEEPER_ENDPOINT",
    "SLEEPER_FORMAT",
    "SLEEPER_SEASON",
    "ROSTERABLE_POSITIONS",
    "AnchorError",
    "SleeperAnchor",
    "AnchorBook",
    "display_price",
    "load_sleeper_csv",
    "AnchorMatchReport",
    "match_anchors_to_contract",
]

#: Recorded verbatim so no downstream report can imply a custom-league source.
SLEEPER_ENDPOINT = "https://api.sleeper.com/players/nfl/values/regular/2026/2qb"
SLEEPER_FORMAT = "2qb"
SLEEPER_SEASON = 2026

#: Positions this league can actually roster. Everything else -- kickers,
#: defenses, the odd linebacker in the source -- is excluded from budget
#: reconciliation, because dollars that cannot be spent here are not part of
#: this room's money.
ROSTERABLE_POSITIONS: Tuple[str, ...] = ("QB", "RB", "WR", "TE")

_REQUIRED_COLUMNS = ("sleeper_player_id", "player_name", "position", "nfl_team",
                     "active", "sleeper_2qb_raw_value")


class AnchorError(Exception):
    """A malformed or self-contradictory anchor file."""


def display_price(raw: Decimal) -> Optional[int]:
    """Sleeper's displayed dollar figure for a **positive** raw value.

    Verified against four observed pairs -- 58.19 to 58, 33.69 to 34, 12.78 to
    13, and 16.50 to 17 -- which fixes the rule as decimal ROUND_HALF_UP.
    Python's built-in :func:`round` is banker's rounding and turns 16.50 into
    16, so it is not used here.

    Returns ``None`` for a nonpositive raw value. The UI's treatment of those
    was never observed, and inventing one -- clamping to zero, to one, or to
    "undrafted" -- would put a fabricated number in a field labelled as
    Sleeper's. A nonpositive anchor is carried as *unpriced* instead.
    """
    if raw <= 0:
        return None
    return int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class SleeperAnchor:
    """One player's Sleeper value, exactly as published."""

    sleeper_player_id: str
    player_name: str
    position: str
    nfl_team: str
    active: bool
    raw_value: Decimal
    """The published number, untouched. Never overwritten by a transformation."""

    @property
    def canonical_key(self) -> str:
        return canonical_player_key(name=self.player_name)

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.player_name)

    @property
    def display_anchor(self) -> Optional[int]:
        """Verified displayed price, or ``None`` when the rule is unverified."""
        return display_price(self.raw_value)

    @property
    def is_positive(self) -> bool:
        return self.raw_value > 0

    @property
    def is_rosterable(self) -> bool:
        """Can this league hold this player at all?

        Position must be one the lineup graph knows, and the player must be
        active. Both exclusions matter for budget reconciliation: money the
        room cannot spend is not part of the room's money.
        """
        return self.active and self.position in ROSTERABLE_POSITIONS

    @property
    def counts_toward_room_budget(self) -> bool:
        return self.is_rosterable and self.is_positive

    @property
    def status(self) -> str:
        """``priced``, ``nonpositive``, ``inactive`` or ``ineligible_position``."""
        if self.position not in ROSTERABLE_POSITIONS:
            return "ineligible_position"
        if not self.active:
            return "inactive"
        if not self.is_positive:
            return "nonpositive"
        return "priced"

    def to_dict(self) -> Dict[str, object]:
        return {
            "sleeper_player_id": self.sleeper_player_id,
            "position": self.position,
            "nfl_team": self.nfl_team,
            "active": self.active,
            "raw_value": str(self.raw_value),
            "display_anchor": self.display_anchor,
            "status": self.status,
            "counts_toward_room_budget": self.counts_toward_room_budget,
        }


@dataclass(frozen=True)
class AnchorBook:
    """Every anchor from one file, with its provenance and its own arithmetic."""

    anchors: Tuple[SleeperAnchor, ...]
    source_sha256: str
    endpoint: str = SLEEPER_ENDPOINT
    source_format: str = SLEEPER_FORMAT
    season: int = SLEEPER_SEASON
    retrieved_at: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        seen: Dict[str, SleeperAnchor] = {}
        for a in self.anchors:
            prior = seen.get(a.sleeper_player_id)
            if prior is not None and prior != a:
                raise AnchorError(
                    f"sleeper_player_id {a.sleeper_player_id} appears twice with "
                    f"different values ({prior.raw_value} vs {a.raw_value}); a "
                    f"conflicting duplicate is refused rather than resolved by "
                    f"whichever row came last")
            seen[a.sleeper_player_id] = a

    # --- selections --------------------------------------------------------

    def __len__(self) -> int:
        return len(self.anchors)

    @property
    def by_sleeper_id(self) -> Dict[str, SleeperAnchor]:
        return {a.sleeper_player_id: a for a in self.anchors}

    @property
    def priced(self) -> Tuple[SleeperAnchor, ...]:
        """Active, rosterable, positive -- the ones a budget must reconcile."""
        return tuple(a for a in self.anchors if a.counts_toward_room_budget)

    @property
    def excluded_from_budget(self) -> Tuple[SleeperAnchor, ...]:
        """Positive but not rosterable here: wrong position, or inactive."""
        return tuple(a for a in self.anchors
                     if a.is_positive and not a.counts_toward_room_budget)

    def by_position(self) -> Dict[str, Tuple[SleeperAnchor, ...]]:
        out: Dict[str, List[SleeperAnchor]] = {p: [] for p in ROSTERABLE_POSITIONS}
        for a in self.priced:
            out[a.position].append(a)
        return {k: tuple(sorted(v, key=lambda x: (-x.raw_value, x.sleeper_player_id)))
                for k, v in out.items()}

    # --- arithmetic --------------------------------------------------------

    @property
    def priced_raw_total(self) -> Decimal:
        return sum((a.raw_value for a in self.priced), Decimal("0"))

    @property
    def priced_display_total(self) -> int:
        return sum(a.display_anchor or 0 for a in self.priced)

    def positional_totals(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for pos, group in self.by_position().items():
            out[pos] = {
                "players": len(group),
                "raw_total": str(sum((a.raw_value for a in group), Decimal("0"))),
                "display_total": sum(a.display_anchor or 0 for a in group),
            }
        return out

    # --- reporting ---------------------------------------------------------

    def fingerprint(self) -> str:
        """Content digest. Any anchor change must change every cost book downstream."""
        h = hashlib.sha256()
        h.update(f"{self.endpoint}|{self.source_format}|{self.season}\n".encode())
        for a in sorted(self.anchors, key=lambda x: x.sleeper_player_id):
            h.update(f"{a.sleeper_player_id}|{a.position}|{a.active}|"
                     f"{a.raw_value}\n".encode())
        return h.hexdigest()[:16]

    def provenance(self) -> Dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "format": self.source_format,
            "season": self.season,
            "source_sha256": self.source_sha256,
            "fingerprint": self.fingerprint(),
            "retrieved_at": self.retrieved_at,
            "label": ("GENERIC Sleeper projected value for the 2026 '2qb' "
                      "format. NOT this league's market, NOT a clearing-price "
                      "prediction, NOT championship-equity value."),
            "notes": self.notes,
        }

    def summary(self) -> Dict[str, object]:
        """Aggregates only -- safe to print and to commit."""
        status: Dict[str, int] = {}
        for a in self.anchors:
            status[a.status] = status.get(a.status, 0) + 1
        return {
            "provenance": self.provenance(),
            "rows": len(self.anchors),
            "status_counts": status,
            "priced_players": len(self.priced),
            "priced_raw_total": str(self.priced_raw_total),
            "priced_display_total": self.priced_display_total,
            "excluded_from_budget": len(self.excluded_from_budget),
            "positional": self.positional_totals(),
        }


def _parse_bool(text: str, row: int) -> bool:
    t = (text or "").strip().lower()
    if t in ("true", "1", "yes", "t"):
        return True
    if t in ("false", "0", "no", "f", ""):
        return False
    raise AnchorError(f"row {row}: {text!r} is not a boolean 'active' value")


def load_sleeper_csv(path, *, retrieved_at: Optional[str] = None,
                     notes: str = "") -> AnchorBook:
    """Read and validate the cleaned Sleeper export.

    Refuses a malformed number rather than coercing it: a price that silently
    became zero would be indistinguishable from a player Sleeper genuinely
    values at nothing, and the two mean very different things here.
    """
    p = Path(path)
    if not p.exists():
        raise AnchorError(f"anchor file not found: {p}")
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    text = data.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise AnchorError(
            f"anchor file is missing required column(s) {missing}; got "
            f"{reader.fieldnames}")

    anchors: List[SleeperAnchor] = []
    for i, row in enumerate(reader, start=2):
        pid = (row["sleeper_player_id"] or "").strip()
        if not pid:
            raise AnchorError(f"row {i}: empty sleeper_player_id")
        name = (row["player_name"] or "").strip()
        if not name:
            raise AnchorError(f"row {i}: empty player_name for id {pid}")
        try:
            raw = Decimal((row["sleeper_2qb_raw_value"] or "").strip())
        except (InvalidOperation, ValueError):
            raise AnchorError(
                f"row {i}: {row['sleeper_2qb_raw_value']!r} is not a number; a "
                f"malformed value is refused rather than coerced to zero, which "
                f"would be indistinguishable from a genuine zero") from None
        anchors.append(SleeperAnchor(
            sleeper_player_id=pid, player_name=name,
            position=(row["position"] or "").strip().upper(),
            nfl_team=(row["nfl_team"] or "").strip().upper(),
            active=_parse_bool(row["active"], i), raw_value=raw))
    if not anchors:
        raise AnchorError("anchor file contains no rows")
    return AnchorBook(anchors=tuple(anchors), source_sha256=sha,
                      retrieved_at=retrieved_at, notes=notes)


# ---------------------------------------------------------------------------
# Joining anchors to the real contract
# ---------------------------------------------------------------------------


@dataclass
class AnchorMatchReport:
    """What the join achieved, and every way it fell short.

    Counts are safe to commit. The name lists are for a local audit only.
    """

    contract_players: int = 0
    anchor_rows: int = 0
    matched: int = 0
    matched_priced: int = 0
    unmatched_contract: List[str] = field(default_factory=list)
    unmatched_anchor: List[str] = field(default_factory=list)
    ambiguous: List[Tuple[str, List[str]]] = field(default_factory=list)
    duplicate_anchor_names: List[Tuple[str, int]] = field(default_factory=list)
    position_conflicts: List[Tuple[str, str, str]] = field(default_factory=list)
    inactive: int = 0
    nonpositive: int = 0
    ineligible_position: int = 0

    @property
    def match_rate(self) -> float:
        return self.matched / self.contract_players if self.contract_players else 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "contract_players": self.contract_players,
            "anchor_rows": self.anchor_rows,
            "matched": self.matched,
            "matched_priced": self.matched_priced,
            "match_rate": round(self.match_rate, 4),
            "unmatched_contract": len(self.unmatched_contract),
            "unmatched_anchor": len(self.unmatched_anchor),
            "ambiguous": len(self.ambiguous),
            "duplicate_anchor_names": len(self.duplicate_anchor_names),
            "position_conflicts": len(self.position_conflicts),
            "anchor_inactive": self.inactive,
            "anchor_nonpositive": self.nonpositive,
            "anchor_ineligible_position": self.ineligible_position,
        }


def match_anchors_to_contract(
    book: AnchorBook,
    contract_players: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, SleeperAnchor], AnchorMatchReport]:
    """Join anchors onto contract rows by canonical player key.

    Returns ``{canonical_key: anchor}`` plus a report. Deliberately not fuzzy:
    an edit-distance match would silently join two different players and there
    would be no way to notice, so anything that does not agree on the key is
    reported as unmatched for a human to resolve.

    An ambiguous anchor name -- two Sleeper rows normalising to the same key --
    is left unmatched rather than resolved by arbitrary choice.
    """
    report = AnchorMatchReport(contract_players=len(contract_players),
                              anchor_rows=len(book))

    by_key: Dict[str, List[SleeperAnchor]] = {}
    for a in book.anchors:
        by_key.setdefault(a.canonical_key, []).append(a)
    for key, group in by_key.items():
        if len(group) > 1:
            report.duplicate_anchor_names.append((key, len(group)))

    out: Dict[str, SleeperAnchor] = {}
    matched_keys = set()
    for row in contract_players:
        try:
            key = canonical_player_key(row.get("player_key"), row.get("name"))
        except ValueError:
            continue
        group = by_key.get(key, [])
        if not group:
            report.unmatched_contract.append(key)
            continue
        if len(group) > 1:
            report.ambiguous.append((key, [a.sleeper_player_id for a in group]))
            continue
        anchor = group[0]
        contract_pos = str(row.get("position") or "").upper()
        if contract_pos and anchor.position and contract_pos != anchor.position:
            report.position_conflicts.append((key, contract_pos, anchor.position))
            continue
        out[key] = anchor
        matched_keys.add(key)
        report.matched += 1
        if anchor.counts_toward_room_budget:
            report.matched_priced += 1
        elif anchor.status == "inactive":
            report.inactive += 1
        elif anchor.status == "nonpositive":
            report.nonpositive += 1
        else:
            report.ineligible_position += 1

    report.unmatched_anchor = sorted(set(by_key) - matched_keys)
    return out, report
