"""Draftable-pool coverage, and the alias mechanism that closes its gaps.

**The full-pool match rate is not the decision-relevant number.** A 549-player
pool includes a long tail nobody drafts, and an unmatched identity down there
costs nothing. What matters is coverage among the players a 12-team, 15-man
league will actually consume -- 180 roster slots, with realistic churn reaching
perhaps 240 or 300 deep. So coverage is reported by depth band, and the bands
are where the argument lives.

Aliases exist because an unresolved identity in the top 240 is a real defect,
not a statistic. They are explicit, reviewed and stored as data rather than
guessed at by an edit-distance rule that would silently join the wrong players.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .identity import normalize_name

__all__ = [
    "DEPTH_BANDS",
    "AliasBook",
    "BandCoverage",
    "coverage_by_band",
    "load_alias_book",
]

#: Depth bands, in players by recomputed season points.
#: 180 = 12 teams x 15 roster slots, the pool a full draft consumes.
#: 240 and 300 allow for in-season churn; ``None`` is the whole pool.
DEPTH_BANDS: Tuple[Optional[int], ...] = (180, 240, 300, None)


@dataclass
class AliasBook:
    """Reviewed name equivalences and per-player overrides.

    ``aliases`` maps a normalised source name to the normalised name it should
    join as. ``overrides`` supplies a field directly for a player the sources
    genuinely disagree about or do not cover.

    Both are **data, reviewed by a human**, not inference. An automatic fuzzy
    matcher would resolve these too, and would also silently join two different
    players with similar names -- with no way to notice afterwards.
    """

    aliases: Dict[str, str] = field(default_factory=dict)
    overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)
    notes: str = ""

    def resolve(self, key: str) -> str:
        return self.aliases.get(key, key)

    def override_for(self, key: str) -> Dict[str, object]:
        return self.overrides.get(self.resolve(key), {})

    def to_dict(self) -> Dict[str, object]:
        return {"notes": self.notes, "aliases": dict(self.aliases),
                "overrides": {k: dict(v) for k, v in self.overrides.items()}}


def load_alias_book(path: Optional[Path]) -> AliasBook:
    """Load an alias book, or an empty one when no path is supplied."""
    if path is None or not Path(path).exists():
        return AliasBook()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AliasBook(
        aliases={normalize_name(k): normalize_name(v)
                 for k, v in (payload.get("aliases") or {}).items()},
        overrides={normalize_name(k): dict(v)
                   for k, v in (payload.get("overrides") or {}).items()},
        notes=str(payload.get("notes", "")))


@dataclass(frozen=True)
class BandCoverage:
    """Coverage for one depth band. Counts and rates only -- safe to publish."""

    band: str
    players: int
    identity: int
    team: int
    bye: int
    injury: int
    dispersion: int
    unresolved_identities: int

    def _pct(self, n: int) -> float:
        return round(100.0 * n / self.players, 1) if self.players else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "band": self.band, "players": self.players,
            "identity_pct": self._pct(self.identity),
            "team_pct": self._pct(self.team),
            "bye_pct": self._pct(self.bye),
            "injury_pct": self._pct(self.injury),
            "dispersion_pct": self._pct(self.dispersion),
            "unresolved_identities": self.unresolved_identities,
        }


def _ranked(payload: Dict) -> List[Dict]:
    rows = list(payload.get("players", []))
    rows.sort(key=lambda p: -((p.get("season_points") or {}).get("points") or 0.0))
    return rows


def coverage_by_band(payload: Dict,
                     bands: Sequence[Optional[int]] = DEPTH_BANDS,
                     ) -> Tuple[List[BandCoverage], Dict[str, List[str]]]:
    """Coverage per depth band, plus the unresolved identities in each.

    The returned name lists are for **local** triage only. They are player
    names from a subscriber-gated source and must not be committed; the
    ``BandCoverage`` counts are what a published report carries.
    """
    rows = _ranked(payload)
    out: List[BandCoverage] = []
    unresolved: Dict[str, List[str]] = {}

    for band in bands:
        subset = rows if band is None else rows[:band]
        label = "full_pool" if band is None else f"top_{band}"
        missing: List[str] = []
        team = bye = injury = disp = 0
        for row in subset:
            has_team = bool(row.get("nfl_team"))
            has_bye = bool(row.get("bye_week"))
            team += has_team
            bye += has_bye
            injury += (row.get("availability") or {}).get("injury_prob") is not None
            disp += (row.get("cohort_dispersion") or {}).get("weekly_cv") is not None
            if not has_team or not has_bye:
                missing.append(str(row.get("name")))
        out.append(BandCoverage(
            band=label, players=len(subset),
            identity=team,           # a matched identity is what supplies team
            team=team, bye=bye, injury=injury, dispersion=disp,
            unresolved_identities=len(missing)))
        unresolved[label] = missing
    return out, unresolved
