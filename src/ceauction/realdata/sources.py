"""Loaders for the four canonical source files the inventory documented.

**Every path is an argument.** Nothing here knows where the data lives, and no
absolute path appears anywhere in this package. The sources are subscriber-gated
vendor exports that are not redistributable and must never enter this public
repository; the caller supplies locations at run time.

Each loader returns typed rows plus a :class:`SourceFile` record carrying the
file's SHA-256, so the output identifies its inputs by content rather than by
filename. The inventory found a case where the later-*sounding* filename held
the older data, which is exactly what a content hash prevents.

Deliberately **not** loaded: ``players_provisional.csv``. It is machine
generated from an uncalibrated placeholder table and must never be imported as
real data. :func:`refuse_synthetic_pool` exists to make that refusal loud.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SourceFile",
    "ProjectionRow",
    "FantasyProsRow",
    "InjuryProfileRow",
    "DispersionFits",
    "SyntheticSourceRefused",
    "load_projections",
    "load_fantasypros",
    "load_injury_profiles",
    "load_dispersion_fits",
    "refuse_synthetic_pool",
    "sha256_of",
]

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

#: Column headers that identify the synthetic provisional pool. Loading that
#: file as real data would populate the engine with placeholder depth-chart
#: tiers that look exactly like projections.
SYNTHETIC_POOL_HEADERS = frozenset(
    {"player_id", "name", "pos", "nfl_team", "depth_rank", "ppg_baseline", "adp"}
)


class SyntheticSourceRefused(RuntimeError):
    """Raised when a file is recognised as generated placeholder data."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceFile:
    """Provenance for one input, identified by content rather than name."""

    logical_name: str
    sha256: str
    role: str
    vendor: Optional[str] = None
    retrieved_at: Optional[str] = None
    notes: str = ""
    rows: int = 0

    def to_dict(self) -> Dict[str, object]:
        """Provenance only -- carries no player data and no local path."""
        return {
            "logical_name": self.logical_name,
            "sha256": self.sha256,
            "vendor": self.vendor,
            "retrieved_at": self.retrieved_at,
            "role": self.role,
            "notes": self.notes,
        }


def _clean(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip()
    return None if text in ("", "-", "nan", "None", "NA", "N/A") else text


def refuse_synthetic_pool(path: Path) -> None:
    """Raise if ``path`` looks like the generated provisional pool.

    Checked by header signature rather than filename, because the danger is
    precisely that the file can be renamed and still look like a real board.
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, StopIteration):
        return
    cols = {c.strip().strip('"').lower() for c in header}
    if SYNTHETIC_POOL_HEADERS <= cols:
        raise SyntheticSourceRefused(
            f"{path.name} has the column signature of the generated provisional "
            "player pool (player_id, name, pos, nfl_team, depth_rank, "
            "ppg_baseline, adp). That file is machine-generated from an "
            "uncalibrated placeholder table and is not real player data. "
            "Refusing to load it. See docs/PLAYER_DATA_INVENTORY.md section 3.6."
        )


# ---------------------------------------------------------------------------
# 1. Component stat projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionRow:
    """One player's season-long component stat projection.

    ``None`` throughout means *not projected*, never zero. The source leaves
    rushing-touchdown cells blank on more than three quarters of its rows.
    """

    name: str
    position: str
    pass_yards: Optional[float] = None
    pass_tds: Optional[float] = None
    interceptions: Optional[float] = None
    rush_yards: Optional[float] = None
    rush_tds: Optional[float] = None
    rec_yards: Optional[float] = None
    rec_tds: Optional[float] = None
    receptions: Optional[float] = None
    fumbles: Optional[float] = None
    raw: Dict[str, str] = field(default_factory=dict)

    def stat_dict(self) -> Dict[str, Optional[float]]:
        return {
            "pass_yards": self.pass_yards, "pass_tds": self.pass_tds,
            "interceptions": self.interceptions, "rush_yards": self.rush_yards,
            "rush_tds": self.rush_tds, "rec_yards": self.rec_yards,
            "rec_tds": self.rec_tds, "receptions": self.receptions,
            "fumbles": self.fumbles,
        }


#: Source header -> canonical stat name. The vendor's own ``Projections`` total
#: is deliberately absent: it is full PPR and must never be used.
PROJECTION_COLUMNS: Dict[str, str] = {
    "Pass Yards": "pass_yards",
    "Pass TDs": "pass_tds",
    "Ints": "interceptions",
    "Rush Yards": "rush_yards",
    "Rush TDs": "rush_tds",
    "Rec Yards": "rec_yards",
    "Rec TDs": "rec_tds",
    "Receptions": "receptions",
    "Fumbles": "fumbles",
}

#: Source columns that exist but must not become fantasy points.
FORBIDDEN_PROJECTION_COLUMNS = frozenset({"Projections", "7-Day Delta"})


def _to_float(value) -> Optional[float]:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def load_projections(path, positions: Sequence[str] = SKILL_POSITIONS
                     ) -> Tuple[List[ProjectionRow], SourceFile]:
    """Load season-long component stat projections."""
    path = Path(path)
    refuse_synthetic_pool(path)
    rows: List[ProjectionRow] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in PROJECTION_COLUMNS if c not in header]
        if "Name" not in header or "Pos" not in header:
            raise ValueError(
                f"{path.name} is missing Name/Pos; header was {header[:8]}")
        for raw in reader:
            clean = {(k or "").strip(): v for k, v in raw.items()}
            pos = _clean(clean.get("Pos"))
            if pos not in positions:
                continue
            name = _clean(clean.get("Name"))
            if not name:
                continue
            values = {canon: _to_float(clean.get(src))
                      for src, canon in PROJECTION_COLUMNS.items()}
            rows.append(ProjectionRow(
                name=name, position=pos,
                raw={k: ("" if v is None else str(v)) for k, v in clean.items()},
                **values))
    return rows, SourceFile(
        logical_name=path.name, sha256=sha256_of(path),
        role="performance_projection", vendor="component-stat projection source",
        rows=len(rows),
        notes=("Component stat lines. The file's own Projections column solves to "
               "full PPR and is never read; " +
               (f"columns absent from this export: {missing}. "
                if missing else "") +
               "blank cells mean not projected, not zero."))


# ---------------------------------------------------------------------------
# 2. FantasyPros: bye weeks, and optional expert metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FantasyProsRow:
    """Identity and bye week, plus expert labels kept as metadata only.

    ``upside`` and ``bust`` are ordinal 1-5 expert labels. They are carried for
    traceability and **must never** be mapped to variance, standard deviation,
    ceiling, floor, spike probability or any other distribution parameter.
    """

    name: str
    position: Optional[str] = None
    team: Optional[str] = None
    bye_week: Optional[int] = None
    upside: Optional[int] = None
    bust: Optional[int] = None
    raw: Dict[str, str] = field(default_factory=dict)


_STARS = re.compile(r"^\s*(\d)\s*out of\s*5")


def _stars(value) -> Optional[int]:
    """``'4 out of 5'`` -> 4; ``'-'``, blank or unparseable -> ``None``.

    ``None`` is the honest missing value. The previous model used ``-1`` as a
    sentinel and counting it as data inflated every correlation it appeared in.
    """
    text = _clean(value)
    if text is None:
        return None
    m = _STARS.match(text)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 5 else None


def load_fantasypros(path, positions: Sequence[str] = SKILL_POSITIONS
                     ) -> Tuple[List[FantasyProsRow], SourceFile]:
    """Load bye weeks and optional expert labels."""
    path = Path(path)
    refuse_synthetic_pool(path)
    rows: List[FantasyProsRow] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            clean = {(k or "").strip().strip('"').strip(): v for k, v in raw.items()}
            name = _clean(clean.get("PLAYER NAME"))
            if not name:
                continue
            pos_raw = _clean(clean.get("POS")) or ""
            m = re.match(r"^([A-Za-z]+)", pos_raw)
            pos = m.group(1).upper() if m else None
            if positions and pos not in positions:
                continue
            bye = _to_float(clean.get("BYE WEEK"))
            rows.append(FantasyProsRow(
                name=name, position=pos, team=_clean(clean.get("TEAM")),
                bye_week=int(bye) if bye else None,
                upside=_stars(clean.get("UPSIDE")),
                bust=_stars(clean.get("BUST")),
                raw={k: ("" if v is None else str(v)) for k, v in clean.items()}))
    return rows, SourceFile(
        logical_name=path.name, sha256=sha256_of(path),
        role="player_identity", vendor="expert consensus source", rows=len(rows),
        notes=("Bye week and team are used. RK and ECR VS. ADP are market signals "
               "and are not read. UPSIDE/BUST are ordinal expert labels carried as "
               "metadata only and never mapped to any distribution parameter."))


# ---------------------------------------------------------------------------
# 3. Injury profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjuryProfileRow:
    """One player's availability signals, kept separate by kind.

    ``injury_prob`` is **season-level injury risk** -- the probability of a
    significant injury at some point this season. It is *not* a weekly
    missed-game probability and must not be used as a weekly hazard.

    ``proj_games_missed`` is projected games missed across the coming season.

    They are different quantities and are never combined here. No weekly injury
    process is derived from either.
    """

    name: str
    position: Optional[str] = None
    injury_prob: Optional[float] = None
    proj_games_missed: Optional[float] = None
    durability: Optional[float] = None
    positional_risk_group: Optional[int] = None
    injury_count: Optional[int] = None
    raw: Dict[str, object] = field(default_factory=dict)


def load_injury_profiles(path, positions: Sequence[str] = SKILL_POSITIONS
                         ) -> Tuple[List[InjuryProfileRow], SourceFile]:
    """Load per-player injury profiles from the extracted JSON."""
    path = Path(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    players = payload.get("players", [])
    rows: List[InjuryProfileRow] = []
    for p in players:
        name = _clean(p.get("name"))
        if not name:
            continue
        pos = _clean(p.get("pos"))
        if positions and pos not in positions:
            continue
        rows.append(InjuryProfileRow(
            name=name, position=pos,
            injury_prob=_to_float(p.get("injury_prob")),
            proj_games_missed=_to_float(p.get("proj_games_missed")),
            durability=_to_float(p.get("durability")),
            positional_risk_group=(int(p["positional_risk_group"])
                                   if p.get("positional_risk_group") is not None
                                   else None),
            injury_count=(int(p["injury_count"])
                          if p.get("injury_count") is not None else None),
            raw=p))
    return rows, SourceFile(
        logical_name=path.name, sha256=sha256_of(path),
        role="injury_availability", vendor=_clean(payload.get("source")),
        retrieved_at=_clean(payload.get("last_update_time")),
        rows=len(rows),
        notes=("injury_prob is season-level injury risk, NOT a weekly hazard. "
               "proj_games_missed is projected games missed this season. Kept "
               "separate; no weekly injury process is derived. The vendor's own "
               "fantasy total in this file is never read. retrieved_at may be "
               "null: the capture carries no update timestamp."))


# ---------------------------------------------------------------------------
# 4. Fitted dispersion / availability cohort statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispersionFits:
    """Positional cohort statistics fitted from historical weekly results.

    ``weekly_cv`` is TOTAL weekly dispersion conditional on appearing -- a
    within-player coefficient of variation averaged across players. It does not
    carry this engine's split between forecastable (``weekly_state_sd``) and
    unforecastable (``week_sd``) variation, and applying all of it to one of
    them would assert a split the data does not support.

    ``weekly_miss`` counts ANY non-appearance -- benching, rest, trade -- so it
    is an availability hazard, not an injury hazard.
    """

    weekly_cv: Dict[str, float]
    weekly_miss: Dict[str, float]
    seasons: Tuple[int, ...] = ()
    cv_sample: Optional[int] = None
    miss_sample: Optional[int] = None
    cv_cohort: str = ""
    miss_cohort: str = ""
    fitted_at: Optional[str] = None


def load_dispersion_fits(path) -> Tuple[DispersionFits, SourceFile]:
    """Load fitted positional dispersion and availability rates."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    fits = payload.get("fits", {})
    cv = fits.get("player.weekly_cv", {})
    miss = fits.get("availability.weekly_miss", {})
    obj = DispersionFits(
        weekly_cv={k: float(v) for k, v in (cv.get("value") or {}).items()},
        weekly_miss={k: float(v) for k, v in (miss.get("value") or {}).items()},
        seasons=tuple(payload.get("seasons", [])),
        cv_sample=cv.get("sample"), miss_sample=miss.get("sample"),
        cv_cohort=str(cv.get("cohort", "")), miss_cohort=str(miss.get("cohort", "")),
        fitted_at=_clean(payload.get("fitted_at")))
    return obj, SourceFile(
        logical_name=path.name, sha256=sha256_of(path),
        role="outcome_distribution", vendor="fitted from historical weekly results",
        retrieved_at=obj.fitted_at, rows=len(obj.weekly_cv),
        notes=("weekly_cv is TOTAL weekly dispersion conditional on appearing, a "
               "cohort average applied to individuals. weekly_miss counts any "
               "non-appearance, so it is availability rather than injury."))
