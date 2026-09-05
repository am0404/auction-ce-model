"""Player identity: normalisation, and a join that reports what it could not do.

A silently half-matched join is worse than no join at all, because it looks like
a signal.  Everything here is built so the caller learns exactly which players
did not match, which matched more than one candidate, and which appeared twice
in a single source.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["normalize_name", "IdentityIndex", "MatchReport", "SUFFIXES",
           "canonical_player_key", "stable_player_id", "assign_stable_ids",
           "IdentityCollision"]

#: Generational and ordinal suffixes stripped before matching.  Kept explicit
#: rather than regex-guessed: "Vi" and "Li" are real name fragments, and a rule
#: that ate them would be silently wrong on a handful of players.
SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")

_PUNCT = re.compile(r"[.'’`,]")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """A canonical key for one player's name.

    Lowercase, accents folded, punctuation removed, generational suffixes
    dropped, whitespace collapsed.  ``"A.J. Brown Jr."`` and ``"AJ Brown"``
    both become ``"aj brown"``.

    Deliberately *not* fuzzy.  Edit-distance matching would join two different
    players with similar names and there would be no way to notice; this returns
    a key, and anything that does not agree on the key is reported as unmatched
    for a human to resolve.
    """
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = _PUNCT.sub("", text)
    text = _NONWORD.sub(" ", text)
    parts = [p for p in _SPACES.split(text) if p]
    while len(parts) > 1 and parts[-1] in SUFFIXES:
        parts.pop()
    return " ".join(parts)


@dataclass
class MatchReport:
    """What a join against one source achieved, and what it did not.

    Every field is a *list of names* rather than a count, because the failures
    are the part a human has to act on.
    """

    source: str
    left_rows: int = 0
    right_rows: int = 0
    matched: int = 0
    unmatched_left: List[str] = field(default_factory=list)
    unmatched_right: List[str] = field(default_factory=list)
    ambiguous: List[Tuple[str, List[str]]] = field(default_factory=list)
    duplicate_left: List[Tuple[str, int]] = field(default_factory=list)
    duplicate_right: List[Tuple[str, int]] = field(default_factory=list)
    conflicting: List[Tuple[str, str, List[str]]] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.matched / self.left_rows if self.left_rows else 0.0

    @property
    def clean(self) -> bool:
        return not (self.ambiguous or self.duplicate_left
                    or self.duplicate_right or self.conflicting)

    def summary(self) -> Dict[str, object]:
        """Counts only -- safe to print or commit."""
        return {
            "source": self.source,
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "matched": self.matched,
            "match_rate": round(self.match_rate, 4),
            "unmatched_left": len(self.unmatched_left),
            "unmatched_right": len(self.unmatched_right),
            "ambiguous": len(self.ambiguous),
            "duplicate_left": len(self.duplicate_left),
            "duplicate_right": len(self.duplicate_right),
            "conflicting": len(self.conflicting),
        }


class IdentityIndex:
    """A name -> row index over one source, with collisions surfaced.

    ``position_of`` is optional.  When supplied, two rows that share a
    normalised name but disagree on position are recorded as *conflicting*
    rather than silently letting the last one win -- that is the shape of a
    genuine two-players-one-name collision.
    """

    def __init__(self, rows: Sequence[object], name_of, position_of=None,
                 source: str = "source"):
        self.source = source
        self.rows = list(rows)
        self._by_key: Dict[str, List[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            key = normalize_name(name_of(row))
            if key:
                self._by_key[key].append(i)
        self._name_of = name_of
        self._position_of = position_of

    @property
    def keys(self):
        return set(self._by_key)

    def duplicates(self) -> List[Tuple[str, int]]:
        """Normalised names appearing on more than one row of this source."""
        return sorted(((k, len(v)) for k, v in self._by_key.items() if len(v) > 1),
                      key=lambda kv: (-kv[1], kv[0]))

    def conflicts(self) -> List[Tuple[str, List[str]]]:
        """Duplicate names whose rows disagree on position."""
        if self._position_of is None:
            return []
        out = []
        for key, idxs in self._by_key.items():
            if len(idxs) < 2:
                continue
            positions = sorted({str(self._position_of(self.rows[i])) for i in idxs})
            if len(positions) > 1:
                out.append((key, positions))
        return sorted(out)

    def get(self, key: str) -> Optional[object]:
        """The single row for ``key``, or ``None`` if absent or ambiguous.

        Ambiguity returns ``None`` on purpose: an arbitrary pick would be a
        guess, and the caller records it in ``MatchReport.ambiguous``.
        """
        idxs = self._by_key.get(key, [])
        return self.rows[idxs[0]] if len(idxs) == 1 else None

    def is_ambiguous(self, key: str) -> bool:
        return len(self._by_key.get(key, [])) > 1

    def candidates(self, key: str) -> List[str]:
        return [str(self._name_of(self.rows[i])) for i in self._by_key.get(key, [])]


def join_report(left: IdentityIndex, right: IdentityIndex,
                source: str) -> MatchReport:
    """Match ``left`` against ``right`` and describe every way it fell short."""
    rep = MatchReport(source=source,
                      left_rows=len(left.rows), right_rows=len(right.rows))
    rep.duplicate_left = left.duplicates()
    rep.duplicate_right = right.duplicates()
    for key, positions in right.conflicts():
        rep.conflicting.append((key, right.source, positions))

    matched_right_keys = set()
    for key in sorted(left.keys):
        if right.is_ambiguous(key):
            rep.ambiguous.append((key, right.candidates(key)))
            continue
        if right.get(key) is not None:
            rep.matched += 1
            matched_right_keys.add(key)
        else:
            rep.unmatched_left.append(key)
    rep.unmatched_right = sorted(right.keys - matched_right_keys
                                 - {k for k, _ in rep.ambiguous})
    return rep


# ---------------------------------------------------------------------------
# Stable player identity
# ---------------------------------------------------------------------------
#
# A ``PlayerSpec.player_id`` is not a label -- it is a coordinate into the
# engine's counter-based RNG.  Every random draw a player receives is keyed by
# it, so if the id moves, the player's entire simulated career moves with it.
#
# Deriving it from a sorted row index, as an earlier pass did, makes the id a
# function of *the pool* rather than of *the player*: change the fumble
# interpretation, the projection interpretation, the pool limit or anything
# else that reorders the rows, and a player silently inherits a different
# season, different injuries and a different set of common random numbers.
# Every paired comparison built on top of that is comparing two different
# people.
#
# So the id is a pure function of the canonical player key and nothing else.


class IdentityCollision(Exception):
    """Two distinct canonical keys hashed to the same stable id.

    Raised rather than worked around.  A silent collision would give two
    players one shared random stream, and the resulting correlation would look
    exactly like a modelling result.
    """


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x00000100000001B3
_U64 = (1 << 64) - 1

#: Ids are masked into 62 bits.  ``PoolArrays.stream_key`` is ``int64`` and the
#: RNG multiplies the coordinate by an odd constant, so staying clear of the
#: sign bit keeps the value representable without relying on wraparound.
_ID_BITS = 62
_ID_MASK = (1 << _ID_BITS) - 1


def canonical_player_key(player_key: Optional[str] = None,
                         name: Optional[str] = None) -> str:
    """The single string a player's identity is derived from.

    Prefers the contract's ``player_key`` (already a stable slug produced by
    ingestion) and falls back to the normalised name.  Both are run through the
    same lowercase/underscore canonicalisation so that ``"Josh Allen"`` and
    ``"josh_allen"`` cannot become two different people.
    """
    raw = player_key if player_key else name
    if not raw:
        raise ValueError("a player needs either a player_key or a name")
    text = normalize_name(str(raw).replace("_", " "))
    if not text:
        raise ValueError(f"player key {raw!r} normalises to nothing")
    return text.replace(" ", "_")


def stable_player_id(canonical_key: str) -> int:
    """A deterministic positive integer id for one canonical key.

    FNV-1a over the key's UTF-8 bytes, masked to 62 bits, with zero mapped
    away so that ``0`` never doubles as a sentinel.  Pure: no seed, no
    ordering, no process state.  The same key gives the same id in every
    scenario, every run and every process.
    """
    if not canonical_key:
        raise ValueError("canonical_key must be non-empty")
    h = _FNV_OFFSET
    for byte in canonical_key.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & _U64
    return (h & _ID_MASK) or 1


def assign_stable_ids(canonical_keys: Iterable[str]) -> Dict[str, int]:
    """``{canonical_key: stable id}``, failing loudly on any collision.

    62 bits over a few hundred players makes a collision astronomically
    unlikely, which is exactly why it must be checked rather than assumed: an
    unchecked assumption that never fires is indistinguishable from one that
    fires once.
    """
    out: Dict[str, int] = {}
    seen: Dict[int, str] = {}
    for key in canonical_keys:
        pid = stable_player_id(key)
        prior = seen.get(pid)
        if prior is not None and prior != key:
            raise IdentityCollision(
                f"stable id {pid} is claimed by both {prior!r} and {key!r}; "
                f"refusing to give two players one random stream")
        seen[pid] = key
        out[key] = pid
    return out
