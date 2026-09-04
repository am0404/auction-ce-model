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

__all__ = ["normalize_name", "IdentityIndex", "MatchReport", "SUFFIXES"]

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
