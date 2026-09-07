"""Draft-day delivery layer: a real opening board and a live local dashboard.

This package ships **no new valuation model**. Every number it shows is either
exact arithmetic over the auction rules, a market estimate produced by the
existing :mod:`ceauction.market` layer, the existing proxy ordering from
:mod:`ceauction.auction.proxy`, or a manual entry the user typed. Where a
championship-equity result does not meet the audit gate in
:mod:`ceauction.draftday.caps`, this package refuses to print a CE max bid and
says which rail bound the number instead.

The governing rule, applied everywhere: **never disguise a market or proxy
estimate as championship-equity evidence.** Every displayed recommendation
carries one of the labels in :data:`ceauction.draftday.caps.BASES`.
"""

from __future__ import annotations

__all__ = ["DRAFTDAY_DIR"]

from pathlib import Path

#: Where every draft-day artifact is written. Gitignored: this tree holds real
#: player-level data and must never reach a commit.
DRAFTDAY_DIR = Path("local_data/draftday")
