"""Championship-equity model for a 12-team, $200, half-PPR Sleeper auction league.

The engine answers exactly one question: *given a drafted 15-man roster, what is
the probability this team wins the league?*  Everything else -- pricing, bidding,
roster completion -- is deliberately out of scope (see ``SPEC.md`` section 11).

Quick start::

    from ceauction.synthetic import make_synthetic_league
    from ceauction.simulate import simulate_seasons

    league = make_synthetic_league()
    out = simulate_seasons(league, n_sims=5000, seed=12345)
    print(out.championship_equity())
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
