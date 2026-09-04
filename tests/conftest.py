"""Shared fixtures.  Everything here is deterministic; no test may be flaky."""

from __future__ import annotations

import pytest

from ceauction.synthetic import make_identical_league, make_synthetic_league


@pytest.fixture(scope="session")
def league():
    return make_synthetic_league()


@pytest.fixture(scope="session")
def identical_league():
    return make_identical_league()
