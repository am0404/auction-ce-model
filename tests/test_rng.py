"""The counter-based RNG underpins reproducibility and common random numbers."""

from __future__ import annotations

import numpy as np
import pytest

from ceauction import rng


def test_uniform_is_in_unit_interval_and_reproducible():
    a = rng.uniform(7, rng.Kind.WEEK_NOISE, np.arange(1000).reshape(-1, 1), np.arange(3))
    b = rng.uniform(7, rng.Kind.WEEK_NOISE, np.arange(1000).reshape(-1, 1), np.arange(3))
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() < 1.0


def test_moments_are_correct():
    n = rng.normal(11, rng.Kind.WEEK_NOISE, np.arange(200_000).reshape(-1, 1), np.arange(4))
    assert abs(float(n.mean())) < 0.01
    assert abs(float(n.std()) - 1.0) < 0.01
    e = rng.exponential(11, rng.Kind.SPIKE_SIZE, np.arange(200_000))
    assert abs(float(e.mean()) - 1.0) < 0.02
    b = rng.bernoulli(0.25, 11, rng.Kind.SPIKE_HIT, np.arange(200_000))
    assert abs(float(b.mean()) - 0.25) < 0.01


def test_streams_are_independent_across_kinds_and_entities():
    coords = np.arange(50_000)
    a = rng.normal(3, rng.Kind.WEEK_NOISE, coords)
    b = rng.normal(3, rng.Kind.PROJ_NOISE, coords)
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.02
    c = rng.normal(3, rng.Kind.WEEK_NOISE, coords + 1)
    assert abs(float(np.corrcoef(a, c)[0, 1])) < 0.02


def test_seed_changes_everything():
    a = rng.uniform(1, rng.Kind.WEEK_NOISE, np.arange(1000))
    b = rng.uniform(2, rng.Kind.WEEK_NOISE, np.arange(1000))
    assert not np.allclose(a, b)


def test_value_depends_only_on_coordinates_not_call_order():
    """This is the property that makes chunking safe."""
    full = rng.uniform(9, rng.Kind.INJURY_ONSET, np.arange(100))
    piece = rng.uniform(9, rng.Kind.INJURY_ONSET, np.arange(40, 60))
    assert np.array_equal(full[40:60], piece)


def test_permutation_batch_is_valid_and_chunk_stable():
    p = rng.permutation_batch(12, 20, 5, rng.Kind.SCHEDULE_PERM)
    for row in p:
        assert sorted(row.tolist()) == list(range(12))
    later = rng.permutation_batch(12, 5, 5, rng.Kind.SCHEDULE_PERM, offset=15)
    assert np.array_equal(p[15:20], later)
