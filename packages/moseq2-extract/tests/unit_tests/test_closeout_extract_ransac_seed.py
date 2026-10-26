"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def _floor():
    y, x = np.mgrid[:30, :40]
    return 700 + .3 * x - .2 * y + np.random.RandomState(3).normal(0, 2, (30, 40))


def test_ransac_repeated_calls_are_exact():
    from moseq2_extract.extract.roi import plane_ransac
    first = plane_ransac(_floor(), iters=30)
    second = plane_ransac(_floor(), iters=30)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_ransac_does_not_consume_global_rng():
    from moseq2_extract.extract.roi import plane_ransac
    np.random.seed(84)
    before = np.random.get_state()
    plane_ransac(_floor(), iters=30)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
