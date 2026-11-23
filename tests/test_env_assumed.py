"""Tests for the assumed-response models (synthetic parameters only)."""
import numpy as np
import pytest

from ttrl.env import assumed as am
from ttrl.env import tariff as tf


def test_categories_by_load():
    slot = np.arange(48)
    prof = 1 + np.sin(2 * np.pi * slot / 48)
    cat = am.categories_by_load(np.tile(prof, (3, 1)))
    assert np.bincount(cat).tolist() == [9, 10, 5]
    assert np.bincount(am.categories_by_load(np.tile(prof, (3, 1)), (8, 8, 8))).tolist() == [8, 8, 8]
    hourly = prof.reshape(24, 2).mean(1)
    assert cat[np.argmax(hourly)] == 2 and cat[np.argmin(hourly)] == 0


def test_linear_own_price_matches_formula():
    m = am.LinearOwnPrice(eps=-0.1)
    day = np.array([[1, 0, -1, 0, 0, 0, 0, 0]])
    e = m.effect(day)[0]
    r_high = (tf.PRICE[tf.HIGH] - tf.PRICE_FLAT) / tf.PRICE_FLAT
    assert e[0] == pytest.approx(100 * -0.1 * r_high)
    assert e[6] == pytest.approx(100 * -0.1 * (tf.PRICE[tf.NORMAL] - tf.PRICE_FLAT) / tf.PRICE_FLAT)
    assert e[12] > 0                                       # Low price raises load


def test_elasticity_matrix_self_and_cross():
    cat = np.array([0] * 8 + [1] * 8 + [2] * 8)
    own = np.array([-0.1, -0.1, -0.1])
    cross = np.array([[0.005, 0.01, 0.02], [0.01, 0.005, 0.016], [0.02, 0.016, 0.005]])
    m = am.ElasticityMatrix(own, cross, cat)
    M = m.matrix()
    assert M[20, 20] == -0.1 and M[20, 21] == 0.005 and M[20, 3] == 0.02
    e0 = m.effect(np.zeros((1, 8), int))
    r_n = (tf.PRICE[tf.NORMAL] - tf.PRICE_FLAT) / tf.PRICE_FLAT
    i = 20                                                  # a peak hour, all-Normal day
    expected = r_n * (own[2] + sum(cross[2, cat[j]] for j in range(24) if j != i))
    assert e0[0, 2 * i] == pytest.approx(100 * expected)
    assert np.allclose(e0[0, 0::2], e0[0, 1::2])            # both half-hours of an hour equal
    # a High block lowers its own hours and raises every other hour (positive cross-elasticities)
    day = np.zeros((1, 8), int)
    day[0, 5] = tf.HIGH                                     # local 17-20 = tariff hours 15-17
    e1 = m.effect(day)
    assert (e1[0, 30:36] < e0[0, 30:36]).all() and (e1[0, :30] > e0[0, :30]).all()
    assert (am.clip_effect(np.array([-150.0, 5.0])) == [-100.0, 5.0]).all()


def test_aalami_2010_values():
    cat = np.array([0] * 9 + [1] * 10 + [2] * 5)
    m = am.aalami_2010(cat)
    M = m.matrix()
    assert np.allclose(np.diag(M), -0.10)
    assert M[23, 10] == 0.016 and M[23, 0] == 0.012 and M[10, 0] == 0.01 and M[23, 22] == 0.0
    assert am.aalami_2010(cat, same_period_cross=0.01).matrix()[23, 22] == 0.01
    assert "Table 1" in m.source
