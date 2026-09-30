import math

import numpy as np
import pytest

from fedroad.client import get_arrivals


def test_cp_time_and_energy(client):
    assert client.get_cp_time() == pytest.approx(25.0)
    assert client.get_cp_energy() == pytest.approx(25.0)


def test_throughput_is_shannon(client):
    assert client.get_throughput() == pytest.approx(1e6 * math.log2(11))


def test_co_time_and_energy(client):
    t = 160_000 / (1e6 * math.log2(11))
    assert client.get_co_time(10_000, 16) == pytest.approx(t)
    assert client.get_co_energy(10_000, 16) == pytest.approx(10 * t)


def test_cp_time_halves_when_cpu_doubles(client):
    t = client.get_cp_time()
    client.cpu_hz *= 2
    assert client.get_cp_time() == pytest.approx(t / 2)


def test_arrivals_sorted_and_in_range():
    np.random.seed(0)
    a = get_arrivals(1000, 0.1)
    assert a == sorted(a)
    assert all(0 < t < 1000 for t in a)


def test_arrivals_match_poisson_rate():
    np.random.seed(0)
    a = get_arrivals(100_000, 0.1)
    assert len(a) == pytest.approx(10_000, rel=0.05)
