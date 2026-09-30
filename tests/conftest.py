import pytest
import torch

from fedroad.client import Client


class FakeSet(torch.utils.data.Dataset):
    """Tiny CIFAR-like dataset, no download needed."""

    def __init__(self, n=200):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.targets = [i % 10 for i in range(n)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


@pytest.fixture
def fake():
    return FakeSet()


@pytest.fixture
def client():
    return Client(
        cid=0,
        t_arrive=0,
        kph=50,
        t_leave=100,
        n_data=500,
        local_data=None,
        cpu_hz=1e9,
        batch=16,
        epochs=5,
        cpu_cycles=1e7,
        eff_capa=1e-27,
        snr_db=10,
        bw_hz=1e6,
        ptx=10,
        lr=1e-3,
        momentum=0.9,
        decay=5e-4,
    )
