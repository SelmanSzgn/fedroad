import torch

from fedroad.client import Client
from fedroad.model import Model
from fedroad.server import aggregate


def make_model(v):
    """Model with all parameters set to v."""
    model = Model()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(v)
    return model


def test_aggregate_weighted_average():
    gmodel = make_model(0.0)
    uploads = [(100, make_model(1.0)), (300, make_model(3.0))]
    aggregate(gmodel, uploads)
    # (100 * 1 + 300 * 3) / 400 = 2.5
    for p in gmodel.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.5))


def test_aggregate_empty_keeps_model():
    gmodel = make_model(7.0)
    aggregate(gmodel, [])
    for p in gmodel.parameters():
        assert torch.allclose(p, torch.full_like(p, 7.0))


def make_client():
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


def test_can_finish_uses_time_left():
    c = make_client()  # compute time: 1e7 * 500 * 5 / 1e9 = 25 s
    assert c.can_finish(10_000, 16, time_left=60)
    assert not c.can_finish(10_000, 16, time_left=10)
