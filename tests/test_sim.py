import csv

import torch
import yaml

from fedroad import sim
from fedroad.config import Cfg
from test_config import CFG


def small_cfg():
    with open(CFG) as f:
        d = yaml.safe_load(f)
    d.update(
        simulation_time_s=300,
        poisson_rate=0.1,
        min_n_data=10,
        max_n_data=30,
        n_local_epochs=1,
    )
    return Cfg(**d)


def patch(monkeypatch, fake):
    dl = torch.utils.data.DataLoader(fake, batch_size=64)
    monkeypatch.setattr(sim, "get_trainset", lambda root: fake)
    monkeypatch.setattr(sim, "get_test_loader", lambda root: dl)


def test_run_writes_metrics_and_is_reproducible(monkeypatch, tmp_path, fake):
    patch(monkeypatch, fake)
    cfg = small_cfg()
    p1 = sim.run(cfg, tmp_path / "a")
    p2 = sim.run(cfg, tmp_path / "b")

    with open(p1) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    for r in rows:
        assert int(r["part"]) + int(r["drop"]) == int(r["active"])
    assert (tmp_path / "a" / "cfg.json").exists()
    assert p1.read_text() == p2.read_text()
