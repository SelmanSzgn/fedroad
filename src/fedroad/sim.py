import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch

from .client import get_all_clients
from .data import create_class_indices, get_test_loader, get_trainset
from .eval import get_client_accuracy, get_test_accuracy, get_test_loss
from .model import Model
from .server import aggregate

log = logging.getLogger("fedroad")

FIELDS = [
    "round",
    "t",
    "active",
    "part",
    "drop",
    "acc",
    "loss",
    "cacc_mean",
    "cacc_min",
    "energy_j",
]


def set_seed(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(cfg, out):
    """Run the simulation and write metrics.csv in out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(cfg.model_dump_json(indent=2))

    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gmodel = Model().to(dev)
    size = sum(p.numel() for p in gmodel.parameters() if p.requires_grad)

    trainset = get_trainset(cfg.data_path)
    test_loader = get_test_loader(cfg.data_path)
    cidx = create_class_indices(trainset)
    clients = get_all_clients(cfg, trainset, cidx)

    dur, prec = cfg.round_duration_s, cfg.model_precision
    n_rounds = int(cfg.simulation_time_s / dur)

    # Untrained model: starting point of the curves
    acc = get_test_accuracy(test_loader, dev, gmodel)
    log.info(f"initial model | test acc: {acc:.2f} %")

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in range(n_rounds):
            t = r * dur
            act = [c for c in clients if c.t_arrive <= t < c.t_leave]
            log.info(
                f"round {r + 1}/{n_rounds} | t = {t} s | active: {len(act)}"
            )

            # A client joins only if it can train and upload before leaving
            ups, energy = [], 0.0
            for c in act:
                if c.can_finish(size, prec, c.t_leave - t):
                    ups.append((c.n_data, c.local_update(gmodel, dev)))
                    energy += c.get_cp_energy() + c.get_co_energy(size, prec)
            n_drop = len(act) - len(ups)

            # No participant: the global model is left unchanged
            gmodel = aggregate(gmodel, ups)

            acc = get_test_accuracy(test_loader, dev, gmodel)
            loss = get_test_loss(test_loader, dev, gmodel)
            log.info(f"  participants: {len(ups)} | dropped: {n_drop}")
            log.info(f"  test acc: {acc:.2f} % | test loss: {loss:.4f}")

            row = dict(
                round=r + 1,
                t=t,
                active=len(act),
                part=len(ups),
                drop=n_drop,
                acc=acc,
                loss=loss,
                cacc_mean=None,
                cacc_min=None,
                energy_j=energy,
            )
            if act:
                cacc = get_client_accuracy(act, gmodel, dev)
                row["cacc_mean"] = float(np.mean(cacc))
                row["cacc_min"] = float(np.min(cacc))
                log.info(
                    f"  client acc: mean {row['cacc_mean']:.2f} % | "
                    f"min {row['cacc_min']:.2f} %"
                )
            w.writerow(row)
    state = {k: v.cpu() for k, v in gmodel.state_dict().items()}
    torch.save(state, out / "model.pt")
    return out / "metrics.csv"
