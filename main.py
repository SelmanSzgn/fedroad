import random
from datetime import datetime

import numpy as np
import torch
import yaml

from client import get_all_clients, get_arrivals
from data import get_trainset, get_test_loader, create_class_indices
from eval import get_test_accuracy, get_test_loss, get_client_accuracy
from model import Model
from server import aggregate


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


def set_seed(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":

    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmodel = Model().to(device)
    size = sum(p.numel() for p in gmodel.parameters() if p.requires_grad)

    trainset = get_trainset()
    test_loader = get_test_loader()
    class_indices = create_class_indices(trainset)

    arrivals = get_arrivals(cfg["simulation_time_s"], cfg["poisson_rate"])
    clients = get_all_clients(
        arrivals,
        trainset,
        class_indices,
        cfg["n_sub_classes"],
        cfg["min_speed_kph"],
        cfg["max_speed_kph"],
        cfg["road_length_m"],
        cfg["min_n_data"],
        cfg["max_n_data"],
        float(cfg["min_cpu_hertz"]),
        float(cfg["max_cpu_hertz"]),
        cfg["batch"],
        cfg["n_local_epochs"],
        float(cfg["n_cpu_cycles_per_data"]),
        float(cfg["effective_capacitance"]),
        float(cfg["snr_db_min"]),
        float(cfg["snr_db_max"]),
        float(cfg["bandwidth_hz"]),
        cfg["tx_power_w"],
        float(cfg["learning_rate"]),
        cfg["momentum"],
        float(cfg["weight_decay"]),
    )

    dur = cfg["round_duration_s"]
    prec = cfg["model_precision"]
    n_rounds = int(cfg["simulation_time_s"] / dur)

    # Untrained model: starting point of the curves
    acc = get_test_accuracy(test_loader, device, gmodel)
    log(f"initial model | test acc: {acc:.2f} %")

    for r in range(n_rounds):
        t = r * dur
        active = [c for c in clients if c.t_arrive <= t < c.t_leave]
        log(f"round {r + 1}/{n_rounds} | t = {t} s | active: {len(active)}")

        # A client joins only if it can train and upload before leaving
        uploads = []
        for c in active:
            if c.can_finish(size, prec, c.t_leave - t):
                uploads.append((c.n_data, c.local_update(gmodel, device)))
        n_drop = len(active) - len(uploads)

        # No participant: the global model is left unchanged
        gmodel = aggregate(gmodel, uploads)

        acc = get_test_accuracy(test_loader, device, gmodel)
        loss = get_test_loss(test_loader, device, gmodel)
        log(f"  participants: {len(uploads)} | dropped: {n_drop}")
        log(f"  test acc: {acc:.2f} % | test loss: {loss:.4f}")
        if active:
            cacc = get_client_accuracy(active, gmodel, device)
            log(f"  client acc: mean {np.mean(cacc):.2f} % | "
                f"min {np.min(cacc):.2f} %")
