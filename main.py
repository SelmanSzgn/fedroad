
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import yaml


from client import Client, get_all_clients, get_arrivals
from data import get_trainset, get_test_loader, create_class_indices
from model import Model
from server import aggregate
from eval import get_test_accuracy, get_test_loss, get_client_accuracy, get_client_loss


if __name__ == "__main__":

    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["numpy_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = Model()
    model_size = sum(
        p.numel() for p in global_model.parameters() if p.requires_grad)

    trainset = get_trainset()
    test_loader = get_test_loader()
    class_indices = create_class_indices(trainset)

    arrivals = get_arrivals(cfg["simulation_time_s"], cfg["poisson_rate"])
    all_clients = get_all_clients(
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
        float(cfg["weight_decay"])
    )

    n_rounds = int(cfg["simulation_time_s"] / cfg["round_duration_s"])

    for _round in range(n_rounds):
        n_drop = 0
        uploads = []
        active_clients = [
            client for client in all_clients
            if (
                client.t_arrive <= _round*cfg["round_duration_s"]
                and client.t_leave > _round*cfg["round_duration_s"]
            )
        ]
        n_active_clients = len(active_clients)
        if n_active_clients == 0:
            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] "
                f"round {_round + 1}/{n_rounds}")
            print("  No active client, continue.")
        else:
            tot_n_data = sum([client.n_data for client in active_clients])
            print(f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] "
                f"round {_round + 1}/{n_rounds}")
            print(f"  Number of active clients: {n_active_clients}")
            for i, client in enumerate(active_clients):
                cp_time = client.get_cp_time()
                cp_energy = client.get_cp_energy()
                co_time = client.get_co_time(
                    model_size, cfg["model_precision"]
                )
                co_energy = client.get_co_energy(
                    model_size, cfg["model_precision"]
                )
                if cp_time + co_time <= client.t_leave:
                    local_model = client.local_update(global_model, device)
                    uploads.append((client.n_data / tot_n_data, local_model))
                else:
                    uploads.append((client.n_data / tot_n_data, global_model))
                    n_drop += 1
            global_model = aggregate(global_model, uploads)

            test_accuracy = get_test_accuracy(test_loader, device, global_model)
            test_loss = get_test_loss(test_loader, device, global_model)
            client_accuracy = get_client_accuracy(active_clients, global_model, device)
            client_loss = get_client_loss(active_clients, global_model, device)

                
            print(f"  Test accuracy: {test_accuracy:3g} %")

