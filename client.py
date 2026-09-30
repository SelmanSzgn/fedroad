import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from model import Model
from data import sample_local_data


class Client:
    def __init__(
            self, 
            cid, 
            t_arrive, 
            kph, 
            t_leave, 
            n_data, 
            local_data, 
            cpu_hz, 
            batch, 
            epochs, 
            cpu_cycles, 
            eff_capa, 
            snr_db, 
            bw_hz,
            ptx, 
            lr, 
            momentum, 
            decay
        ):
        # client id
        self.cid = cid
        # arrival timestamp
        self.t_arrive = t_arrive
        # speed in kilometer per hour
        self.kph = kph
        # speed in meter per second
        self.mps = kph / 3.6
        # leaving timestamp
        self.t_leave = t_leave
        # number of local data
        self.n_data = n_data
        # local dataset
        self.local_data = local_data
        # cpu frequency in hertz
        self.cpu_hz = cpu_hz
        # batch size
        self.batch = batch
        # number of local epochs
        self.epochs = epochs
        # number of cpu cycles per data
        self.cpu_cycles = cpu_cycles
        # effective capacitance
        self.eff_capa = eff_capa
        # channel signal-to-noise ratio in decibel
        self.snr_db = snr_db
        # channel signal-to-noise ratio in linear scale
        self.snr_linear = 10 ** (snr_db / 10)
        # allocated bandwidth
        self.bw_hz = bw_hz
        # client transmission power in watts
        self.ptx = ptx
        # learning rate
        self.lr = lr
        # sgd momentum
        self.momentum = momentum
        # sgd weight decay
        self.decay = decay

    def get_cp_time(self):
        """Compute client computation time (seconds)."""
        return (self.cpu_cycles*self.n_data*self.epochs) / self.cpu_hz
    
    def get_cp_energy(self):
        """Compute client communication energy (joules)."""
        return self.get_cp_time()*self.eff_capa*(self.cpu_hz ** 3)
    
    def get_throughput(self):
        """Compute client uplink throughput (bit per second)."""
        return self.bw_hz * np.log2(1 + self.snr_linear)
    
    def get_co_time(self, model_size, model_precision):
        """Compute client communication time (seconds)."""
        bits = model_size * model_precision
        return bits / self.get_throughput()
    
    def get_co_energy(self, model_size, model_precision):
        """Compute client communication energy (joules)."""
        return self.ptx * self.get_co_time(model_size, model_precision)
    
    def local_update(self, global_model, device):
        """Run local training."""
        trainloader = torch.utils.data.DataLoader(
            self.local_data, 
            batch_size=self.batch, 
            shuffle=True
        )
        model = Model()
        model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), 
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.decay
        )
        for _ in range(self.epochs):
            model.train()
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model
    


def get_arrivals(sim_time, poisson):
    """Compute all clients arrival timestamps according to Poisson process."""
    arrivals = []
    timestamp = 0
    stop = False
    while not stop:
        timestamp += np.random.exponential(1/poisson)
        if timestamp < sim_time:
            arrivals.append(timestamp)
        else:
            stop = True
    return arrivals

def get_all_clients(
        arrivals, 
        trainset, 
        class_indices, 
        n_sub_classes, 
        min_kph,
        max_kph, 
        road_length_m, 
        min_n_data, 
        max_n_data, 
        min_cpu_hz,
        max_cpu_hz, 
        batch, 
        epochs, 
        cpu_cycles,
        eff_capa, 
        snr_db_min, 
        snr_db_max, 
        bw_hz, 
        ptx, 
        lr, 
        momentum, 
        decay
    ):
    """Create all clients."""
    all_clients = []
    for i in range(len(arrivals)):
        kph = np.random.uniform(min_kph, max_kph)
        mps = kph / 3.6
        t_leave = arrivals[i] + road_length_m / mps
        n_data = np.random.randint(min_n_data, max_n_data + 1)
        local_data = sample_local_data(
            trainset, 
            class_indices, 
            n_data, 
            n_sub_classes
        )
        cpu_hz = np.random.uniform(min_cpu_hz, max_cpu_hz)
        snr_db = np.random.uniform(snr_db_min, snr_db_max)
        client = Client(
            i, 
            arrivals[i], 
            kph, 
            t_leave, 
            n_data, 
            local_data,
            cpu_hz, 
            batch, 
            epochs, 
            cpu_cycles,
            eff_capa, 
            snr_db, 
            bw_hz, 
            ptx, 
            lr, 
            momentum, 
            decay
        )
        all_clients.append(client)
    return all_clients

