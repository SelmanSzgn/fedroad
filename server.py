import torch


def aggregate(model, uploads):
    """FedAvg. uploads: list of (n_data, client_model)."""
    if not uploads:
        return model
    total = sum(n for n, _ in uploads)
    states = [(n / total, m.state_dict()) for n, m in uploads]
    new = {}
    with torch.no_grad():
        for k in model.state_dict():
            new[k] = sum(w * s[k] for w, s in states)
    model.load_state_dict(new)
    return model
