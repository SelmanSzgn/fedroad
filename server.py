import torch


def aggregate(global_model, uploads):
    state = global_model.state_dict()
    with torch.no_grad():
        for key in state.keys():
            state[key] = sum(m[0] * m[1].state_dict()[key] for m in uploads)
    global_model.load_state_dict(state)
    return global_model

