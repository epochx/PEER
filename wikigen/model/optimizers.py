import torch

optimizers = vars(torch.optim)
optimizers = {
    name: optim for name, optim in optimizers.items()
    if '__' not in name and name != 'Optimizer'}
