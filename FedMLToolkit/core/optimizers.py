#!/usr/bin/env python3
"""
Módulo para múltiples optimizadores (SGD, Adam, etc.)
"""

import torch

def get_optimizer(optimizer_name, model_parameters, lr=0.01):
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model_parameters, lr=lr)
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} no está implementado.")
