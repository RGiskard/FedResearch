#!/usr/bin/env python3
"""
Funciones de entrenamiento y evaluación.
"""

import torch
import torch.nn.functional as F

def train_local(model, data_loader, optimizer=None, epochs=1):
    """
    Entrena el modelo localmente usando los datos del data_loader.
    Retorna la pérdida total.
    """
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    total_loss = 0.0

    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss
