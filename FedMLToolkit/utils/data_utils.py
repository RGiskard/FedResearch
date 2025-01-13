#!/usr/bin/env python3
"""
Funciones de carga de datos (.pkl) y conversión a DataLoaders de PyTorch.
"""

import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_path, batch_size=32):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # data debe ser un diccionario con 'inputs' y 'labels'
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)

    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
