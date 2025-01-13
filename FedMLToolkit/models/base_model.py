#!/usr/bin/env python3
"""
Clase base para los modelos.
Incluye métodos fit y evaluate con soporte para tqdm y verbose.
"""

import torch
from tqdm import tqdm

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def fit(self, data_loader, optimizer, epochs=1, verbose=False):
        self.train()
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            for data, target in tqdm(data_loader, desc="Training", disable=not verbose):
                optimizer.zero_grad()
                output = self.forward(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

    def evaluate(self, data_loader, verbose=False):
        self.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating", disable=not verbose):
                output = self.forward(data)
                loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_samples += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        if verbose:
            print(f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
