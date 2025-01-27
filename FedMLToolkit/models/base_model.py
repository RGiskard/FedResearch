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

    def fit(self, data_loader, optimizer, epochs=1, verbose=False, calculate_metrics=True):
        """
        Entrenar el modelo en el conjunto de datos proporcionado.

        Args:
            data_loader (DataLoader): DataLoader con los datos de entrenamiento.
            optimizer (Optimizer): Optimizador para actualizar los pesos del modelo.
            epochs (int): Número de épocas para entrenar.
            verbose (bool): Imprime métricas por época si es True.
            calculate_metrics (bool): Calcula y devuelve métricas como pérdida promedio y precisión.

        Returns:
            tuple: (avg_loss, accuracy) si calculate_metrics=True; de lo contrario, devuelve None.
        """
        self.train()
        device = next(self.parameters()).device
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total_samples = 0

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")

            for data, target in tqdm(data_loader, desc="Training", disable=not verbose):
                optimizer.zero_grad()

                # Mover datos y etiquetas al dispositivo
                data, target = data.to(device), target.to(device)

                # Forward y cálculo de pérdida
                output = self.forward(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                total_loss += loss.item()  # Acumular la pérdida

                # Backpropagación y optimización
                loss.backward()
                optimizer.step()

                # Calcular precisión solo si está habilitado
                if calculate_metrics:
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == target).sum().item()
                    total_samples += target.size(0)

            # Cálculo de métricas por época
            if calculate_metrics:
                avg_loss = total_loss / len(data_loader.dataset)
                accuracy = correct / total_samples
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Training completed")

        # Devolver métricas si están habilitadas
        if calculate_metrics:
            return avg_loss, accuracy
        return None


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
