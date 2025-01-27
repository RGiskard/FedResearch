#!/usr/bin/env python3
"""
Funciones de carga de datos (.pkl) y conversión a DataLoaders de PyTorch.
"""

import pickle
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_path_template, cid, data_mode="static", round=None, batch_size=32, device="cpu"):
    """
    Cargar datos desde un archivo .npz en modo estático o dinámico.
    
    Args:
        data_path_template (str): Plantilla para generar la ruta del archivo.
        cid (int): ID del cliente.
        data_mode (str): Modo de datos ("static" o "dynamic").
        round (int, opcional): Número de ronda (requerido en modo dinámico).
        batch_size (int): Tamaño del batch para el DataLoader.
        device (str): Dispositivo al que se transferirán los tensores ("cpu" o "cuda").
    
    Returns:
        DataLoader: Un DataLoader con los datos cargados en el dispositivo especificado.
    """
    # Resolver la ruta del archivo dependiendo del modo
    if data_mode == "dynamic":
        if round is None:
            raise ValueError("El número de ronda ('round') es obligatorio en el modo dinámico.")
        data_path = data_path_template.format(cid=cid, round=round)
    elif data_mode == "static":
        data_path = data_path_template.format(cid=cid)
    else:
        raise ValueError(f"Modo de datos no reconocido: {data_mode}")

    # Verificar que el archivo exista
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    # Cargar datos desde el archivo .npz
    with np.load(data_path, allow_pickle=True) as data:
        # Convertir x_train y y_train a arrays numéricos si es necesario
        x_train = np.array(data['x_train'], dtype=np.float32)
        y_train = np.array(data['y_train'], dtype=np.int64)
        x_train = np.expand_dims(x_train, axis=1)  # De [batch_size, height, width] a [batch_size, 1, height, width]

        # Convertir los datos a tensores de PyTorch y transferir al dispositivo
        inputs = torch.tensor(x_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)

    # Crear el dataset y DataLoader
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def preprocess_data(data):
    """
    Normalizar y reformatear las imágenes para que sean compatibles con el modelo.

    Args:
        data (np.ndarray): Conjunto de datos a preprocesar.

    Returns:
        np.ndarray: Datos normalizados y reformateados.
    """
    data = data / 255.0  # Normalización al rango [0, 1]
    data = np.expand_dims(data, axis=1)  # Añadir un canal extra
    return data

def load_validation_data(validation_file, batch_size=32, device="cpu"):
    """
    Cargar datos de validación desde un único archivo .npz.

    Args:
        validation_file (str): Ruta al archivo de validación.
        batch_size (int): Tamaño del batch para el DataLoader.
        device (str): Dispositivo al que se transferirán los tensores ("cpu" o "cuda").

    Returns:
        DataLoader: Un DataLoader con los datos cargados en el dispositivo especificado.
    """
    # Verificar que el archivo exista
    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"No se encontró el archivo de validación: {validation_file}")

    # Cargar datos desde el archivo .npz
    with np.load(validation_file, allow_pickle=True) as data:
        # Verificar que contenga datos de validación
        if 'x_test' not in data or 'y_test' not in data:
            raise KeyError(f"El archivo {validation_file} no contiene datos de validación ('x_val', 'y_val').")

        # Convertir x_val y y_val a arrays numéricos si es necesario
        x_val = np.array(data['x_test'], dtype=np.float32)
        #x_val = preprocess_data(np.array(data['x_test'], dtype=np.float32))
        y_val = np.array(data['y_test'], dtype=np.int64)
        x_val = np.expand_dims(x_val, axis=1)  # De [batch_size, height, width] a [batch_size, 1, height, width]

        # Convertir los datos a tensores de PyTorch y transferir al dispositivo
        inputs = torch.tensor(x_val, dtype=torch.float32).to(device)
        labels = torch.tensor(y_val, dtype=torch.long).to(device)

    # Crear el dataset y DataLoader
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def calculate_entropy(labels):
    """Calcular la entropía de un conjunto de etiquetas."""
    from collections import Counter
    from scipy.stats import entropy

    label_counts = Counter(labels)
    total = sum(label_counts.values())
    probabilities = [count / total for count in label_counts.values()]
    return entropy(probabilities, base=2)