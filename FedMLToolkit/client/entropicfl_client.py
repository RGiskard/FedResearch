#!/usr/bin/env python3
"""
EntropicFLClient

Características:
 - Entrenamiento local con cálculo de divergencia.
 - Soporte para verbose y barras de progreso (tqdm).
 - Guarda métricas de ejecución (pérdida, divergencia, relevancia) en SQLite.
 - Lectura de datos desde archivos .pkl configurados en client_config.json.
"""

import os
import json
import sqlite3
from tqdm import tqdm
import torch

from utils.data_utils import load_data
from core.training import train_local

class EntropicFLClient:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.db_path = self.config["db_path"]
        self.data_path = self.config["data_path"]

        # Conexión a la base de datos SQLite
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        # Crear tabla de métricas si no existe
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            round INTEGER,
            loss REAL,
            divergence REAL,
            relevance REAL
        )
        """)
        self.conn.commit()

    def train(self, model):
        data_loader = load_data(self.data_path)
        total_loss = 0.0
        divergence = 0.0
        relevance = 0.0

        # Ejemplo de bucle de entrenamiento
        for epoch in tqdm(range(1, 6), desc="Local training (EntropicFL)", leave=False):
            epoch_loss = train_local(model, data_loader)
            total_loss += epoch_loss

            # Ejemplos de cálculo de divergencia y relevancia
            if self.config.get("use_divergence", False):
                divergence += 0.01  # Placeholder
            if self.config.get("use_relevance", False):
                relevance += 0.02   # Placeholder

        # Guardar métricas en la base de datos
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO metrics (round, loss, divergence, relevance) VALUES (?, ?, ?, ?)",
                       (1, total_loss, divergence, relevance))
        self.conn.commit()

        # Si no cumple umbrales, retornamos null
        if divergence >  self.config.get("divergence_threshold", 9999) or \
           relevance  <  self.config.get("relevance_threshold", -1):
            return None
        else:
            # Devolver parámetros del modelo (placeholder)
            return [param.data for param in model.parameters()]

    def close(self):
        self.conn.close()
