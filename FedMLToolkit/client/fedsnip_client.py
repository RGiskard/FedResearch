#!/usr/bin/env python3
"""
FedSNIPClient

Características:
 - Similar a EntropicFLClient, pero incluye lógica de poda SNIP
   antes de cada entrenamiento local.
 - Soporte para verbose y barras de progreso (tqdm).
 - Guarda métricas en SQLite.
"""

import os
import json
import sqlite3
from tqdm import tqdm
import torch

from utils.data_utils import load_data
from core.training import train_local
from core.pruning import snip_pruning, apply_pruning

class FedSNIPClient:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.db_path = self.config["db_path"]
        self.data_path = self.config["data_path"]
        self.pruning_ratio = self.config["pruning_ratio"]

        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            round INTEGER,
            loss REAL
        )
        """)
        self.conn.commit()

    def train(self, model):
        # Aplicar SNIP pruning
        masks = snip_pruning(model, self.pruning_ratio)
        apply_pruning(model, masks)

        data_loader = load_data(self.data_path)
        total_loss = 0.0

        for epoch in tqdm(range(1, 6), desc="Local training (FedSNIP)", leave=False):
            epoch_loss = train_local(model, data_loader)
            total_loss += epoch_loss

        # Guardar métricas en DB
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO metrics (round, loss) VALUES (?, ?)",
                       (1, total_loss))
        self.conn.commit()

        # Devuelve parámetros del modelo (placeholder)
        return [param.data for param in model.parameters()]

    def close(self):
        self.conn.close()
