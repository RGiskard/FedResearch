#!/usr/bin/env python3
"""
Lógica del servidor federado.
Selecciona estrategia, coordina rondas de entrenamiento y agrega parámetros.
"""

import json
import torch
from strategies.entropicfl_strategy import EntropicFLStrategy
from strategies.fedsnip_strategy import FedSNIPStrategy

class FederatedServer:
    def __init__(self, config):
        self.num_rounds = config.get("num_rounds", 1)
        self.divergence_threshold = config.get("divergence_threshold", 0.5)
        self.relevance_threshold = config.get("relevance_threshold", 0.7)
        self.strategy = None

        # Selección de estrategia como ejemplo (hard-coded)
        # En la práctica, esto podría cargarse de config o línea de comandos.
        self.strategy = EntropicFLStrategy() 
        # self.strategy = FedSNIPStrategy()  # Descomentar para usar FedSNIP

    def start(self):
        """
        Inicia el bucle de rondas federadas.
        """
        print(f"Inicializando servidor con {self.num_rounds} rondas.")
        for round_idx in range(self.num_rounds):
            print(f"\n--- Ronda {round_idx+1}/{self.num_rounds} ---")
            self.strategy.run_round()
