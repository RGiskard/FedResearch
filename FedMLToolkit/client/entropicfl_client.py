#!/usr/bin/env python3
"""
EntropicFLClient

Cliente federado para Flower con soporte para modelos dinámicos, modos de datos configurables y guardado de métricas.
"""

import os
import json
import csv
import torch
import flwr as fl
from utils.data_utils import load_data
from models.lenet5 import LeNet5
from models.alexnet import AlexNet
from models.vgg import VGG


class EntropicFLClient(fl.client.NumPyClient):
    def __init__(self, cid, config_path=None):
        # Cargar configuración
        self.config = self._load_config(config_path)

        # Inicializar cliente
        self.cid = cid
        self._initialize_client()

    def _load_config(self, config_path=None):
        """Cargar configuración desde un archivo JSON."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "client_config.json")
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_client(self):
        """Inicializar el cliente con los parámetros de configuración."""
        # Configuración del cliente
        self.data_mode = self.config.get("data_mode", "static").lower()  # static o dynamic
        self.data_path = self.config["data_path"]
        self.data_template = self.config.get("data_template", None)
        self.model_name = self.config.get("model", "lenet")
        self.optimizer_name = self.config.get("optimizer", "adam")
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.epochs = self.config.get("epochs", 5)
        self.batch_size = self.config.get("batch_size", 32)
        self.verbose = self.config.get("verbose", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validar modo de datos
        if self.data_mode not in ["static", "dynamic"]:
            raise ValueError(f"Modo de datos no válido: {self.data_mode}. Use 'static' o 'dynamic'.")

        # Inicializar modelo y optimizador
        self.model = self._load_model().to(self.device)
        self.optimizer = self._load_optimizer()

        # Inicializar archivo de métricas
        self.metrics_file = f"client_{self.cid}_metrics.csv"
        self._create_csv_header()

    def _load_model(self):
        """Cargar el modelo según la configuración."""
        models = {
            "lenet": LeNet5,
            "alexnet": AlexNet,
            "vgg": VGG
        }
        if self.model_name.lower() not in models:
            raise ValueError(f"Modelo no soportado: {self.model_name}")
        return models[self.model_name.lower()]()

    def _load_optimizer(self):
        """Cargar el optimizador según la configuración."""
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop
        }
        if self.optimizer_name.lower() not in optimizers:
            raise ValueError(f"Optimizador no soportado: {self.optimizer_name}")
        return optimizers[self.optimizer_name.lower()](self.model.parameters(), lr=self.learning_rate)

    def _resolve_data_path(self, round=None):
        """
        Resolver la ruta de los datos según el modo:
        - Modo estático: Se usa solo el cid.
        - Modo dinámico: Se usa cid y round.
        """
        if self.data_mode == "dynamic":
            if self.data_template is None:
                raise ValueError("La plantilla de datos (data_template) no está definida en el modo dinámico.")
            if round is None:
                raise ValueError("El número de ronda ('round') es obligatorio en el modo dinámico.")
            return self.data_template.format(cid=self.cid, round=round)
        elif self.data_mode == "static":
            return self.data_path.format(cid=self.cid)
        else:
            raise ValueError(f"Modo de datos no reconocido: {self.data_mode}")

    def _load_data(self, round=None):
        """Cargar los datos de entrenamiento y prueba."""
        data_path = self._resolve_data_path(round)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")
        return load_data(data_path, batch_size=self.batch_size)

    def _create_csv_header(self):
        """Crear el archivo CSV con los encabezados si no existe."""
        fieldnames = [
            "Client ID", "Epoch", "Loss", "Divergence", "Relevance",
            "Accuracy", "Training Time (s)"
        ]
        if not os.path.isfile(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

    def _save_metrics(self, metrics):
        """Guardar las métricas en un archivo CSV."""
        with open(self.metrics_file, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writerow(metrics)

    def get_parameters(self, config):
        """Obtener los parámetros actuales del modelo."""
        return [param.data.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """Establecer los parámetros recibidos en el modelo."""
        for param, new_data in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_data).to(self.device)

    def fit(self, parameters, config):
        """Entrenar el modelo localmente."""
        # Verificar si existe la variable de cids en la configuración
        if "selected_cids" not in config:
            raise ValueError("La lista de cids seleccionados ('selected_cids') no fue proporcionada por el servidor.")

        # Parsear la lista de cids seleccionados
        selected_cids = config["selected_cids"].split(",")

        # Si el cliente no está seleccionado, no entrena pero devuelve los parámetros
        if str(self.cid) not in selected_cids:
            print(f"Cliente {self.cid} no seleccionado para entrenar en esta ronda.")
            return self.get_parameters(config), len(self._load_data()[0].dataset), {"skipped": True}

        # Cargar datos según el modo y ronda
        round_number = config.get("round", None)
        self.train_loader, self.test_loader = self._load_data(round=round_number)

        # Entrenar utilizando el método `fit` del modelo
        self.model.fit(
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            epochs=self.epochs,
            verbose=self.verbose
        )

        # Guardar métricas de entrenamiento en archivo CSV
        self._save_metrics({
            "Client ID": self.cid,
            "Epoch": self.epochs,
            "Loss": None,
            "Divergence": None,
            "Relevance": None,
            "Accuracy": None,
            "Training Time (s)": None
        })

        # Devolver los parámetros actualizados
        return self.get_parameters(config), len(self.train_loader.dataset), {"skipped": False}

    def evaluate(self, parameters, config):
        """Evaluar el modelo localmente."""
        # Establecer los parámetros recibidos
        self.set_parameters(parameters)

        # Cargar datos (modo estático por defecto para evaluación)
        self.train_loader, self.test_loader = self._load_data()

        # Evaluar utilizando el método `evaluate` del modelo
        avg_loss, accuracy = self.model.evaluate(
            data_loader=self.test_loader,
            verbose=self.verbose
        )

        # Guardar métricas de evaluación en archivo CSV
        self._save_metrics({
            "Client ID": self.cid,
            "Epoch": None,
            "Loss": avg_loss,
            "Divergence": None,
            "Relevance": None,
            "Accuracy": accuracy,
            "Training Time (s)": None
        })

        # Devolver pérdida, tamaño de conjunto y métricas adicionales
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy}
