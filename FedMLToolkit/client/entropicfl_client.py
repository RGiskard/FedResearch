#!/usr/bin/env python3
"""
EntropicFLClient

Cliente federado para Flower con soporte para modelos dinámicos, modos de datos configurables y guardado de métricas.
"""

import os
import json
import csv
import torch
import time  # Importar el módulo time
import flwr as fl
from utils.data_utils import load_data, load_validation_data
from models.lenet5 import LeNet5
from models.alexnet import AlexNet
from models.vgg import VGG


class EntropicFLClient(fl.client.NumPyClient):
    def __init__(self, cid, config_path=None):
        # Cargar configuración
        self.config = self._load_config(config_path)

        # Inicializar cliente
        self.cid = cid
        self.first_round = True  # Bandera para indicar si es la primera ronda
        self._initialize_client()

    def _load_config(self, config_path=None):
        """Cargar configuración desde un archivo JSON."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "client_config.json"
            )
        with open(config_path, 'r') as f:
            return json.load(f)

    def _initialize_client(self):
        """Inicializar el cliente con los parámetros de configuración."""
        # Configuración del cliente
        self.data_mode = self.config.get("data_mode", "static").lower()
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


        base_output_dir = self.config.get("output_dir")
        output_dir = os.path.join(base_output_dir, f"results")
        # Directorio de salida
        os.makedirs(output_dir, exist_ok=True)  # Crear directorio si no existe
        # Inicializar archivo de métricas
        self.metrics_file = os.path.join(output_dir, f"client_{self.cid}_metrics.csv")
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
        """Resolver la ruta de los datos según el modo."""
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
        """Cargar datos de entrenamiento según el modo."""
        train_loader = load_data(
            data_path_template=self.data_template if self.data_mode == "dynamic" else self.data_path,
            cid=self.cid,
            data_mode=self.data_mode,
            round=round,
            batch_size=self.batch_size,
            device=self.device
        )
        return train_loader

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

    def _load_validation_data(self):
        """Cargar datos de validación."""
        validation_file = self.config.get("validation_file", None)
        if validation_file is None:
            raise ValueError("El archivo de validación ('validation_file') no está especificado en la configuración.")
        validation_loader = load_validation_data(
            validation_file=validation_file,
            batch_size=self.batch_size,
            device=self.device
        )
        return validation_loader


    def _calculate_entropy(self, data_loader):
        """
        Calcular la entropía de Shannon usando los datos de un DataLoader.
        :param data_loader: DataLoader de PyTorch con los datos.
        :return: Entropía calculada.
        """
        import numpy as np

        # Acumular frecuencias relativas de las clases
        class_counts = {}
        total_samples = 0

        for inputs, labels in data_loader:
            labels = labels.cpu().numpy()
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += len(labels)

        # Convertir a probabilidades
        probabilities = np.array(list(class_counts.values())) / total_samples

        # Calcular entropía de Shannon
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Evitar log(0)

        return entropy



    def get_parameters(self, config):
        """Obtener los parámetros actuales del modelo."""
        return [param.data.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """Establecer los parámetros recibidos en el modelo."""
        for param, new_data in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_data).to(self.device)

    def fit(self, parameters, config):
        """Entrenar el modelo localmente y enviar la entropía al servidor."""

        # 🔍 Verificar los parámetros recibidos
        print(f"\n🔎 Cliente {self.cid}: Configuración recibida en `fit()`: {config}")

        # Verificar si `selected_clients` está en `config`
        selected_clients = config.get("selected_clients", None)
        if selected_clients is not None:
            print(f"🔹 Cliente {self.cid}: Lista de clientes seleccionados para esta ronda: {selected_clients}")

        # Establecer los parámetros recibidos
        self.set_parameters(parameters)
        print(f" Cliente {self.cid}: Ejecutando fit() en la ronda {config.get('server_round', 'Desconocida')}")
        
        # Verificar si es la primera ronda
        if self.first_round:
            print(f"Cliente {self.cid}: Primera ronda detectada. Configuración predeterminada aplicada.")
            round_number = 1
        else:
            round_number = config.get("server_round", None)
            if round_number is None:
                raise ValueError("El número de ronda ('server_round') es obligatorio después de la primera ronda.")

        # Cargar los datos de entrenamiento para la ronda actual
        self.train_loader = self._load_data(round=round_number)

        # Medir el tiempo de inicio
        start_time = time.time()

        # Entrenar el modelo y calcular la pérdida y precisión
        loss, acc = self.model.fit(
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            epochs=self.epochs,
            verbose=self.verbose
        )

        # Medir el tiempo de fin
        end_time = time.time()
        training_time = end_time - start_time

        # Calcular la entropía para la siguiente ronda
        entropy = self._calculate_entropy(self._load_data(round_number+1))
        print(f"La entropia para el cliente:{self.cid} es {entropy}")

        # Guardar métricas en un archivo CSV
        self._save_metrics({
            "Client ID": self.cid,
            "Epoch": self.epochs,
            "Loss": loss,
            "Divergence": None,
            "Relevance": None,
            "Accuracy": acc,
            "Training Time (s)": training_time
        })

        self.first_round = False

        # Retornar los parámetros, el tamaño de los datos y las métricas (incluyendo la entropía)
        return self.get_parameters(config), len(self.train_loader.dataset), {
            "cid": str(self.cid),
            "accuracy": acc,
            "loss": loss,
            "entropy": entropy,
            "skipped": False
        }


    
    def evaluate(self, parameters, config):
        """Evaluar el modelo localmente con datos de validación."""
        self.set_parameters(parameters)
        validation_loader = self._load_validation_data()

        avg_loss, accuracy = self.model.evaluate(
            data_loader=validation_loader,
            verbose=self.verbose
        )

#        self._save_metrics({
#            "Client ID": self.cid,
#            "Epoch": None,
#            "Loss": avg_loss,
#            "Divergence": None,
#            "Relevance": None,
#            "Accuracy": accuracy,
#            "Training Time (s)": None
#        })

        return avg_loss, len(validation_loader.dataset), {"accuracy": accuracy, "loss":avg_loss}
