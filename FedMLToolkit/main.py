#!/usr/bin/env python3
"""
Archivo principal para iniciar el servidor de aprendizaje federado.

Este script cargará la configuración del servidor, inicializará la estrategia
correspondiente y comenzará el bucle de entrenamiento federado.
"""

import json
import os
from server.server import FederatedServer

def main():
    # Cargar configuración del servidor
    config_path = os.path.join("configs", "server_config.json")
    with open(config_path, 'r') as f:
        server_config = json.load(f)

    # Inicializar servidor federado
    server = FederatedServer(config=server_config)
    server.start()

if __name__ == "__main__":
    main()
