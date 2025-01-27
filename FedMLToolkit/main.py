import flwr as fl
from client.entropicfl_client import EntropicFLClient
from strategies.entropicfl_strategy import EntropicFLServer

# Añade el directorio raíz al PYTHONPATH (opcional si usas la estructura del proyecto correctamente)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def client_fn(cid: str):
    """Crea una instancia del cliente EntropicFLClient."""
    return EntropicFLClient(cid=int(cid))

# Configura el servidor
server_strategy = EntropicFLServer()

# Ejecuta la simulación
if __name__ == "__main__":
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=server_strategy.config["min_available_clients"],  # Leído desde el archivo de configuración
        strategy=server_strategy,
        config=fl.server.ServerConfig(num_rounds=server_strategy.config["num_rounds"]),  # Rondas también desde el archivo
    )
    # (Opcional) Analiza y guarda los resultados
    print("Simulation complete!")
    print(history)
