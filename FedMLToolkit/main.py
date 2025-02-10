import flwr as fl
from client.entropicfl_client import EntropicFLClient
from strategies.entropicfl_strategy import EntropicFLServer

# Añade el directorio raíz al PYTHONPATH (opcional si usas la estructura del proyecto correctamente)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_ray_logs(enable=True, custom_dir=None):
    """
    Configura los directorios de logs de Ray si `enable=True`.

    Parámetros:
        - enable (bool): Si es True, configura Ray con logs en un directorio personalizado.
        - custom_dir (str): Ruta personalizada para los logs de Ray. Si es None, usa '~/ray_logs'.
    """
    if not enable:
        print("⚠️ Ray no ha sido configurado.")
        return  # No hace nada si enable=False

    # Importar Ray solo si se va a usar
    import ray

    # Definir el directorio de logs
    log_directory = custom_dir if custom_dir else os.path.join(os.path.expanduser("~"), "ray_logs")

    # Crear la carpeta si no existe
    os.makedirs(log_directory, exist_ok=True)

    # Configurar variables de entorno
    os.environ["RAY_TEMP_DIR"] = log_directory
    os.environ["RAY_LOG_TO_STDERR"] = "0"
    os.environ["RAY_LOG_DIR"] = log_directory

    # Inicializar Ray si no está inicializado
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print(f"✅ Ray configurado correctamente en {log_directory}")

def client_fn(cid: str):
    """Crea una instancia del cliente EntropicFLClient."""
    return EntropicFLClient(cid=cid)

# Configura el servidor
server_strategy = EntropicFLServer()

# Ejecuta la simulación
if __name__ == "__main__":
    setup_ray_logs(enable=False)  # Cambia a False si no quieres configurar Ray

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=server_strategy.config["min_available_clients"],  # Leído desde el archivo de configuración
        strategy=server_strategy,
        config=fl.server.ServerConfig(num_rounds=server_strategy.config["num_rounds"]),  # Rondas también desde el archivo
    )
    # (Opcional) Analiza y guarda los resultados
    print("Simulation complete!")
    print(history)
