"""
q_giskard.runner

Runner Orchestration module (R in GISKARD):
Spins up Flower server and simulated clients.
"""
import threading
import numpy as np
import flwr as fl

from q_giskard.federation.server import start_server
from q_giskard.federation.client import GiskardClient

def make_dummy_dataset(n, q):
    X = np.random.randn(n, q)
    y = np.random.randint(0, 2, size=(n,))
    return (X, y)

def run_client_instance(init_params, train, test):
    client = GiskardClient(train, test, init_params)
    fl.client.start_numpy_client("127.0.0.1:8080", client=client)

def run_experiment(n_qubits: int, num_clients: int):
    init_params = np.random.randn(2, n_qubits)
    threading.Thread(target=start_server, daemon=True).start()
    threads = []
    for _ in range(num_clients):
        train, test = make_dummy_dataset(100, n_qubits), make_dummy_dataset(30, n_qubits)
        t = threading.Thread(target=run_client_instance, args=(init_params, train, test))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("✅ Federated quantum experiment completed!")
