"""
q_giskard.federation.server

Flower Server for Federated Quantum Learning.

References:
- Bonawitz, K., et al. (2019). Towards Federated Learning at Scale: System Design. MLSys.
"""
import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
