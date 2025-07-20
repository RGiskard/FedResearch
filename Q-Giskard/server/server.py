"""
q_giskard.server

Starts Flower server for Q-Giskard federated quantum training.

References:
- Bonawitz, K., Eichner, H., et al. (2019). Towards federated learning at scale: system design. *Proceedings of MLSys*.

GISKARD:
A - Aggregation & Averaging
R - Remote Coordination
D - Distribution & Deployment
"""
import flwr as fl

def start_server():
    # Use FedAvg strategy over qubit parameter vectors
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
