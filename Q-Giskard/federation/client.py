"""
q_giskard.federation.client

Inference Engine for Federated Quantum Training.

References:
- Kairouz, P., et al. (2021). Advances and open problems in federated learning. Found. Trends® Mach. Learn. 14(1–2), 1–210.
"""
import flwr as fl

class GiskardClient(fl.client.NumPyClient):
    def __init__(self, train_data, test_data, initial_params):
        self.x_train, self.y_train = train_data
        self.x_test,  self.y_test  = test_data
        self.params = initial_params

    def get_parameters(self):
        return [self.params.flatten()]

    def fit(self, parameters, config):
        # TODO: update self.params and perform local training
        return [self.params.flatten()], len(self.x_train), {}

    def evaluate(self, parameters, config):
        # TODO: compute local loss/accuracy
        return 0.0, len(self.x_test), {}
