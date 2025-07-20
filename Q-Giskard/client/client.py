"""
q_giskard.client

Inference Engine for federated quantum training using Flower.

References:
- Kairouz, P., McMahan, H. B., et al. (2021). Advances and open problems in federated learning. *Foundations and Trends® in Machine Learning*, 14(1–2), 1–210.

GISKARD:
I - Inference Engine
A - Aggregation & Averaging
R - Remote Coordination
"""
import flwr as fl

class GiskardClient(fl.client.NumPyClient):
    def __init__(self, train_data, test_data, initial_params):
        self.x_train, self.y_train = train_data
        self.x_test,  self.y_test  = test_data
        self.params = initial_params

    def get_parameters(self):
        # Return flattened parameters for server aggregation
        return [self.params.flatten()]

    def fit(self, parameters, config):
        """
        Local training loop.
        Args:
            parameters (list[array]): global parameters from server
            config (dict): training config, e.g. {'lr': 0.1, 'local_epochs': 1}
        Returns:
            list[array], int, dict: updated parameters, train size, metrics
        """
        # TODO: update self.params, perform local epochs
        return [self.params.flatten()], len(self.x_train), {}

    def evaluate(self, parameters, config):
        """
        Local evaluation.
        Returns:
            loss (float), size (int), metrics (dict)
        """
        # TODO: compute local loss/accuracy
        return 0.0, len(self.x_test), {}
