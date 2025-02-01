import os
import json
import csv
import flwr as fl
from typing import List, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import EvaluateRes, FitRes, ndarrays_to_parameters, parameters_to_ndarrays


class EntropicFLServer(fl.server.strategy.FedAvg):
    def __init__(self, config_path=None):
        """
        Initialize the server with configuration from a JSON file and set up logging directories for metrics.
        :param config_path: Path to the JSON configuration file.
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Create output directory for logs
        base_output_dir = self.config.get("output_dir", "logs")  # Default to "logs" if not specified
        output_dir = os.path.join(base_output_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        # Set paths for log files
        self.fit_log_file = os.path.join(output_dir, "fit_metrics.csv")
        self.eval_log_file = os.path.join(output_dir, "eval_metrics.csv")
        self.participation_log_file = os.path.join(output_dir, "client_participation.csv")

        # Initialize base strategy
        super().__init__(
            fraction_fit=self.config["fraction_fit"],
            min_fit_clients=self.config["min_fit_clients"],
            min_available_clients=self.config["min_available_clients"],
        )

        # Create log files if not present
        self._initialize_csv_files()

    def _load_config(self, config_path=None):
        """
        Load configuration from a JSON file. If no path is provided, a default path is used.
        :param config_path: Path to the configuration file.
        :return: Configuration dictionary.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs",
                "server_config.json",
            )

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as f:
            return json.load(f)

    def _initialize_csv_files(self):
        """Create CSV log files if they do not exist."""
        self._write_to_csv(self.fit_log_file, ["server_round", "aggregated_accuracy", "aggregated_loss", "available_clients"], init=True)
        self._write_to_csv(self.eval_log_file, ["server_round", "aggregated_accuracy", "aggregated_loss", "available_clients"], init=True)
        self._write_to_csv(self.participation_log_file, ["server_round", "client_id", "entropy", "participated"], init=True)

    def _write_to_csv(self, file_path, row, init=False):
        """
        Write a row to a CSV file. Create the file if it does not exist.
        :param file_path: Path to the CSV file.
        :param row: List representing a row to write.
        :param init: If True, treat the row as a header.
        """
        mode = "w" if init else "a"
        with open(file_path, mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def _select_clients_based_on_entropy(self, client_metrics):
        """
        Select exactly C clients based on the configured entropy criterion.
        """
        server_capacity = self.config["server_capacity"]
        criterion = self.config["entropy_criterion"]  # "greater", "lesser", "random"

        if criterion == "greater":
            sorted_clients = sorted(client_metrics, key=lambda x: x["entropy"], reverse=True)
        elif criterion == "lesser":
            sorted_clients = sorted(client_metrics, key=lambda x: x["entropy"])
        elif criterion == "random":
            import random
            sorted_clients = random.sample(client_metrics, len(client_metrics))
        else:
            raise ValueError(f"Unknown entropy criterion: {criterion}")

        return [client["cid"] for client in sorted_clients[:server_capacity]]

    def _log_participation(self, server_round, client_metrics, selected_clients):
        """
        Log which clients participated and their entropy in the current round.
        """
        rows = [
            [server_round, client["cid"], client["entropy"], client["cid"] in selected_clients]
            for client in client_metrics
        ]
        for row in rows:
            self._write_to_csv(self.participation_log_file, row)

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure the next round of training.
        On the first round, all clients participate.
        On subsequent rounds, selection is based on entropy.
        """
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        available_clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round == 1:
            selected_clients = [client.cid for client in available_clients]
            client_metrics = [{"cid": cid, "entropy": 0.0} for cid in selected_clients]
        else:
            client_metrics = [
                {"cid": fit_res.metrics["client_id"], "entropy": fit_res.metrics["entropy"]}
                for _, fit_res in self._fit_results
                if "entropy" in fit_res.metrics and "client_id" in fit_res.metrics
            ]
            selected_clients = self._select_clients_based_on_entropy(client_metrics)

        self._log_participation(server_round, client_metrics, selected_clients)

        config = {
            "server_round": server_round,
            "selected_clients": " ".join(map(str, selected_clients))  # Convertir IDs a cadenas
        }

        fit_ins = fl.common.FitIns(parameters, config)
        return [
            (client, fit_ins) for client in available_clients if client.cid in selected_clients
        ]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        print(f"Iniciando aggregate_fit para la ronda {server_round}")
        self._fit_results = results  # Save results for later use in configure_fit
        if not results:
            print(f"Round {server_round}: No results received from clients.")
            return None, {}

        # Calculate aggregated metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        accuracies = [fit_res.metrics["accuracy"] * fit_res.num_examples for _, fit_res in results]
        losses = [fit_res.metrics["loss"] * fit_res.num_examples for _, fit_res in results]

        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0.0

        # Write metrics to CSV
        self._write_to_csv(
            self.fit_log_file,
            [server_round, len(results), aggregated_loss, aggregated_accuracy]
        )

        print(f"Ronda {server_round}: Loss agregada: {aggregated_loss:.4f}, Accuracy agregada: {aggregated_accuracy:.4f}")

        # Aggregate parameters using FedAvg
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated, {"loss": aggregated_loss, "accuracy": aggregated_accuracy}


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):
        """
        Aggregate evaluation results sent by the clients.
        """
        if not results:
            print(f"Round {server_round}: No evaluation results received.")
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        losses = [res.metrics["loss"] * res.num_examples for _, res in results]

        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0
        aggregated_loss = sum(losses) / total_examples if total_examples > 0 else 0.0

        self._write_to_csv(self.eval_log_file, [server_round, aggregated_accuracy, aggregated_loss, len(results)])

        print(f"Round {server_round}: Aggregated loss: {aggregated_loss:.4f}, Aggregated accuracy: {aggregated_accuracy:.4f}")

        return aggregated_loss, {"aggregated_accuracy": aggregated_accuracy}
