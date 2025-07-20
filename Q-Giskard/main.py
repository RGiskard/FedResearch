#!/usr/bin/env python3
"""
q_giskard.main

Deployment Entry Point (D in GISKARD): CLI to launch experiments.
"""
import argparse
from q_giskard.runner import run_experiment

def parse_args():
    parser = argparse.ArgumentParser(
        description="Q-Giskard: Federated Quantum Learning CLI"
    )
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients")
    return parser.parse_args()

def main():
    args = parse_args()
    run_experiment(n_qubits=args.qubits, num_clients=args.clients)

if __name__ == "__main__":
    main()
