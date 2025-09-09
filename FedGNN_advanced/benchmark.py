# benchmark.py
"""
Benchmark runner: runs FedAvg vs SCAFFOLD experiment (in-process simulation) on small dataset (Cora-like),
records per-round accuracies and computes (epsilon, delta) via RDP accountant for the given noise multiplier.
This is a simplified runner: replace model/data loader with your project-specific functions.
"""
import os
import json
import argparse
import random
import torch
import numpy as np
from privacy.rdp_accountant import compute_eps_for_gaussian_subsample

# placeholders to import your server/client code
from aggregator import FedAvgAggregator, SCAFFOLDAggregator
from learner import Learner
from server import BonawitzServer  # or your server implementation

def run_simulation(num_clients=10, clients_per_round=5, rounds=50, noise_multiplier=1.0, delta=1e-5):
    # This function must be adapted to your project specifics.
    # For demonstration, we only compute communication rounds -> privacy epsilon via RDP accountant.
    N = num_clients
    q = clients_per_round / N
    sigma = noise_multiplier
    steps = rounds
    orders = list(range(2, 128))
    eps, order, rdp = compute_eps_for_gaussian_subsample(q=q, sigma=sigma, steps=steps, delta=delta, orders=orders)
    print(f"RDP -> epsilon={eps:.4f} at order {order} for delta={delta}, q={q}, sigma={sigma}, steps={steps}")
    # Here you should run the actual experiment and record metrics; we'll output dummy values
    results = {
        'num_clients': num_clients,
        'clients_per_round': clients_per_round,
        'rounds': rounds,
        'noise_multiplier': noise_multiplier,
        'delta': delta,
        'epsilon': eps,
        'order': order,
        'rdp_sample': {str(k): float(v) for k,v in list(rdp.items())[:5]}
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/benchmark_privacy.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Benchmark privacy saved to outputs/benchmark_privacy.json")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=50)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    args = parser.parse_args()
    run_simulation(args.num_clients, args.clients_per_round, args.rounds, args.noise, args.delta)
