# client_manager.py
import random
import numpy as np
import torch

class ClientManager:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def select_clients(self, round_idx, clients, num_select, strategy='random'):
        n = len(clients)
        if num_select is None or num_select >= n:
            return clients
        if strategy == 'random':
            return random.sample(clients, num_select)
        if strategy == 'pow_d':
            # weighted by num_samples if available
            sizes = [getattr(c, 'num_samples', 1) for c in clients]
            probs = np.array(sizes, dtype=float)
            probs = probs / probs.sum()
            idxs = np.random.choice(range(n), size=num_select, replace=False, p=probs)
            return [clients[i] for i in idxs]
        if strategy == 'loss_driven':
            # pick clients with larger last_loss (if history exists)
            losses = []
            for c in clients:
                h = getattr(c, 'history', None)
                if h and len(h)>0:
                    losses.append(h[-1].get('loss', 0.0))
                else:
                    losses.append(0.0)
            probs = np.array(losses, dtype=float)
            if probs.sum() == 0:
                probs = np.ones_like(probs)
            probs = probs / probs.sum()
            idxs = np.random.choice(range(n), size=num_select, replace=False, p=probs)
            return [clients[i] for i in idxs]
        # fallback
        return random.sample(clients, num_select)
