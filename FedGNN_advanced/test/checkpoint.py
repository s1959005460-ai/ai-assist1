# checkpoint.py
import os
import torch
import json
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, out_dir='./checkpoints'):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _checkpoint_path(self, round_idx):
        return os.path.join(self.out_dir, f"checkpoint_round_{round_idx}.pt")

    def save_checkpoint(self, round_idx, server, aggregator=None, clients=None, extra: dict = None):
        """
        Save atomic checkpoint containing:
        - round
        - global_model_state
        - aggregator_state
        - client states (if client implements get_state())
        - random states (python, numpy, torch)
        """
        ckpt = {}
        ckpt['round'] = int(round_idx)
        ckpt['global_model'] = server.global_model.state_dict()
        if aggregator is not None:
            try:
                ckpt['aggregator_state'] = aggregator.state_dict()
            except Exception:
                ckpt['aggregator_state'] = None
        # client states
        client_states = {}
        if clients:
            for c in clients:
                try:
                    client_states[getattr(c, 'id', str(id(c)))] = c.get_state()
                except Exception:
                    client_states[getattr(c, 'id', str(id(c)))] = None
        ckpt['client_states'] = client_states
        ckpt['random_state'] = random.getstate()
        ckpt['numpy_random_state'] = np.random.get_state()
        ckpt['torch_random_state'] = torch.get_rng_state()
        # optional extra metadata
        if extra:
            ckpt['extra'] = extra

        tmp = self._checkpoint_path(f"{round_idx}.tmp")
        final = self._checkpoint_path(round_idx)
        # write atomically
        torch.save(ckpt, tmp)
        os.replace(tmp, final)
        logger.info("Saved checkpoint to %s", final)
        return final

    def load_checkpoint(self, path, server=None, aggregator=None, clients=None, device='cpu'):
        """
        Load checkpoint and restore to server/aggregator/clients if provided.
        Returns dict of checkpoint content.
        """
        ckpt = torch.load(path, map_location=device)
        # restore model
        if server is not None and 'global_model' in ckpt:
            server.global_model.load_state_dict(ckpt['global_model'])
        # restore aggregator state
        if aggregator is not None and 'aggregator_state' in ckpt:
            try:
                aggregator.load_state_dict(ckpt['aggregator_state'])
            except Exception:
                logger.warning("Failed to load aggregator_state")
        # restore client states
        if clients and 'client_states' in ckpt:
            for c in clients:
                cid = getattr(c, 'id', str(id(c)))
                state = ckpt['client_states'].get(cid, None)
                if state is not None and hasattr(c, 'load_state'):
                    try:
                        c.load_state(state)
                    except Exception:
                        logger.warning("Failed to load client state for %s", cid)
        # restore RNGs
        try:
            import random as _py_random
            _py_random.setstate(ckpt['random_state'])
            np.random.set_state(ckpt['numpy_random_state'])
            torch.set_rng_state(ckpt['torch_random_state'])
        except Exception as e:
            logger.warning("Failed to restore RNGs: %s", e)
        logger.info("Loaded checkpoint %s (round=%s)", path, ckpt.get('round'))
        return ckpt
