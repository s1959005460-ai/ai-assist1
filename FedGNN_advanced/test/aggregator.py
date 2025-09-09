# aggregator.py
from abc import ABC, abstractmethod
import copy
import math
import torch
import logging

logger = logging.getLogger(__name__)

class Aggregator(ABC):
    """Abstract aggregator interface."""

    @abstractmethod
    def aggregate(self, server_model_state: dict, client_deltas: list, client_weights: list):
        """
        :param server_model_state: dict of server model tensors (on device)
        :param client_deltas: list of dicts; each dict maps key->tensor (CPU tensors)
        :param client_weights: list of floats
        :returns: (new_server_state_dict, info_dict)
        """
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass

class FedAvgAggregator(Aggregator):
    def __init__(self, device='cpu'):
        self.device = device

    def aggregate(self, server_model_state, client_deltas, client_weights):
        if not client_deltas:
            raise ValueError("No client deltas")
        if client_weights is None:
            client_weights = [1.0/len(client_deltas)] * len(client_deltas)
        # weighted sum of deltas (client deltas are CPU tensors)
        agg_delta = {}
        keys = list(server_model_state.keys())
        for k in keys:
            agg_delta[k] = torch.zeros_like(server_model_state[k]).to(self.device)
        for delta, w in zip(client_deltas, client_weights):
            for k, v in delta.items():
                agg_delta[k] += v.to(self.device) * float(w)
        # apply: server = server + agg_delta
        new_state = {}
        for k, v in server_model_state.items():
            new_state[k] = (v.to(self.device) + agg_delta[k]).to(self.device)
        info = {'agg_norm': torch.sqrt(sum((x.float().pow(2).sum() for x in agg_delta.values()))).item()}
        return new_state, info

class FedAdamAggregator(Aggregator):
    """
    FedAdam: treat aggregated delta as (negative) gradient and run Adam update on server.
    We keep m, v, t as state.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, device='cpu'):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device
        self.m = None
        self.v = None
        self.t = 0

    def _ensure_state(self, server_model_state):
        if self.m is None or self.v is None:
            self.m = {k: torch.zeros_like(v).to(self.device) for k,v in server_model_state.items()}
            self.v = {k: torch.zeros_like(v).to(self.device) for k,v in server_model_state.items()}
            self.t = 0

    def aggregate(self, server_model_state, client_deltas, client_weights):
        if not client_deltas:
            raise ValueError("No client deltas")
        if client_weights is None:
            client_weights = [1.0/len(client_deltas)] * len(client_deltas)

        self._ensure_state(server_model_state)

        # compute weighted average delta (on device)
        agg_delta = {k: torch.zeros_like(v).to(self.device) for k,v in server_model_state.items()}
        for delta, w in zip(client_deltas, client_weights):
            for k, v in delta.items():
                agg_delta[k] += v.to(self.device) * float(w)

        # Treat g = -agg_delta (so that if clients sent local_state - global_state,
        # then negative delta is the descent direction)
        g_dict = {k: (-agg_delta[k]).to(self.device) for k in agg_delta.keys()}

        # Adam update
        self.t += 1
        new_state = {}
        for k, param in server_model_state.items():
            g = g_dict[k]
            m_k = self.m[k]
            v_k = self.v[k]
            m_k.mul_(self.beta1).add_(g, alpha=(1.0 - self.beta1))
            v_k.mul_(self.beta2).addcmul_(g, g, value=(1.0 - self.beta2))
            m_hat = m_k / (1.0 - (self.beta1 ** self.t))
            v_hat = v_k / (1.0 - (self.beta2 ** self.t))
            step = self.lr * m_hat / (v_hat.sqrt() + self.eps)
            new_param = (param.to(self.device) + (-step)).to(self.device)  # minus because g already negative direction
            new_state[k] = new_param
            # store back
            self.m[k] = m_k
            self.v[k] = v_k
        info = {'t': self.t}
        return new_state, info

    def state_dict(self):
        return {
            'm': {k: v.cpu() for k,v in self.m.items()} if self.m else None,
            'v': {k: v.cpu() for k,v in self.v.items()} if self.v else None,
            't': int(self.t)
        }

    def load_state_dict(self, sd):
        if sd is None:
            return
        m = sd.get('m')
        v = sd.get('v')
        t = sd.get('t', 0)
        if m is not None and v is not None:
            # convert to device
            self.m = {k: vv.to(self.device) for k, vv in m.items()}
            self.v = {k: vv.to(self.device) for k, vv in v.items()}
            self.t = int(t)
