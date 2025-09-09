# learner.py (SCAFFOLD-aware version)
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any

class Learner:
    def __init__(self, model: nn.Module, data, device='cpu', lr=0.01, local_epochs=1, client_id=None, use_scaffold=False):
        self.device = device
        self.model = model.to(device)
        self.data = data
        self.lr = lr
        self.local_epochs = local_epochs
        self.id = client_id or str(id(self))
        self.use_scaffold = use_scaffold
        if use_scaffold:
            # c_i is same structure as state_dict
            self.c_i = {k: torch.zeros_like(v).to(device) for k, v in self.model.state_dict().items()}
        else:
            self.c_i = None
        self.scaler = GradScaler() if torch.cuda.is_available() else None

    def train_local(self, global_state: Dict[str, torch.Tensor], c_global: Dict[str, torch.Tensor]=None, prox_mu: float=0.0):
        """
        Implements SCAFFOLD local update and exact control variate update.
        Algorithm outline:
            - Start from w_0 = global_state
            - For t in 1..tau: perform SGD step on local loss, but replace gradient g with g - c_i + c_global
            - After tau steps, produce w_tau.
            - c_i_new = c_i - c_global + (1/(tau * lr)) * (w_tau - w_0)
            - c_delta = c_i_new - c_i  (to be sent back to server)
        """
        # load global
        self.model.load_state_dict({k: v.to(self.device) for k,v in global_state.items()})
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = nn.NLLLoss()
        tau = max(1, int(self.local_epochs))
        # caching w0
        w0 = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        for epoch in range(tau):
            self.model.train()
            optimizer.zero_grad()
            x = self.data['x'].to(self.device)
            edge_index = self.data.get('edge_index', None)
            out = self.model(x, edge_index) if edge_index is not None else self.model(x)
            mask = self.data.get('train_mask', torch.ones(out.size(0), dtype=torch.bool, device=self.device)).to(self.device)
            if mask.sum() == 0:
                continue
            if self.scaler:
                with autocast():
                    loss = loss_fn(out[mask], self.data['y'].to(self.device)[mask])
                self.scaler.scale(loss).backward()
                # gradient correction: subtract c_i, add c_global
                if self.use_scaffold and c_global is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        key = name  # assume mapping identical
                        if key in self.c_i and key in c_global:
                            param.grad.data.add_( - (self.c_i[key].to(self.device) - c_global[key].to(self.device)) )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss = loss_fn(out[mask], self.data['y'].to(self.device)[mask])
                loss.backward()
                if self.use_scaffold and c_global is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        key = name
                        if key in self.c_i and key in c_global:
                            param.grad.data.add_( - (self.c_i[key].to(self.device) - c_global[key].to(self.device)) )
                optimizer.step()
        # compute delta and c_delta
        wtau = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        delta = {k: (wtau[k] - w0[k]) for k in w0.keys()}
        result = {'delta': delta}
        if self.use_scaffold and c_global is not None:
            # c_i_new = c_i - c_global + (1 / (tau * lr)) * (w_tau - w0)
            c_delta = {}
            factor = 1.0 / (tau * self.lr)
            for k in self.c_i.keys():
                term = (wtau[k] - w0[k]) * factor
                c_new = (self.c_i[k].to('cpu') - c_global[k].to('cpu') + term)
                c_delta[k] = (c_new - self.c_i[k].to('cpu')).clone()
                # update local c_i (keep on device)
                self.c_i[k] = c_new.to(self.device)
            result['c_delta'] = c_delta
        return result
