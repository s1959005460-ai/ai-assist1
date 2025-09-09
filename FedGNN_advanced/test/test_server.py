# tests/test_server.py
import torch
from models import get_model
from server import AsyncFedServer

def test_aggregate_deltas_simple():
    model = get_model('gcn', in_ch=4, hidden=8, out=3)
    server = AsyncFedServer(model, [], device='cpu')
    gs = model.state_dict()
    d1 = {k: torch.ones_like(v) * 0.1 for k,v in gs.items()}
    d2 = {k: torch.ones_like(v) * 0.2 for k,v in gs.items()}
    agg = server.aggregate_deltas([d1, d2], weights=[0.5, 0.5])
    for k in agg.keys():
        assert torch.allclose(agg[k], torch.ones_like(agg[k]) * 0.15)
