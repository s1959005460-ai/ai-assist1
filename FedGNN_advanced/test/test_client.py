# tests/test_client.py
import torch
from models import get_model
from learner import Learner

def test_learner_returns_delta():
    model = get_model('gcn', in_ch=4, hidden=8, out=3)
    data = {'x': torch.randn(10,4), 'adj': torch.eye(10), 'y': torch.randint(0,3,(10,)), 'train_mask': torch.ones(10,dtype=torch.bool)}
    learner = Learner(model, data, lr=0.01, local_epochs=1)
    gs = {k: v.clone() for k,v in model.state_dict().items()}
    res = learner.train_local(gs)
    assert 'delta' in res and 'train_loss' in res
    assert set(res['delta'].keys()) == set(gs.keys())
