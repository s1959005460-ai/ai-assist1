# tests/test_aggregator.py
import torch
from aggregator import FedAvgAggregator, SCAFFOLDAggregator

def make_state(shape=(2,2), val=0.0):
    return {'w': torch.ones(*shape) * val}

def test_fedavg():
    agg = FedAvgAggregator(device='cpu')
    g = make_state(val=0.0)
    d1 = {'w': torch.ones(2,2) * 1.0}
    d2 = {'w': torch.ones(2,2) * 3.0}
    res = agg.aggregate(g, [d1, d2], [0.5, 0.5])
    assert torch.allclose(res['w'], torch.ones(2,2) * 1.5)

def test_scaffold_basic():
    agg = SCAFFOLDAggregator(device='cpu')
    # create fake client results
    g = make_state(val=0.0)
    # client 1 delta: +1, c_delta small
    c1 = {'delta': {'w': torch.ones(2,2) * 1.0}, 'c_delta': {'w': torch.zeros(2,2)}}
    c2 = {'delta': {'w': torch.ones(2,2) * 3.0}, 'c_delta': {'w': torch.zeros(2,2)}}
    res = agg.aggregate(g, [c1, c2], [0.5, 0.5])
    assert 'w' in res
