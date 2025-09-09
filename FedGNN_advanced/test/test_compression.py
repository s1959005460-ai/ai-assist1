# tests/test_compression.py
import torch
from compression import topk_sparsify, serialize_sparse, deserialize_sparse, topk_aggregate

def test_topk_roundtrip():
    delta = {'w': torch.randn(20)}
    sparse = topk_sparsify(delta, k_ratio=0.2)
    b = serialize_sparse(sparse)
    sparse2 = deserialize_sparse(b)
    # aggregate single
    agg = topk_aggregate([sparse2], device='cpu')
    assert 'w' in agg

