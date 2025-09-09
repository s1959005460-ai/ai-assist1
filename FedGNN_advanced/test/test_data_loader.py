# tests/test_data_loader.py
from data_loader import synthetic_graph, split_non_iid_by_label
def test_split_non_iid_basic():
    data = synthetic_graph(num_nodes=12, in_feats=4, num_classes=3)
    parts = split_non_iid_by_label(data, num_clients=3)
    assert len(parts) == 3
