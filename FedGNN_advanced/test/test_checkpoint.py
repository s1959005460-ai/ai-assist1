# tests/test_checkpoint.py
import os
import torch
from checkpoint import CheckpointManager
from models import get_model

def test_checkpoint_save_load(tmp_path):
    model = get_model('gcn', in_ch=4, hidden=8, out=3)
    class Dummy:
        pass
    server = Dummy()
    server.global_model = model
    ck = CheckpointManager(str(tmp_path))
    p = ck.save_checkpoint(0, server, aggregator=None, clients=[])
    assert os.path.exists(p)
    ck.load_checkpoint(p, server=server, aggregator=None, clients=[])
