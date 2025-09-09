# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        if bias:
            self.b = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('b', None)
    def forward(self, x, adj):
        xW = x @ self.W
        out = adj @ xW
        if self.b is not None:
            out = out + self.b
        return out

class PureGCN(nn.Module):
    """
    Two-layer GCN. We expose both embedding (hidden) and final log-prob output.
    forward returns logits; get_embedding returns hidden features.
    """
    def __init__(self, in_feats, hidden, out_feats, dropout=0.5):
        super().__init__()
        self.conv1 = PureGCNLayer(in_feats, hidden)
        self.conv2 = PureGCNLayer(hidden, out_feats)
        self.dropout = dropout

    def embed(self, x, adj):
        h = self.conv1(x, adj)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, adj):
        h = self.embed(x, adj)
        out = self.conv2(h, adj)
        return F.log_softmax(out, dim=1)

def get_model(name: str, in_ch:int, hidden:int, out:int, **kw):
    name = name.lower()
    if name in ('gcn','puregcn'):
        return PureGCN(in_ch, hidden, out, dropout=kw.get('dropout', 0.5))
    raise ValueError(f"Unknown model {name}")
