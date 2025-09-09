\
import os
import json
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

def load_planetoid(name:str="Cora"):
    ds = Planetoid(root=os.path.join("data", name), name=name)
    return ds[0], ds

def load_custom_from_csv(node_csv:str, edge_csv:str):
    """
    节点CSV: 必须包含 id 列，其他列为特征；可选 label 列
    边CSV: 两列 src, dst （0-based）
    """
    import pandas as pd
    nodes = pd.read_csv(node_csv)
    edges = pd.read_csv(edge_csv)

    if "id" not in nodes.columns:
        raise ValueError("节点CSV必须包含 'id' 列")
    if not {"src","dst"}.issubset(edges.columns):
        raise ValueError("边CSV必须包含 'src','dst' 两列")

    nodes = nodes.sort_values("id").reset_index(drop=True)
    feat_cols = [c for c in nodes.columns if c not in ("id","label")]
    x = torch.tensor(nodes[feat_cols].values, dtype=torch.float)
    y = None
    if "label" in nodes.columns:
        y = torch.tensor(nodes["label"].values, dtype=torch.long)

    edge_index = torch.tensor(edges[["src","dst"]].values.T, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, None

def load_custom_from_json(json_path:str):
    """
    JSON 格式:
    {
      "nodes":[ {"id":0,"feat":[...], "label":0}, ... ],
      "edges":[ [src,dst], ... ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    nodes = obj["nodes"]
    edges = obj["edges"]
    x = torch.tensor([n["feat"] for n in nodes], dtype=torch.float)
    y = None
    if all("label" in n for n in nodes):
        y = torch.tensor([n["label"] for n in nodes], dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).T
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, None

def split_non_iid_by_label(data:Data, num_clients:int=3):
    """
    Non-IID: 按标签划分，每个客户端获得若干标签子集（若无标签则随机）
    返回: List[Tensor[node_idx]]
    """
    n = data.num_nodes
    if data.y is None:
        perm = torch.randperm(n)
        per = n // num_clients
        return [perm[i*per:(i+1)*per] if i < num_clients-1 else perm[i*per:] for i in range(num_clients)]

    labels = data.y
    unique = labels.unique()
    chunks = torch.chunk(unique, num_clients)
    idxs = []
    for ch in chunks:
        mask = torch.isin(labels, ch)
        idxs.append(mask.nonzero(as_tuple=True)[0])
    # 若 unique < num_clients，做补齐
    while len(idxs) < num_clients:
        idxs.append(torch.tensor([], dtype=torch.long))
    return idxs[:num_clients]
