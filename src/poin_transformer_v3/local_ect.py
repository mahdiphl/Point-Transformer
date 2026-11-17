import math
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph

from layers.ect import EctLayer
from layers.config import EctConfig


def fibonacci_directions(m):
    """Deterministic, near-uniform unit directions on S^2, shape (3, m)."""
    i = torch.arange(m, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / m)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.cos(theta) * torch.sin(phi),
                        torch.sin(theta) * torch.sin(phi),
                        torch.cos(phi)])


def compute_local_ect(dataset,
                      radius=1,
                      ECT_TYPE='points',
                      NUM_THETAS=64,
                      NUM_STEPS=None,
                      DEVICE='cpu',
                      subsample_size=None,
                      v=None,
                      center=True,
                      rescale=True,
                      chunk=256):
    """
    Local ECT of the `radius`-hop patch of every node.

    v        : (3, NUM_THETAS) fixed directions (default: Fibonacci sphere), so that
               scores are deterministic and comparable across scales / shapes.
    center   : subtract the patch centre node, removing absolute position.
    rescale  : divide by the largest point norm in the patch so it fills [-1, 1]
               (assumes EctConfig.R >= 1; check layers/config.py).
    chunk    : patches per EctLayer call (bounds memory at radius >= 2).
    Returns (n_nodes, NUM_STEPS * NUM_THETAS); axis layout after reshape is
    (threshold, direction), as produced by layers/ect.py.
    """
    NUM_STEPS = NUM_STEPS or NUM_THETAS
    data = dataset
    X, E, N = data.x, data.edge_index, data.x.size(0)
    if subsample_size is not None:
        g = torch.Generator().manual_seed(42)
        sub_nodes = torch.randperm(N, generator=g)[:subsample_size].tolist()
    else:
        sub_nodes = list(range(N))

    v = (fibonacci_directions(NUM_THETAS) if v is None else v).to(DEVICE)

    patches = []
    for i in sub_nodes:
        subset, ei, _, _ = k_hop_subgraph(int(i), radius, E, relabel_nodes=True, num_nodes=N)
        x = X[subset]
        if center:
            x = x - X[i]
            if rescale:
                x = x / (x.norm(dim=1).max() + 1e-8)
        patches.append(Data(x=x, edge_index=ei))

    cfg = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_STEPS, normalized=False,
                    device=DEVICE, num_features=X.size(1), ect_type=ECT_TYPE)
    layer = EctLayer(config=cfg, v=v)

    outs = []
    for a in range(0, len(patches), chunk):
        batch = Batch.from_data_list(patches[a:a + chunk]).to(DEVICE)
        ect = layer(batch)                                  # (n, steps, thetas)
        outs.append(ect.reshape(ect.shape[0], -1).cpu())
    return torch.cat(outs)
