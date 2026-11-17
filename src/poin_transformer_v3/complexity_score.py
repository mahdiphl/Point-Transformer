
import torch
from functools import lru_cache
from torch_geometric.transforms import Compose, KNNGraph, NormalizeScale

from local_ect import compute_local_ect, fibonacci_directions
from visualize_ect_complexity import compute_complexity


class PosToX:
    def __call__(self, data):
        data.x = data.pos
        del data.pos
        return data


ect_transform = Compose([
    KNNGraph(k=5),
    NormalizeScale(),
    PosToX(),
])


@lru_cache(maxsize=None)
def _directions(num_thetas):
    return fibonacci_directions(num_thetas)


def compute_complexity_score(data, method='entropy', num_thetas=64,
                             num_steps=None, scales=(1,)):
    """
    One complexity score per point and scale, each min-max normalised over the shape.

    scales : hop radii of the patches, e.g. (1,) or (1, 2).
    Returns (N,) for a single scale (backward compatible) and (N, L) otherwise.
    All scales share the same fixed directions.
    """
    data = ect_transform(data)
    v = _directions(num_thetas)
    cols = []
    for r in scales:
        ect = compute_local_ect(data, radius=int(r), ECT_TYPE='points',
                                NUM_THETAS=num_thetas, NUM_STEPS=num_steps, v=v)
        s = compute_complexity(ect, num_thetas=num_thetas, method=method,
                               num_steps=num_steps)               # numpy (N,)
        cols.append(torch.from_numpy(s).float())
    return cols[0] if len(cols) == 1 else torch.stack(cols, dim=1)
