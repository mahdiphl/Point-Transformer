


"""A hookable copy of the Point Transformer V3 forward pass, for adding your own features.

``PointTransformerV3Custom`` subclasses the package's ``PointTransformerV3`` and re-implements
only ``forward`` / ``_forward_levels`` (copied, same math) so that extra tensors can be injected
at well-defined places. Everything else (stem, encoder/decoder stages, Triton kernels, config,
init) is inherited, so a model built with no hooks is numerically the original one.

Structure (default "ptv3" preset; level s = grid resolution 2**s * grid_size)::

    feat (N, C_in)
      │  embedding            SubMConv3d(k=5) + BN + GELU          -> 32 ch      [hook: on_stem]
      ▼
    enc[0]  blocks x2            (level 0, 32 ch,  2 heads)                      [hook: on_encoder_stage(0)]
    enc[1]  down + blocks x2     (level 1, 64 ch,  4 heads)                      [hook: on_encoder_stage(1)]
    enc[2]  down + blocks x2     (level 2, 128 ch, 8 heads)                      [hook: on_encoder_stage(2)]
    enc[3]  down + blocks x6     (level 3, 256 ch, 16 heads)                     [hook: on_encoder_stage(3)]
    enc[4]  down + blocks x2     (level 4, 512 ch, 32 heads)                     [hook: on_encoder_stage(4)]
      │                                                           (encoder-only models stop here)
    dec[3]  unpool(+skip enc[3]) + blocks x2   256 ch                            [hook: on_decoder_stage(3)]
    dec[2]  unpool(+skip enc[2]) + blocks x2   128 ch                            [hook: on_decoder_stage(2)]
    dec[1]  unpool(+skip enc[1]) + blocks x2    64 ch                            [hook: on_decoder_stage(1)]
    dec[0]  unpool(+skip enc[0]) + blocks x2    64 ch                            [hook: on_decoder_stage(0)]

    Block:  x += CPE(x);  x += attn(LN(x)) [serialized patches];  x += MLP(LN(x))

Extra inputs: put any tensor in the input ``Point`` (e.g. ``point["ect"]``, rows aligned with
``feat``); hooks read it through ``self.ctx`` (the input dict of the current forward). Features
of coarser levels have fewer rows: use ``grid.pool.inverse`` / ``grid.parent`` or pool the extra
tensor yourself (see ``pool_to_level``).

Usage::

    class MyModel(PointTransformerV3Custom):
        def build_extras(self):                       # create new parameters here
            self.ect_proj = nn.Linear(64, self.cfg.enc_channels[0])

        def on_stem(self, x, grid):
            return x + self.ect_proj(self.ctx["ect"].to(x.dtype))

    model = MyModel.from_preset("ptv3", in_channels=3)
    model.describe()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from point_transformer_v3.model import (
    PointTransformerV3,
    _LevelOrders,
    _release,
    _to_point,
)
from point_transformer_v3.modules import check_input
from point_transformer_v3.structure import Level, Point, PointGrid
from point_transformer_v3.structure import auto_depth

__all__ = ["PointTransformerV3Custom", "pool_to_level"]


class PointTransformerV3Custom(PointTransformerV3):
    """PTv3 with identity hooks at the stem, after every encoder stage and every decoder stage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ctx: Mapping[str, Any] = {}
        self.build_extras()
        # Initialise the new modules too. Seeded, so the original weights come out identical.
        self._init_weights()

    # ------------------------------------------------------------------ override these
    def build_extras(self) -> None:
        """Create your extra modules / parameters here (called once, at the end of __init__)."""

    def on_stem(self, x: Tensor, grid: PointGrid) -> Tensor:
        """After the stem: ``x`` (N0, enc_channels[0]) on the level-0 rows of ``grid``."""
        return x

    def on_encoder_stage(self, s: int, x: Tensor, grid: PointGrid) -> Tensor:
        """After encoder stage ``s``: ``x`` (N_s, enc_channels[s]) on the rows of ``grid``
        (level s). The returned tensor feeds stage s+1 and, as skip, decoder stage s."""
        return x

    def on_decoder_stage(self, s: int, x: Tensor, grid: PointGrid) -> Tensor:
        """After decoder stage ``s``: ``x`` (N_s, dec_channels[s]) on the rows of ``grid``
        (level s; ``s = 0`` is the final output)."""
        return x

    # ------------------------------------------------------------------ forward (copied)
    def forward(self, data: Mapping[str, Any], orders: Tensor | None = None) -> Point:
        point = data if isinstance(data, Point) else Point(data)
        device = next(self.parameters()).device
        self.ctx = point
        try:
            with self._autocast(device):
                grid, feat = self._level0(point)
                levels = self._forward_levels(Level(feat, grid), orders)
        finally:
            self.ctx = {}
        out = _to_point(point, levels, has_decoder=bool(self.dec))
        _release(levels)
        return out

    def _forward_levels(self, point: Level, orders: Tensor | None) -> list[Level]:
        cfg = self.cfg
        depth = cfg.serialization_depth
        auto = not isinstance(depth, int)
        grid = check_input(point, cfg.in_channels, None if auto else depth)
        if orders is None:
            orders = self.draw_orders()
        level_orders = _LevelOrders(cfg.orders, orders, grid.device)

        # level chain first (kernel maps are built top-down)
        grids = [grid]
        max_coord = getattr(grid, "_max_coord", None)
        for i, stride in enumerate(cfg.strides):
            d = None
            if auto and cfg.pool_policy == "reencode":
                bits = stride.bit_length() - 1
                max_coord = None if max_coord is None else max_coord >> bits
                nxt = cfg.strides[i + 1].bit_length() if i + 1 < len(cfg.strides) else 1
                d = auto_depth(max_coord or 0, rule=str(cfg.serialization_depth), minimum=nxt)
            grids.append(
                grids[-1].downsample(stride, cfg.pool_policy, order=cfg.orders[0], depth=d)
            )

        # stem
        x = self.embedding(point.feat, grid) if cfg.stem == "conv" else self.embedding(point.feat)
        x = self.on_stem(x, grid)

        # encoder
        levels: list[Level] = []
        for s, stage in enumerate(self.enc):
            grid = grids[s]
            x = stage(x, grid, level_orders(s, grid))
            x = self.on_encoder_stage(s, x, grid)
            levels.append(Level(x, grid, levels[-1] if levels else None))
        if not self.dec:
            return levels

        # decoder (coarse -> fine); x enters at the coarsest level
        for s in reversed(range(len(self.dec))):
            x = self.dec[s](x, levels[s].feat, levels[s + 1].grid, level_orders(s, levels[s].grid))
            x = self.on_decoder_stage(s, x, levels[s].grid)
        levels.append(Level(x, point.grid))
        return levels

    # ------------------------------------------------------------------ inspection
    def describe(self, print_fn=print) -> None:
        """Print the architecture: stages, channels, heads, patch sizes and parameter counts."""
        cfg = self.cfg

        def n(m: nn.Module) -> float:
            return sum(p.numel() for p in m.parameters()) / 1e6

        print_fn(f"PointTransformerV3  in_channels={cfg.in_channels}  orders={cfg.orders}")
        print_fn(f"  stem        {cfg.stem:6s} -> {cfg.enc_channels[0]} ch            {n(self.embedding):7.3f} M")
        for s, st in enumerate(self.enc):
            print_fn(
                f"  enc[{s}] lvl {s}  depth={cfg.enc_depths[s]} ch={cfg.enc_channels[s]:4d} "
                f"heads={cfg.enc_num_heads[s]:2d} patch={cfg.enc_patch_sizes[s]}"
                f"{' (down stride %d)' % cfg.strides[s - 1] if s else ''}  {n(st):7.3f} M"
            )
        for s, st in reversed(list(enumerate(self.dec))):
            print_fn(
                f"  dec[{s}] lvl {s}  depth={cfg.dec_depths[s]} ch={cfg.dec_channels[s]:4d} "
                f"heads={cfg.dec_num_heads[s]:2d} patch={cfg.dec_patch_sizes[s]}  {n(st):7.3f} M"
            )
        extras = [k for k, _ in self.named_children() if k not in ("embedding", "enc", "dec")]
        for k in extras:
            print_fn(f"  {k:10s}  {n(getattr(self, k)):7.3f} M")
        print_fn(f"  total {n(self):.3f} M parameters")


def pool_to_level(values: Tensor, grid: PointGrid, reduce: str = "mean") -> Tensor:
    """Bring a per-row tensor of level 0 down to ``grid``'s level (walks ``grid.parent`` chain).

    ``values``: (N_0, D) rows of the level-0 grid; returns (N_grid, D). Each coarse row reduces
    (``"mean"`` or ``"max"``) the rows of its children, using the pooling index the model already
    built (``grid.pool.inverse``: parent row -> coarse row).
    """
    chain = []
    g = grid
    while g.parent is not None:
        chain.append(g)
        g = g.parent
    for g in reversed(chain):
        inv = g.pool.inverse.long()  # (N_parent_rows,) -> coarse row
        m = int(inv.max().item()) + 1 if inv.numel() else 0
        out = values.new_zeros(m, values.shape[1])
        if reduce == "max":
            out = out.scatter_reduce(0, inv[:, None].expand_as(values), values, "amax", include_self=False)
        else:
            out.index_add_(0, inv, values)
            out = out / torch.bincount(inv, minlength=m).clamp(min=1)[:, None].to(values.dtype)
        values = out
    return values


if __name__ == "__main__":
    PointTransformerV3Custom.from_preset("ptv3", in_channels=3).describe()
