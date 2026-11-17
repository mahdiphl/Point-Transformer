
"""Train Point Transformer V3 (random init) on the datasets of ``data_pyg.py``.

Tasks (picked from ``--dataset``):
  * classification  : modelnet40, shapenet, donut_genus, donut_components
  * part segmentation: shapenetpart (50 part labels)

Pipeline per batch
  dataset -> (B, N, 3) clouds -> ptv3.prepare (grid sampling, one point per grid cell)
          -> ptv3.collate (pack the B clouds, ``offset``) -> backbone -> task head -> loss

The ECT / complexity feature returned by the datasets is ignored for now (see ``unpack``).

Examples
    python main.py --dataset modelnet40 --epochs 100 --batch-size 32
    python main.py --dataset shapenetpart --epochs 100 --batch-size 16
    python main.py --dataset donut_genus --donut-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

#import point_transformer_v3 as ptv3
from data_pyg import NUM_CLASSES, get_dataset  # rename "data_pyg (6).py" -> data_pyg.py
from ptv3_custom import PointTransformerV3Custom  # hookable copy of the model (subclass it)

SEG_DATASETS = {"shapenetpart"}


# --------------------------------------------------------------------------- model
def build_backbone(task: str, in_channels: int, patch_size: int, drop_path: float):
    """Randomly initialised PTv3 from the ``"ptv3"`` preset, with a few overrides.

    * classification: encoder only (``dec_depths=()``); the output is the coarsest level
      (512 channels) and is pooled per sample.
    * segmentation: encoder + decoder; the output is (N, 64) on the level-0 rows.

    Change the architecture here (or pass a ``Config``) when you start modifying the model.
    """
    overrides = dict(
        in_channels=in_channels,
        enc_patch_sizes=(patch_size,) * 5,
        drop_path=drop_path,
    )
    if task == "cls":
        overrides.update(dec_depths=(), dec_channels=(), dec_num_heads=(), dec_patch_sizes=())
    else:
        overrides.update(dec_patch_sizes=(patch_size,) * 4)
    return PointTransformerV3Custom.from_preset("ptv3", **overrides)


class Classifier(nn.Module):
    """PTv3 encoder -> [mean ‖ max] pooling per sample -> MLP."""

    def __init__(self, backbone, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        c = backbone.out_channels
        self.head = nn.Sequential(
            nn.Linear(2 * c, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, point):
        out = self.backbone(point)
        feat, batch = out["feat"].float(), out["batch"].long()
        b = int(point["offset"].numel())
        pooled = torch.cat(
            [global_mean_pool(feat, batch, size=b), global_max_pool(feat, batch, size=b)], dim=1
        )
        return self.head(pooled)  # (B, num_classes)


class Segmenter(nn.Module):
    """PTv3 U-Net -> linear head on every kept point."""

    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, point):
        out = self.backbone(point)
        return self.head(out["feat"].float())  # (sum kept points, num_classes)


# --------------------------------------------------------------------------- data
def unpack(batch, task: str):
    """Dataset item -> (clouds (B,N,3), labels). The ECT features (2nd element) are dropped."""
    if task == "seg":
        clouds, _ect, _category, part_labels = batch  # part_labels: (B, N)
        return clouds, part_labels
    clouds, _ect, label = batch
    return clouds, label.view(-1)  # label: (B, 1) -> (B,)


def make_point(clouds: torch.Tensor, device, grid_size: float):
    """Grid-sample every cloud (PTv3 needs one point per cell) and pack them into one batch.

    Returns the packed ``Point`` and, per sample, ``inverse`` (N,): the kept row of each
    original point, used to bring per-row predictions back to all N input points.
    """
    clouds = clouds.to(device, non_blocking=True)
    items = [
        ptv3.prepare(c, feat=("coord",), center=False, grid_size=grid_size, pick="random",
                     device=device)
        for c in clouds
    ]
    point = ptv3.collate([p for p, _ in items])
    return point, [inv for _, inv in items]


def seg_logits_per_point(logits, point, inverses):
    """(sum kept rows, C) -> (B*N, C): every original point takes the logits of its cell."""
    ends = point["offset"].tolist()
    parts, start = [], 0
    for end, inv in zip(ends, inverses):
        parts.append(logits[start:end][inv])
        start = end
    return torch.cat(parts)


# --------------------------------------------------------------------------- train / eval
def run_epoch(model, loader, task, args, device, num_classes, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, n_seen, n_correct = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long) if task == "seg" else None

    for batch in loader:
        clouds, labels = unpack(batch, task)
        labels = labels.to(device, non_blocking=True)
        point, inverses = make_point(clouds, device, args.grid_size)

        with torch.set_grad_enabled(train):
            out = model(point)
            if task == "seg":
                out = seg_logits_per_point(out, point, inverses)
                labels = labels.reshape(-1)
            loss = F.cross_entropy(out, labels, label_smoothing=args.label_smoothing if train else 0.0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

        pred = out.argmax(1)
        loss_sum += loss.item() * labels.numel()
        n_seen += labels.numel()
        n_correct += (pred == labels).sum().item()
        if confusion is not None:
            idx = labels * num_classes + pred
            confusion += torch.bincount(idx, minlength=num_classes**2).view(num_classes, -1).cpu()

    metrics = {"loss": loss_sum / max(n_seen, 1), "acc": n_correct / max(n_seen, 1)}
    if confusion is not None:
        tp = confusion.diag().float()
        union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
        present = union > 0  # ignore parts that never occur
        metrics["miou"] = (tp[present] / union[present]).mean().item()
    return metrics


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="modelnet40", choices=sorted(NUM_CLASSES))
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--donut-fraction", type=float, default=1.0)
    p.add_argument("--complexity-method", default="entropy")
    p.add_argument("--num-thetas", type=int, default=64)
    # model
    p.add_argument("--grid-size", type=float, default=0.02,
                   help="grid cell (clouds are unit-normalised, so ~[-1, 1]); smaller = more points kept")
    p.add_argument("--patch-size", type=int, default=1024)
    p.add_argument("--drop-path", type=float, default=0.1)
    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # misc
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ptv3")
    p.add_argument("--resume", default=None, help="path to a last.pt checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: PTv3's fast kernels are Triton/CUDA; expect it to be slow or fail on CPU.")

    task = "seg" if args.dataset in SEG_DATASETS else "cls"
    num_classes = NUM_CLASSES[args.dataset]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))

    # ---- data
    ds_kw = dict(num_points=args.num_points, complexity_method=args.complexity_method,
                 num_thetas=args.num_thetas, fraction=args.donut_fraction)
    train_set = get_dataset(args.dataset, "train", **ds_kw)
    test_set = get_dataset(args.dataset, "test", **ds_kw)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)
    print(f"{args.dataset}: task={task}, classes={num_classes}, "
          f"train={len(train_set)}, test={len(test_set)}")

    # ---- model (input features = the grid-sampled xyz -> 3 channels)
    backbone = build_backbone(task, in_channels=3, patch_size=args.patch_size,
                              drop_path=args.drop_path)
    model = (Classifier if task == "cls" else Segmenter)(backbone, num_classes).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # ---- optimiser: no weight decay on biases / norm parameters
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
    )
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, args.epochs * steps_per_epoch,
                                   args.warmup_epochs * steps_per_epoch)

    start_epoch, best = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    key = "miou" if task == "seg" else "acc"  # metric that decides the best checkpoint
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, task, args, device, num_classes, optimizer, scheduler)
        te = run_epoch(model, test_loader, task, args, device, num_classes)
        msg = (f"epoch {epoch + 1:3d}/{args.epochs}  lr {scheduler.get_last_lr()[0]:.2e}  "
               f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  |  "
               f"test loss {te['loss']:.4f} acc {te['acc']:.4f}")
        if task == "seg":
            msg += f" mIoU {te['miou']:.4f}"
        print(msg + f"  ({time.time() - t0:.0f}s)")

        if te[key] > best:
            best = te[key]
            torch.save(model.state_dict(), out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best},
                   out_dir / "last.pt")

    print(f"best test {key}: {best:.4f}")


if __name__ == "__main__":
    main()
