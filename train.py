"""训练 U-Net，用验证集选择模型，最后只对最佳模型评估一次测试集。"""
import argparse
import csv
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import ROOT, PennFudan, choose_device
from model import UNet


def segmentation_loss(logits, target):
    # AMP 下也使用 FP32 计算损失，避免大掩码的 FP16 求和溢出。
    logits, target = logits.float(), target.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probability * target).sum(dims) + 1) / (
        probability.sum(dims) + target.sum(dims) + 1
    )
    return bce + (1 - dice).mean()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    count = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.sigmoid() >= 0.5
        targets_bool = targets.bool()
        dims = (1, 2, 3)
        intersection = (predictions & targets_bool).sum(dims).float()
        union = (predictions | targets_bool).sum(dims).float()
        total_area = predictions.sum(dims) + targets_bool.sum(dims)
        totals["loss"] += segmentation_loss(logits, targets).item() * len(images)
        totals["dice"] += ((2 * intersection + 1e-6) / (total_area + 1e-6)).sum().item()
        totals["iou"] += ((intersection + 1e-6) / (union + 1e-6)).sum().item()
        count += len(images)
    return {key: value / count for key, value in totals.items()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "demo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="默认 cuda；可用 cuda:1、cpu、mps、auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA 默认使用 FP16 混合精度，--no-amp 关闭")
    parser.add_argument("--num-workers", type=int, default=0, help="缓存的小数据集默认无需额外进程，可按需增加")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None, help="仅用于快速冒烟验证")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.base_channels, args.threads) < 1 or args.image_size < 8:
        parser.error("轮数、批大小、通道和线程数必须为正，图像尺寸必须 >= 8")
    if args.lr <= 0 or (args.limit_train is not None and args.limit_train < 1):
        parser.error("学习率和训练样本上限必须为正")
    if args.num_workers < 0:
        parser.error("num-workers 必须 >= 0")
    try:
        device = choose_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if (args.output_dir / "best.pt").exists():
        parser.error("输出目录已有模型；请使用新的 --output-dir，避免覆盖已有结果")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and args.amp
    if is_cuda:
        torch.cuda.manual_seed_all(args.seed)
    train = PennFudan(args.data_dir, "train", args.image_size, True, args.limit_train)
    val = PennFudan(args.data_dir, "val", args.image_size)
    test = PennFudan(args.data_dir, "test", args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=is_cuda, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val, **loader_options)
    test_loader = DataLoader(test, **loader_options)
    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device_used=str(device), python=platform.python_version(), torch=str(torch.__version__),
                  cuda_version=torch.version.cuda, amp_enabled=use_amp,
                  gpu_name=torch.cuda.get_device_name(device) if is_cuda else None,
                  parameters=sum(p.numel() for p in model.parameters()),
                  split_counts={"train": len(train), "val": len(val), "test": len(test)})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "splits.json").write_text((args.data_dir / "splits.json").read_text())
    print(f"Device={device}; parameters={config['parameters']:,}; train/val/test={len(train)}/{len(val)}/{len(test)}", flush=True)
    print(f"GPU={config['gpu_name']}; CUDA={config['cuda_version']}; AMP={use_amp}", flush=True)
    if is_cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    best_dice = -1.0
    with (args.output_dir / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(images)
                loss = segmentation_loss(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(images)
            metrics = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": total_loss / len(train),
                   **{f"val_{key}": value for key, value in metrics.items()}}
            writer.writerow(row)
            file.flush()
            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                            "val_metrics": metrics}, args.output_dir / "best.pt")
            print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={row['train_loss']:.4f} "
                  f"val_dice={metrics['dice']:.4f} val_iou={metrics['iou']:.4f}", flush=True)
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    result = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["val_metrics"],
              "test": evaluate(model, test_loader, device),
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {args.output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
