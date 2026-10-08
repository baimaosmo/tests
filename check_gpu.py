"""服务器 GPU 自检：执行本项目 U-Net 的前向、损失和反向传播。"""
import argparse
import os
import torch
from data import choose_device
from model import UNet
from train import segmentation_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}", flush=True)
    device = choose_device(args.device)
    if device.type != "cuda":
        parser.error("此脚本专用于 CUDA GPU 自检")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}; memory: {properties.total_memory / 1024**3:.1f} GiB", flush=True)
    model = UNet(base_channels=8).to(device)
    images = torch.rand(2, 3, 64, 64, device=device)
    masks = (torch.rand(2, 1, 64, 64, device=device) > 0.5).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=128)
    with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
        logits = model(images)
    loss = segmentation_loss(logits, masks)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("GPU 自检失败：损失或梯度存在非有限数值")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    print(f"GPU forward/backward OK; AMP={args.amp}; loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
