"""将实例标注转换为二分类标注，图像双线性缩放，标注最近邻缩放。"""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def image_tensor(image, size):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class PennFudan(Dataset):
    def __init__(self, data_dir, split="train", size=128, augment=False, limit=None):
        data_dir = Path(data_dir)
        split_path = data_dir / "splits.json"
        if not split_path.exists():
            raise FileNotFoundError("请先运行 python download_data.py 下载数据并创建划分")
        self.names = json.loads(split_path.read_text())[split]
        if limit is not None:
            self.names = self.names[:limit]
        self.augment = augment
        self.samples = []
        # 数据很小，提前缓存缩放后的张量，训练时不重复解码原图。
        for name in self.names:
            with Image.open(data_dir / "PennFudanPed" / "PNGImages" / f"{name}.png") as image:
                x = image_tensor(image, size)
            with Image.open(data_dir / "PennFudanPed" / "PedMasks" / f"{name}_mask.png") as mask:
                mask = mask.resize((size, size), Image.Resampling.NEAREST)
                y = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        if self.augment and random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def choose_device(requested="auto"):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用：请检查 NVIDIA 驱动、GPU 分配和 CUDA 版 PyTorch。运行 python check_gpu.py 排查；CPU 调试请显式指定 --device cpu。")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"GPU 索引 {device.index} 不存在，可见 GPU 数量为 {torch.cuda.device_count()}")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Apple MPS 不可用")
        elif device.type != "cpu":
            raise ValueError("device 仅支持 cuda、cuda:N、cpu、mps 或 auto")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
