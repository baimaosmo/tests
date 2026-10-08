"""轻量检查：尺寸兼容、反向传播、数据划分和二值标注。"""
import json
import unittest
from unittest.mock import patch
import subprocess
import sys
import torch
from data import ROOT, PennFudan, choose_device
from model import UNet
from train import segmentation_loss


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_odd_image_size_and_backward(self):
        model = UNet(base_channels=4)
        x = torch.randn(2, 3, 33, 41)
        target = torch.zeros(2, 1, 33, 41)
        target[:, :, 8:24, 10:30] = 1
        logits = model(x)
        self.assertEqual(logits.shape, target.shape)
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_split_no_overlap(self):
        splits = json.loads((ROOT / "data" / "splits.json").read_text())
        names = [name for split in splits.values() for name in split]
        self.assertEqual(len(names), 170)
        self.assertEqual(len(set(names)), 170)
        self.assertEqual([len(splits[k]) for k in ["train", "val", "test"]], [120, 25, 25])

    def test_cuda_request_does_not_fall_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA 不可用"):
                choose_device("cuda")

    def test_half_precision_large_mask_loss_is_finite(self):
        logits = torch.full((1, 1, 256, 256), 10.0, dtype=torch.float16, requires_grad=True)
        target = torch.ones_like(logits)
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "当前机器没有 CUDA GPU")
    def test_real_cuda_amp_step(self):
        subprocess.run([sys.executable, str(ROOT / "check_gpu.py")], check=True)

    def test_binary_masks_and_pixels(self):
        dataset = PennFudan(ROOT / "data", "train", size=64)
        for image, mask in dataset:
            self.assertEqual(image.shape, (3, 64, 64))
            self.assertEqual(mask.shape, (1, 64, 64))
            self.assertTrue(((mask == 0) | (mask == 1)).all())
            self.assertTrue((image >= 0).all() and (image <= 1).all())
            self.assertGreater(mask.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
