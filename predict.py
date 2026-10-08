"""生成原图、真实标注、预测标注和叠加图；也支持预测自己的单张图片。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

from data import ROOT, PennFudan, choose_device, image_tensor
from model import UNet


def make_panel(image, prediction, truth=None):
    array = np.array(image)
    red = np.zeros_like(array)
    red[:, :, 0] = 255
    overlay = array.copy()
    overlay[prediction] = (0.55 * array[prediction] + 0.45 * red[prediction]).astype(np.uint8)
    panels = [("Image", image)]
    if truth is not None:
        panels.append(("Ground truth", Image.fromarray(truth.astype(np.uint8) * 255).convert("RGB")))
    panels.extend([("Prediction", Image.fromarray(prediction.astype(np.uint8) * 255).convert("RGB")),
                   ("Overlay", Image.fromarray(overlay))])
    width, height = image.size
    canvas = Image.new("RGB", (len(panels) * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        draw.text((index * width + 5, 5), label, fill="black")
        canvas.paste(panel, (index * width, 24))
    return canvas


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "demo" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "demo" / "predictions")
    parser.add_argument("--image", type=Path, help="可选：自己的图片路径；省略则预测固定测试集")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto、cuda、cuda:N、cpu 或 mps")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须为正")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = UNet(config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = config["image_size"]
    if args.image:
        with Image.open(args.image) as source:
            original = source.convert("RGB")
        x = image_tensor(original, size)
        probabilities = model(x.unsqueeze(0).to(device)).sigmoid()
        probabilities = torch.nn.functional.interpolate(probabilities, size=(original.height, original.width),
                                                        mode="bilinear", align_corners=False)
        mask = probabilities[0, 0].cpu().numpy() >= 0.5
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output_dir / f"{args.image.stem}_mask.png")
        make_panel(original, mask).save(args.output_dir / f"{args.image.stem}_comparison.png")
    else:
        dataset = PennFudan(args.data_dir, "test", size, limit=args.count)
        rows = []
        for name, (x, y) in zip(dataset.names, dataset):
            prediction = model(x.unsqueeze(0).to(device)).sigmoid()[0, 0].cpu().numpy() >= 0.5
            original = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            panel = make_panel(original, prediction, y[0].numpy().astype(bool))
            panel.save(args.output_dir / f"{name}_comparison.png")
            Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output_dir / f"{name}_mask.png")
            rows.append(panel)
        grid = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
        offset = 0
        for row in rows:
            grid.paste(row, (0, offset))
            offset += row.height
        grid.save(args.output_dir / "comparison_grid.png")
    print(f"Predictions saved: {args.output_dir}")


if __name__ == "__main__":
    main()
