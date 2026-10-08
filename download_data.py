"""从发布者网站下载完整 Penn-Fudan 数据，生成固定且互斥的数据划分。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
import certifi

ROOT = Path(__file__).resolve().parent
URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
MIRROR = "https://raw.githubusercontent.com/aws-samples/aws-stepfunctions-byoc-mlops-using-data-science-sdk/master/PennFudanPed.zip"
# 本项目首次验证的压缩包指纹，用于防止后续下载内容发生变化。
SHA256 = "9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2"


def prepare(data_dir, source="official"):
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "PennFudanPed.zip"
    origin_path = data_dir / "archive_source.json"
    download_url = URL if source == "official" else MIRROR
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {download_url}", flush=True)
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(download_url, timeout=120, context=context) as response, partial.open("wb") as out:
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloaded {downloaded / 1024**2:.1f} MiB", flush=True)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
                raise ValueError("压缩包 SHA-256 与项目记录不一致，请检查下载源")
            partial.replace(archive)
            origin_path.write_text(json.dumps({"download_url": download_url}, indent=2) + "\n")
        finally:
            partial.unlink(missing_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError("本地压缩包 SHA-256 与项目记录不一致，请检查文件")
    with zipfile.ZipFile(archive) as z:
        corrupt = z.testzip()
        if corrupt:
            raise ValueError(f"损坏的压缩包条目：{corrupt}；请删除压缩包后重新下载")
        if not (data_dir / "PennFudanPed").exists():
            # 先检查路径再解压到临时目录，成功后才移动到最终位置。
            with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
                destination = Path(tmp).resolve()
                for info in z.infolist():
                    target = (destination / info.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError(f"非法压缩包路径：{info.filename}")
                z.extractall(destination)
                shutil.move(str(destination / "PennFudanPed"), data_dir / "PennFudanPed")
    folder = data_dir / "PennFudanPed"
    names = sorted(p.stem for p in (folder / "PNGImages").glob("*.png"))
    if len(names) != 170:
        raise ValueError(f"预期 170 张图片，实际 {len(names)} 张")
    for name in names:
        if not (folder / "PedMasks" / f"{name}_mask.png").exists():
            raise FileNotFoundError(f"缺少标注：{name}")
    random.Random(42).shuffle(names)
    splits = {"train": names[:120], "val": names[120:145], "test": names[145:]}
    (data_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    metadata = {
        "name": "Penn-Fudan Pedestrian Detection and Segmentation",
        "source": "https://www.cis.upenn.edu/~jshi/ped_html/",
        "official_download_url": URL,
        "download_url": json.loads(origin_path.read_text()).get("download_url") if origin_path.exists() else None,
        "archive_bytes": archive.stat().st_size,
        "sha256": digest,
        "images": len(names), "split_seed": 42,
        "counts": {key: len(value) for key, value in splits.items()},
        "task": "binary semantic segmentation: mask > 0 is pedestrian",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Ready: {folder}\nSplit: train=120, val=25, test=25", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source", choices=["official", "mirror"], default="official", help="mirror 为 AWS 官方示例仓库中的备份")
    args = parser.parse_args()
    prepare(args.data_dir, args.source)
