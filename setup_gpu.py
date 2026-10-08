"""在 Linux 服务器创建独立环境，安装 CUDA 版 PyTorch 并自检。"""
import argparse
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", choices=["cu126", "cu118", "cu128"], default="cu126")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("此安装脚本用于 Linux NVIDIA 服务器，请同步后在服务器执行")
    subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
    python = str(ROOT / ".venv" / "bin" / "python")
    commands = [
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        [python, "-m", "pip", "install", f"torch==2.7.1+{args.cuda}",
         "--index-url", f"https://download.pytorch.org/whl/{args.cuda}"],
        [python, "-m", "pip", "install", "-r", str(ROOT / "requirements-gpu.txt")],
        [python, "-m", "pip", "check"],
        [python, str(ROOT / "check_gpu.py")],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print("安装和 GPU 自检通过。在项目目录运行：.venv/bin/python run_gpu.py")


if __name__ == "__main__":
    main()
