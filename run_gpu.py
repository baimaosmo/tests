"""Python 一键入口：准备数据 → 训练 U-Net → 生成预测图，默认使用 CUDA。"""
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

from data import ROOT, choose_device
from train import build_parser


def run_logged(command, log):
    """使用当前 Python 环境运行子脚本，同时输出到终端和日志。"""
    line = f"\n$ {shlex.join(command)}\n"
    print(line, end="", flush=True)
    log.write(line)
    log.flush()
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=None)
    parser.add_argument("--count", type=int, default=8, help="训练后生成多少张测试图预测")
    parser.add_argument("--source", choices=["official", "mirror"], default="official", help="缺少数据时使用的下载源")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count 必须为正")
    try:
        device = choose_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    args.device = str(device)
    run_name = f"gpu_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    args.output_dir = args.output_dir or Path(os.environ.get("RUN_DIR") or ROOT / "runs" / run_name)
    for key in ("data_dir", "output_dir"):
        setattr(args, key, (ROOT / getattr(args, key)).resolve())
    if (args.output_dir / "best.pt").exists():
        parser.error("输出目录已有模型，请通过 --output-dir 指定新的目录")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open("a", encoding="utf-8") as log:
        run_logged([sys.executable, "-u", str(ROOT / "download_data.py"),
                    "--data-dir", str(args.data_dir), "--source", args.source], log)
        with (args.output_dir / "requirements-server.txt").open("w", encoding="utf-8") as environment:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=environment, check=True)
        command = [sys.executable, "-u", str(ROOT / "train.py")]
        for key, value in vars(args).items():
            if key in ("source", "count") or value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                command.append(flag if value else "--no-" + key.replace("_", "-"))
            else:
                command.extend([flag, str(value)])
        run_logged(command, log)
        run_logged([sys.executable, "-u", str(ROOT / "predict.py"),
                    "--device", args.device, "--data-dir", str(args.data_dir),
                    "--checkpoint", str(args.output_dir / "best.pt"),
                    "--output-dir", str(args.output_dir / "predictions"),
                    "--count", str(args.count)], log)
    print(f"\n完成！模型、指标、日志和预测图位于：{args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"运行失败（退出码 {error.returncode}），请检查上面的输出及 train.log。", file=sys.stderr)
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except KeyboardInterrupt:
        print("\n运行已中断。", file=sys.stderr)
        sys.exit(130)
