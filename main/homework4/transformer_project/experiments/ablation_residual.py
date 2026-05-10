"""Run residual ablation experiments."""

from __future__ import annotations

import subprocess


def run(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    run("python train.py --save checkpoints/residual_on.pt --log_csv logs/residual_on_train.csv")
    run("python train.py --no_residual --save checkpoints/residual_off.pt --log_csv logs/residual_off_train.csv")

    run("python eval.py --ckpt checkpoints/residual_on.pt --save_json logs/residual_on_eval.json")
    run("python eval.py --ckpt checkpoints/residual_off.pt --save_json logs/residual_off_eval.json")

    print("Residual作用：稳定深层训练、改善梯度传播、提升收敛速度与最终性能。")


if __name__ == "__main__":
    main()
