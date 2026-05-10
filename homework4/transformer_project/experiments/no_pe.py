"""Run positional encoding ablations: sinusoidal vs learned vs none."""

from __future__ import annotations

import subprocess


def run(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    run("python train.py --pe_mode sinusoidal --save checkpoints/pe_sinusoidal.pt --log_csv logs/pe_sinusoidal_train.csv")
    run("python train.py --pe_mode learned --save checkpoints/pe_learned.pt --log_csv logs/pe_learned_train.csv")
    run("python train.py --pe_mode none --save checkpoints/pe_none.pt --log_csv logs/pe_none_train.csv")

    run("python eval.py --ckpt checkpoints/pe_sinusoidal.pt --save_json logs/pe_sinusoidal_eval.json")
    run("python eval.py --ckpt checkpoints/pe_learned.pt --save_json logs/pe_learned_eval.json")
    run("python eval.py --ckpt checkpoints/pe_none.pt --save_json logs/pe_none_eval.json")

    print("位置编码核心作用：提供序列顺序信息，使 self-attention 区分 token 的相对/绝对位置。")


if __name__ == "__main__":
    main()
