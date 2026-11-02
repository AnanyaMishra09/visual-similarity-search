"""Runs the three required ablations (margin, embedding dim, mining
strategy) as separate train.py subprocesses, then collects their
summary.json files into results/ablations.md.

The baseline config (margin=0.3, embed_dim=128, mining=hard) is shared
across all three ablation axes, so it's trained once and reused rather than
retrained 7 times for 7 cells.

Usage:
    python scripts/run_ablations.py --epochs 30 --out-root runs
    python scripts/run_ablations.py --epochs 3 --out-root runs_quick  # faster, for iterating
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def run_variant(name, overrides, args):
    out_dir = os.path.join(args.out_root, name)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path) and not args.force:
        print(f"[skip] {name} already has results at {summary_path}")
        return summary_path

    cmd = [
        sys.executable, os.path.join(SRC, "train.py"),
        "--data-root", args.data_root,
        "--out-dir", out_dir,
        "--epochs", str(args.epochs),
        "--batch-p", str(args.batch_p),
        "--batch-k", str(args.batch_k),
        "--seed", str(args.seed),
    ]
    for k, v in overrides.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    print(f"[run] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out-root", default="./runs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-p", type=int, default=16)
    p.add_argument("--batch-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    baseline = dict(margin=0.3, embed_dim=128, mining="hard", optimizer="adam", lr=1e-3)

    variants = {
        "baseline_m0.3_d128_hard": baseline,
        "margin_0.1": {**baseline, "margin": 0.1},
        "margin_0.5": {**baseline, "margin": 0.5},
        "dim_64": {**baseline, "embed_dim": 64},
        "dim_256": {**baseline, "embed_dim": 256},
        "mining_random": {**baseline, "mining": "random"},
    }

    summaries = {}
    for name, overrides in variants.items():
        path = run_variant(name, overrides, args)
        with open(path) as f:
            summaries[name] = json.load(f)

    os.makedirs("results", exist_ok=True)
    with open("results/ablations_raw.json", "w") as f:
        json.dump(summaries, f, indent=2)

    lines = ["| Variant | margin | embed_dim | mining | R@1 | R@5 | R@10 | mAP |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        a = s["args"]
        m = s["test_metrics"]
        lines.append(
            f"| {name} | {a['margin']} | {a['embed_dim']} | {a['mining']} | "
            f"{m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mAP']:.3f} |"
        )
    table = "\n".join(lines)
    with open("results/ablations.md", "w") as f:
        f.write("# Ablation results\n\n" + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
