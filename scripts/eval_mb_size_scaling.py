#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from tabicl.prior.dataset import PriorDataset, infer_mb_pseudo_labels
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


DEFAULT_CPU_METHODS = ["iamb_fdr", "mmpc", "corr_fdr", "mi_fdr", "lasso"]
DEFAULT_GPU_METHODS = ["mlp_saliency", "tabtransformer_probe"]


def parse_methods(text: str) -> list[str]:
    return [m.strip().lower() for m in text.split(",") if m.strip()]


def safe_div(x: float, y: float) -> float:
    return float(x) / float(y) if y != 0 else 0.0


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * prec * rec, prec + rec)
    tnr = safe_div(tn, tn + fp)
    bal_acc = 0.5 * (rec + tnr)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "bal_acc": bal_acc,
    }


def summarize_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups[key].append(row)

    summaries: list[dict] = []
    for key, items in groups.items():
        out = {k: v for k, v in zip(group_keys, key)}
        out["n_records"] = len(items)
        for mk in metric_keys:
            vals = [float(it[mk]) for it in items if np.isfinite(float(it[mk]))]
            if vals:
                out[f"{mk}_mean"] = float(np.mean(vals))
                out[f"{mk}_std"] = float(np.std(vals))
            else:
                out[f"{mk}_mean"] = float("nan")
                out[f"{mk}_std"] = float("nan")
        summaries.append(out)

    def _sort_key(d: dict):
        return tuple(d[k] for k in group_keys)

    summaries.sort(key=_sort_key)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MB pseudo-label quality over multiple dataset sizes. "
            "Classical methods run on CPU, neural methods run on GPU."
        )
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results/mb_size_scaling")
    parser.add_argument("--size_start", type=int, default=1000)
    parser.add_argument("--size_step", type=int, default=2000)
    parser.add_argument("--n_sizes", type=int, default=50)
    parser.add_argument("--datasets_per_size", type=int, default=1)

    parser.add_argument("--cpu_methods", type=str, default=",".join(DEFAULT_CPU_METHODS))
    parser.add_argument("--gpu_methods", type=str, default=",".join(DEFAULT_GPU_METHODS))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_per_gp", type=int, default=1)
    parser.add_argument("--min_features", type=int, default=32)
    parser.add_argument("--max_features", type=int, default=32)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--min_train_size", type=float, default=0.1)
    parser.add_argument("--max_train_size", type=float, default=0.9)
    parser.add_argument("--dag_edge_prob", type=float, default=0.3)
    parser.add_argument("--dag_edge_drop_prob", type=float, default=0.2)
    parser.add_argument("--ci_alpha", type=float, default=0.01)
    parser.add_argument("--max_condition_set", type=int, default=2)
    parser.add_argument("--nn_train_steps", type=int, default=180)
    parser.add_argument("--nn_batch_size", type=int, default=1024)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_sizes <= 0:
        raise ValueError("--n_sizes must be > 0")
    if args.datasets_per_size <= 0:
        raise ValueError("--datasets_per_size must be > 0")

    cpu_methods = parse_methods(args.cpu_methods)
    gpu_methods = parse_methods(args.gpu_methods)
    methods = cpu_methods + gpu_methods
    if not methods:
        raise ValueError("No methods configured.")

    size_list = [args.size_start + i * args.size_step for i in range(args.n_sizes)]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.cpu_threads)))

    has_cuda = torch.cuda.is_available()
    if gpu_methods and not has_cuda:
        print("[warn] CUDA unavailable. GPU methods will run on CPU.")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    detail_csv = os.path.join(run_dir, "detail_rows.csv")
    summary_size_method_csv = os.path.join(run_dir, "summary_by_size_method.csv")
    summary_method_csv = os.path.join(run_dir, "summary_by_method.csv")

    meta_txt = os.path.join(run_dir, "meta.txt")
    with open(meta_txt, "w") as f:
        f.write(f"size_start={args.size_start}\n")
        f.write(f"size_step={args.size_step}\n")
        f.write(f"n_sizes={args.n_sizes}\n")
        f.write(f"datasets_per_size={args.datasets_per_size}\n")
        f.write(f"cpu_methods={cpu_methods}\n")
        f.write(f"gpu_methods={gpu_methods}\n")
        f.write(f"has_cuda={has_cuda}\n")
        f.write(f"min_features={args.min_features}\n")
        f.write(f"max_features={args.max_features}\n")
        f.write(f"ci_alpha={args.ci_alpha}\n")
        f.write(f"max_condition_set={args.max_condition_set}\n")
        f.write(f"nn_train_steps={args.nn_train_steps}\n")
        f.write(f"nn_batch_size={args.nn_batch_size}\n")

    scm_fixed_hp = dict(DEFAULT_FIXED_HP)
    scm_sampled_hp = dict(DEFAULT_SAMPLED_HP)
    scm_fixed_hp["graph_sparsity"] = 0.0
    scm_fixed_hp["is_causal"] = True
    scm_fixed_hp["y_is_effect"] = True
    scm_fixed_hp["edge_prob"] = float(args.dag_edge_prob)
    scm_fixed_hp["edge_drop_prob"] = float(args.dag_edge_drop_prob)
    scm_sampled_hp.pop("is_causal", None)
    scm_sampled_hp.pop("y_is_effect", None)

    fieldnames = [
        "n_samples",
        "dataset_index",
        "method",
        "device",
        "n_features",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "bal_acc",
        "runtime_sec",
        "ok",
        "error",
    ]
    rows: list[dict] = []

    with open(detail_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        fcsv.flush()

        for size_idx, n_samples in enumerate(size_list):
            print(
                f"[size {size_idx + 1}/{len(size_list)}] n_samples={n_samples} "
                f"datasets_per_size={args.datasets_per_size}"
            )
            dataset = PriorDataset(
                batch_size=args.batch_size,
                batch_size_per_gp=args.batch_size_per_gp,
                min_features=args.min_features,
                max_features=args.max_features,
                max_classes=args.max_classes,
                min_seq_len=n_samples,
                max_seq_len=n_samples + 1,
                log_seq_len=False,
                min_train_size=args.min_train_size,
                max_train_size=args.max_train_size,
                replay_small=False,
                prior_type="mlp_scm",
                scm_fixed_hp=scm_fixed_hp,
                scm_sampled_hp=scm_sampled_hp,
                return_x_node_binary=True,
                node_pseudo_label_method="none",
                device="cpu",
                n_jobs=1,
            )

            for rep in range(args.datasets_per_size):
                X, y, d, seq_lens, _train_sizes, x_node_binary = dataset.get_batch()
                for bi in range(X.shape[0]):
                    seq_len_i = int(seq_lens[bi].item())
                    d_i = int(d[bi].item())
                    if d_i <= 0 or seq_len_i <= 3:
                        continue

                    X_i = X[bi, :seq_len_i, :d_i]
                    y_i = y[bi, :seq_len_i]
                    true_i = x_node_binary[bi, :d_i].long()

                    for method in methods:
                        method_is_gpu = method in gpu_methods and has_cuda
                        target_device = torch.device("cuda" if method_is_gpu else "cpu")
                        start = time.perf_counter()
                        err = ""
                        ok = 1
                        try:
                            Xin = X_i.to(target_device, non_blocking=True)
                            yin = y_i.to(target_device, non_blocking=True)
                            pred_i = infer_mb_pseudo_labels(
                                Xin,
                                yin,
                                method=method,
                                ci_alpha=args.ci_alpha,
                                max_condition_set=args.max_condition_set,
                                nn_train_steps=args.nn_train_steps,
                                nn_batch_size=args.nn_batch_size,
                            ).to("cpu")
                            if method_is_gpu:
                                torch.cuda.synchronize()
                        except Exception as e:
                            pred_i = torch.zeros_like(true_i)
                            ok = 0
                            err = str(e).replace("\n", " ")[:300]
                        runtime_sec = time.perf_counter() - start

                        tp = int(((pred_i == 1) & (true_i == 1)).sum().item())
                        fp = int(((pred_i == 1) & (true_i == 0)).sum().item())
                        tn = int(((pred_i == 0) & (true_i == 0)).sum().item())
                        fn = int(((pred_i == 0) & (true_i == 1)).sum().item())
                        metrics = compute_metrics(tp, fp, tn, fn)

                        row = {
                            "n_samples": n_samples,
                            "dataset_index": rep * int(args.batch_size) + bi,
                            "method": method,
                            "device": str(target_device),
                            "n_features": d_i,
                            "tp": tp,
                            "fp": fp,
                            "tn": tn,
                            "fn": fn,
                            "accuracy": metrics["accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1": metrics["f1"],
                            "bal_acc": metrics["bal_acc"],
                            "runtime_sec": runtime_sec,
                            "ok": ok,
                            "error": err,
                        }
                        rows.append(row)
                        writer.writerow(row)
                        fcsv.flush()

    metric_keys = ["accuracy", "precision", "recall", "f1", "bal_acc", "runtime_sec"]
    by_size_method = summarize_rows(rows, ["n_samples", "method"], metric_keys)
    by_method = summarize_rows(rows, ["method"], metric_keys)

    with open(summary_size_method_csv, "w", newline="") as f:
        fieldnames = ["n_samples", "method", "n_records"] + [
            f"{k}_{suffix}" for k in metric_keys for suffix in ("mean", "std")
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_size_method:
            writer.writerow(row)

    with open(summary_method_csv, "w", newline="") as f:
        fieldnames = ["method", "n_records"] + [
            f"{k}_{suffix}" for k in metric_keys for suffix in ("mean", "std")
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_method:
            writer.writerow(row)

    print("\n=== Run complete ===")
    print(f"run_dir: {run_dir}")
    print(f"detail_csv: {detail_csv}")
    print(f"summary_size_method_csv: {summary_size_method_csv}")
    print(f"summary_method_csv: {summary_method_csv}")
    print("\n=== Overall method summary (accuracy_mean, recall_mean, runtime_sec_mean) ===")
    for row in by_method:
        print(
            f"{row['method']}: "
            f"acc={row['accuracy_mean']:.4f}, "
            f"recall={row['recall_mean']:.4f}, "
            f"time={row['runtime_sec_mean']:.2f}s, "
            f"n={row['n_records']}"
        )


if __name__ == "__main__":
    main()
