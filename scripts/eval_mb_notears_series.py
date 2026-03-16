#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from tabicl.prior.dataset import PriorDataset
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


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

    out: list[dict] = []
    for key, items in groups.items():
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_records"] = len(items)
        for mk in metric_keys:
            vals = [float(it[mk]) for it in items if np.isfinite(float(it[mk]))]
            rec[f"{mk}_mean"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{mk}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(rec)
    out.sort(key=lambda d: tuple(d[k] for k in group_keys))
    return out


def mb_from_adj(adj: np.ndarray, y_idx: int) -> set[int]:
    adj_bin = (np.asarray(adj) != 0).astype(np.int64)
    parents = set(np.where(adj_bin[:, y_idx] == 1)[0].tolist())
    children = set(np.where(adj_bin[y_idx, :] == 1)[0].tolist())
    spouses: set[int] = set()
    for c in children:
        spouses.update(np.where(adj_bin[:, c] == 1)[0].tolist())
    mb = (parents | children | spouses) - {y_idx}
    return {int(i) for i in mb}


def standardize_cols(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def infer_mb_notears(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    max_rows_linear: int,
    max_rows_nonlinear: int,
    seed: int,
    w_threshold: float,
    linear_max_iter: int,
    lowrank_max_iter: int,
    nonlinear_max_iter: int,
    nonlinear_hidden: tuple[int, ...],
    nonlinear_device: str,
) -> set[int]:
    from castle.algorithms import Notears, NotearsLowRank, NotearsNonlinear

    n, d = X.shape
    y_col = y.reshape(-1, 1)
    data = np.concatenate([X, y_col], axis=1).astype(np.float64, copy=False)

    if method == "notears_nonlinear":
        row_cap = max_rows_nonlinear
    else:
        row_cap = max_rows_linear
    if n > row_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=row_cap, replace=False)
        data = data[idx]

    data = standardize_cols(data)

    if method == "notears":
        learner = Notears(
            lambda1=0.05,
            loss_type="l2",
            max_iter=int(linear_max_iter),
            w_threshold=float(w_threshold),
        )
    elif method == "notears_lowrank":
        learner = NotearsLowRank(
            max_iter=int(lowrank_max_iter),
            w_threshold=float(w_threshold),
        )
    elif method == "notears_nonlinear":
        learner = NotearsNonlinear(
            lambda1=0.01,
            lambda2=0.01,
            max_iter=int(nonlinear_max_iter),
            w_threshold=float(w_threshold),
            hidden_layers=nonlinear_hidden,
            model_type="mlp",
            device_type=nonlinear_device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    learner.learn(data)
    if not hasattr(learner, "causal_matrix"):
        raise RuntimeError(f"{method} learner has no causal_matrix output")
    adj = np.asarray(learner.causal_matrix)
    y_idx = d
    mb_all = mb_from_adj(adj, y_idx=y_idx)
    return {i for i in mb_all if 0 <= i < d}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NOTEARS-series methods for MB prediction.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/mb_notears_series")
    parser.add_argument("--methods", type=str, default="notears,notears_lowrank,notears_nonlinear")

    parser.add_argument("--size_start", type=int, default=1000)
    parser.add_argument("--size_step", type=int, default=2000)
    parser.add_argument("--n_sizes", type=int, default=50)
    parser.add_argument("--datasets_per_size", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_size_per_gp", type=int, default=1)
    parser.add_argument("--min_features", type=int, default=32)
    parser.add_argument("--max_features", type=int, default=32)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--min_train_size", type=float, default=0.1)
    parser.add_argument("--max_train_size", type=float, default=0.9)
    parser.add_argument("--dag_edge_prob", type=float, default=0.3)
    parser.add_argument("--dag_edge_drop_prob", type=float, default=0.2)

    parser.add_argument("--max_rows_linear", type=int, default=20000)
    parser.add_argument("--max_rows_nonlinear", type=int, default=6000)
    parser.add_argument("--linear_max_iter", type=int, default=60)
    parser.add_argument("--lowrank_max_iter", type=int, default=20)
    parser.add_argument("--nonlinear_max_iter", type=int, default=60)
    parser.add_argument("--nonlinear_hidden", type=str, default="32,8,1")
    parser.add_argument("--w_threshold", type=float, default=0.3)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = parse_methods(args.methods)
    if not methods:
        raise ValueError("No methods configured.")

    hidden = tuple(int(x.strip()) for x in args.nonlinear_hidden.split(",") if x.strip())
    if len(hidden) == 0:
        raise ValueError("--nonlinear_hidden must be non-empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("castle").setLevel(logging.ERROR)
    logging.getLogger("castle.algorithms").setLevel(logging.ERROR)

    has_cuda = torch.cuda.is_available()
    nonlinear_device = "gpu" if has_cuda else "cpu"

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    detail_csv = os.path.join(run_dir, "detail_rows.csv")
    summary_size_method_csv = os.path.join(run_dir, "summary_by_size_method.csv")
    summary_method_csv = os.path.join(run_dir, "summary_by_method.csv")
    meta_txt = os.path.join(run_dir, "meta.txt")

    with open(meta_txt, "w") as f:
        f.write(f"methods={methods}\n")
        f.write(f"size_start={args.size_start}\n")
        f.write(f"size_step={args.size_step}\n")
        f.write(f"n_sizes={args.n_sizes}\n")
        f.write(f"datasets_per_size={args.datasets_per_size}\n")
        f.write(f"max_rows_linear={args.max_rows_linear}\n")
        f.write(f"max_rows_nonlinear={args.max_rows_nonlinear}\n")
        f.write(f"linear_max_iter={args.linear_max_iter}\n")
        f.write(f"lowrank_max_iter={args.lowrank_max_iter}\n")
        f.write(f"nonlinear_max_iter={args.nonlinear_max_iter}\n")
        f.write(f"nonlinear_hidden={hidden}\n")
        f.write(f"w_threshold={args.w_threshold}\n")
        f.write(f"has_cuda={has_cuda}\n")
        f.write(f"nonlinear_device={nonlinear_device}\n")

    scm_fixed_hp = dict(DEFAULT_FIXED_HP)
    scm_sampled_hp = dict(DEFAULT_SAMPLED_HP)
    scm_fixed_hp["graph_sparsity"] = 0.0
    scm_fixed_hp["is_causal"] = True
    scm_fixed_hp["y_is_effect"] = True
    scm_fixed_hp["edge_prob"] = float(args.dag_edge_prob)
    scm_fixed_hp["edge_drop_prob"] = float(args.dag_edge_drop_prob)
    scm_sampled_hp.pop("is_causal", None)
    scm_sampled_hp.pop("y_is_effect", None)

    size_list = [args.size_start + i * args.size_step for i in range(args.n_sizes)]

    fieldnames = [
        "n_samples",
        "dataset_index",
        "method",
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
            print(f"[size {size_idx + 1}/{len(size_list)}] n_samples={n_samples}")
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

                    X_i = X[bi, :seq_len_i, :d_i].cpu().numpy().astype(np.float64, copy=False)
                    y_i = y[bi, :seq_len_i].cpu().numpy().astype(np.float64, copy=False)
                    true_i = x_node_binary[bi, :d_i].long()

                    for method in methods:
                        start = time.perf_counter()
                        ok = 1
                        err = ""
                        try:
                            selected = infer_mb_notears(
                                X=X_i,
                                y=y_i,
                                method=method,
                                max_rows_linear=int(args.max_rows_linear),
                                max_rows_nonlinear=int(args.max_rows_nonlinear),
                                seed=int(args.seed + size_idx * 17 + rep * 3 + bi),
                                w_threshold=float(args.w_threshold),
                                linear_max_iter=int(args.linear_max_iter),
                                lowrank_max_iter=int(args.lowrank_max_iter),
                                nonlinear_max_iter=int(args.nonlinear_max_iter),
                                nonlinear_hidden=hidden,
                                nonlinear_device=nonlinear_device,
                            )
                        except Exception as e:
                            selected = set()
                            ok = 0
                            err = str(e).replace("\n", " ")[:320]

                        runtime_sec = time.perf_counter() - start
                        pred_i = torch.zeros(d_i, dtype=torch.long)
                        for j in selected:
                            if 0 <= j < d_i:
                                pred_i[j] = 1

                        tp = int(((pred_i == 1) & (true_i == 1)).sum().item())
                        fp = int(((pred_i == 1) & (true_i == 0)).sum().item())
                        tn = int(((pred_i == 0) & (true_i == 0)).sum().item())
                        fn = int(((pred_i == 0) & (true_i == 1)).sum().item())
                        metrics = compute_metrics(tp, fp, tn, fn)

                        row = {
                            "n_samples": n_samples,
                            "dataset_index": rep * int(args.batch_size) + bi,
                            "method": method,
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
    print("\n=== Method Summary (accuracy_mean, recall_mean, runtime_sec_mean) ===")
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
