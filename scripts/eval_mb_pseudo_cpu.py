#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from tabicl.prior.dataset import PriorDataset, infer_mb_pseudo_labels
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only MB pseudo-label evaluation using synthetic SCM data."
    )
    parser.add_argument("--method", type=str, default="iamb_fdr", choices=["iamb_fdr", "mmpc"])
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_per_gp", type=int, default=4)
    parser.add_argument("--min_features", type=int, default=8)
    parser.add_argument("--max_features", type=int, default=64)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--min_train_size", type=float, default=0.1)
    parser.add_argument("--max_train_size", type=float, default=0.9)
    parser.add_argument("--dag_edge_prob", type=float, default=0.3)
    parser.add_argument("--dag_edge_drop_prob", type=float, default=0.2)
    parser.add_argument("--ci_alpha", type=float, default=0.01)
    parser.add_argument("--max_condition_set", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scm_fixed_hp = dict(DEFAULT_FIXED_HP)
    scm_sampled_hp = dict(DEFAULT_SAMPLED_HP)
    scm_fixed_hp["graph_sparsity"] = 0.0
    scm_fixed_hp["edge_prob"] = float(args.dag_edge_prob)
    scm_fixed_hp["edge_drop_prob"] = float(args.dag_edge_drop_prob)

    dataset = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        max_seq_len=args.max_seq_len,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        prior_type="mlp_scm",
        scm_fixed_hp=scm_fixed_hp,
        scm_sampled_hp=scm_sampled_hp,
        return_x_node_binary=True,
        node_pseudo_label_method="none",
        device="cpu",
        n_jobs=1,
    )

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    dataset_accs: list[float] = []

    for _ in range(args.n_batches):
        X, y, d, seq_lens, _train_sizes, x_node_binary = dataset.get_batch()
        for bi in range(X.shape[0]):
            seq_len_i = int(seq_lens[bi].item())
            d_i = int(d[bi].item())
            if d_i <= 0 or seq_len_i <= 3:
                continue

            X_i = X[bi, :seq_len_i, :d_i]
            y_i = y[bi, :seq_len_i]
            true_i = x_node_binary[bi, :d_i].long()

            pseudo_i = infer_mb_pseudo_labels(
                X_i,
                y_i,
                method=args.method,
                ci_alpha=args.ci_alpha,
                max_condition_set=args.max_condition_set,
            ).long()

            eq = (pseudo_i == true_i)
            dataset_accs.append(float(eq.float().mean().item()))

            tp = int(((pseudo_i == 1) & (true_i == 1)).sum().item())
            fp = int(((pseudo_i == 1) & (true_i == 0)).sum().item())
            tn = int(((pseudo_i == 0) & (true_i == 0)).sum().item())
            fn = int(((pseudo_i == 0) & (true_i == 1)).sum().item())

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

    total = total_tp + total_fp + total_tn + total_fn
    if total == 0:
        raise RuntimeError("No valid samples were evaluated.")

    accuracy = (total_tp + total_tn) / total
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    dataset_acc_mean = float(np.mean(dataset_accs)) if dataset_accs else float("nan")
    dataset_acc_std = float(np.std(dataset_accs)) if dataset_accs else float("nan")

    print("=== MB Pseudo Label CPU Eval ===")
    print(f"method={args.method}")
    print(
        f"config: batches={args.n_batches}, batch_size={args.batch_size}, "
        f"features=[{args.min_features},{args.max_features}], max_seq_len={args.max_seq_len}"
    )
    print(
        f"ci: alpha={args.ci_alpha}, max_condition_set={args.max_condition_set}, "
        f"edge_prob={args.dag_edge_prob}, edge_drop_prob={args.dag_edge_drop_prob}"
    )
    print(
        f"confusion: TP={total_tp} FP={total_fp} TN={total_tn} FN={total_fn} total={total}"
    )
    print(
        f"metrics: accuracy={accuracy:.4f} precision={precision:.4f} "
        f"recall={recall:.4f} f1={f1:.4f}"
    )
    print(
        f"dataset_acc: mean={dataset_acc_mean:.4f} std={dataset_acc_std:.4f} n={len(dataset_accs)}"
    )


if __name__ == "__main__":
    main()

