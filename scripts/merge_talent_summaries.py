#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


def extract_step(model_name: str) -> int | None:
    nums = re.findall(r"\d+", model_name)
    if not nums:
        return None
    return int(nums[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge and sort multi-node TALENT summaries by checkpoint step.")
    p.add_argument("--input_glob", type=str, required=True, help="Glob for per-node all_models_summary.tsv files.")
    p.add_argument("--output", type=str, required=True, help="Output merged tsv file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    rows = []
    seen = set()
    for f in files:
        path = Path(f)
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            for row in reader:
                model_name = (row.get("model_name") or "").strip()
                if not model_name:
                    continue
                # Deduplicate by model name in case of reruns.
                if model_name in seen:
                    continue
                seen.add(model_name)
                rows.append(
                    {
                        "model_name": model_name,
                        "total_datasets": (row.get("total_datasets") or "").strip(),
                        "average_accuracy": (row.get("average_accuracy") or "").strip(),
                        "total_time_s": (row.get("total_time_s") or "").strip(),
                        "average_time_s": (row.get("average_time_s") or "").strip(),
                        "average_train_ratio": (row.get("average_train_ratio") or "").strip(),
                    }
                )

    def sort_key(r: dict) -> tuple[int, int, str]:
        step = extract_step(r["model_name"])
        if step is None:
            return (1, 0, r["model_name"])
        return (0, step, r["model_name"])

    rows.sort(key=sort_key)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(
            [
                "model_name",
                "total_datasets",
                "average_accuracy",
                "total_time_s",
                "average_time_s",
                "average_train_ratio",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["model_name"],
                    r["total_datasets"],
                    r["average_accuracy"],
                    r["total_time_s"],
                    r["average_time_s"],
                    r["average_train_ratio"],
                ]
            )

    print(f"merged_files={len(files)} rows={len(rows)} output={out}")


if __name__ == "__main__":
    main()
