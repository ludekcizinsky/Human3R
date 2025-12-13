#!/usr/bin/env python3
"""
Utility to compare two SMPL-X parameter JSON files.

Defaults point to the HI4D example paths:
  - Pipeline A: /scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair00_fight/lhm/motion/00/smplx_params/00001.json
  - Pipeline B: /scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair00_fight/lhm/motion_human3r/00/smplx_params/00000.json
"""

import argparse
import json
from pathlib import Path
import numpy as np


SMPL_KEYS = [
    "betas",
    "root_pose",
    "body_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "lhand_pose",
    "rhand_pose",
    "trans",
]


def to_array(item):
    """Convert nested lists to float32 numpy arrays."""
    return np.asarray(item, dtype=np.float32)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_arrays(name, a, b, atol):
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "shape_a": a.shape,
            "shape_b": b.shape,
        }
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "shape_match": True,
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "l2_norm": float(np.linalg.norm(diff)) if abs_diff.size else 0.0,
        "within_tol": bool(np.all(abs_diff <= atol)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two SMPL-X param JSON files.")
    parser.add_argument(
        "--a",
        type=Path,
        default=Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair00_fight/lhm/motion/00/smplx_params/00001.json"),
        help="Path to first SMPL-X JSON (reference).",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair00_fight/lhm/motion_human3r/00/smplx_params/00000.json"),
        help="Path to second SMPL-X JSON (to compare).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for within_tol flag.",
    )
    args = parser.parse_args()

    data_a = load_json(args.a)
    data_b = load_json(args.b)

    print(f"Comparing:\n  A: {args.a}\n  B: {args.b}\n  atol: {args.atol}\n")
    for key in SMPL_KEYS:
        if key not in data_a or key not in data_b:
            print(f"{key}: missing (A has: {key in data_a}, B has: {key in data_b})")
            continue
        arr_a = to_array(data_a[key])
        arr_b = to_array(data_b[key])
        stats = compare_arrays(key, arr_a, arr_b, args.atol)
        if not stats["shape_match"]:
            print(f"{key}: shape mismatch A{stats['shape_a']} vs B{stats['shape_b']}")
        else:
            print(
                f"{key}: max_abs={stats['max_abs']:.4e}, mean_abs={stats['mean_abs']:.4e}, "
                f"l2_norm={stats['l2_norm']:.4e}, within_tol={stats['within_tol']}"
            )


if __name__ == "__main__":
    main()
