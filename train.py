#!/usr/bin/env python3
"""
MudraForge Decompiler - Unified Training & RL Alignment Engine
=============================================================
Provides reproducible command-line interfaces for:
  1. Synthetic Dataset Generation (computational tax, commercial law, non-computational states)
  2. Stage 1: Supervised Disentangled Dual-Stream Training (DomainSensor + LogicReasoner)
  3. Stage 2: Execution-Guided GRPO Reinforcement Learning (LogicReasoner + USVM Rewards)

Usage:
  python train.py --smoke-test
  python train.py --stage dataset --samples 3000
  python train.py --stage supervised --epochs 10
  python train.py --stage grpo --epochs 5
  python train.py --stage all
"""

import argparse
import sys
import time
from pathlib import Path

from compiler.training.dataset_generator import create_comprehensive_dataset, generate_synthetic_dataset
from compiler.training.train_dual_stream import (
    train_supervised_dual_stream,
    train_grpo_logic_only,
    DEFAULT_SUP_CKPT,
    DEFAULT_RL_CKPT,
    DEFAULT_DATASET,
)


def run_smoke_test():
    print("=" * 75)
    print("MUDRAFORGE DECOMPILER - TRAINING PIPELINE SMOKE TEST")
    print("=" * 75)
    t0 = time.time()

    print("[1/3] Generating in-memory synthetic training pairs...")
    smoke_dataset = generate_synthetic_dataset(num_samples=30)
    print(f"      Train instances: {len(smoke_dataset['train'])} | Val instances: {len(smoke_dataset['val'])}")

    smoke_sup_ckpt = Path("checkpoints/smoke_test_sup.pt")
    smoke_rl_ckpt = Path("checkpoints/smoke_test_rl.pt")

    print("\n[2/3] Executing Stage 1 Supervised forward/backward pass (1 epoch)...")
    train_supervised_dual_stream(
        dataset_path=smoke_dataset,
        checkpoint_out=smoke_sup_ckpt,
        epochs=1,
        smoke_test=True,
    )
    assert smoke_sup_ckpt.exists(), "Supervised smoke checkpoint was not saved."
    print("      Stage 1 Supervised passed successfully.")

    print("\n[3/3] Executing Stage 2 GRPO RL execution loop (USVM rewards)...")
    train_grpo_logic_only(
        checkpoint_in=smoke_sup_ckpt,
        checkpoint_out=smoke_rl_ckpt,
        dataset_path=smoke_dataset,
        epochs=1,
        smoke_test=True,
    )
    assert smoke_rl_ckpt.exists(), "GRPO smoke checkpoint was not saved."
    print("      Stage 2 GRPO RL passed successfully.")

    # Clean up temporary smoke test checkpoints
    try:
        if smoke_sup_ckpt.exists():
            smoke_sup_ckpt.unlink()
        if smoke_rl_ckpt.exists():
            smoke_rl_ckpt.unlink()
    except Exception:
        pass

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"[PASS] Full training pipeline smoke test verified in {elapsed:.2f}s.")
    print("=" * 75)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MudraForge Statutory Decompiler - Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast 1-epoch end-to-end smoke test on synthetic samples",
    )
    parser.add_argument(
        "--stage",
        choices=["dataset", "supervised", "grpo", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to training dataset file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4000,
        help="Number of synthetic samples to generate (for --stage dataset)",
    )
    parser.add_argument(
        "--sup-ckpt",
        type=str,
        default=str(DEFAULT_SUP_CKPT),
        help="Path for supervised checkpoint",
    )
    parser.add_argument(
        "--rl-ckpt",
        type=str,
        default=str(DEFAULT_RL_CKPT),
        help="Path for GRPO RL aligned checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (defaults: 15 for supervised, 5 for grpo)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override",
    )

    args = parser.parse_args()

    if args.smoke_test:
        return run_smoke_test()

    if args.stage == "dataset":
        create_comprehensive_dataset(output_path=args.dataset, num_samples=args.samples)
        return 0

    sup_ckpt = args.sup_ckpt
    if args.stage in ("supervised", "all"):
        sup_kwargs = {
            "dataset_path": args.dataset,
            "checkpoint_out": args.sup_ckpt,
        }
        if args.epochs is not None and args.stage == "supervised":
            sup_kwargs["epochs"] = args.epochs
        if args.lr is not None and args.stage == "supervised":
            sup_kwargs["lr"] = args.lr
        sup_ckpt = train_supervised_dual_stream(**sup_kwargs)

    if args.stage in ("grpo", "all"):
        grpo_kwargs = {
            "checkpoint_in": sup_ckpt,
            "checkpoint_out": args.rl_ckpt,
            "dataset_path": args.dataset,
        }
        if args.epochs is not None and args.stage in ("grpo", "all"):
            grpo_kwargs["epochs"] = args.epochs
        if args.lr is not None and args.stage in ("grpo", "all"):
            grpo_kwargs["lr"] = args.lr
        train_grpo_logic_only(**grpo_kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
