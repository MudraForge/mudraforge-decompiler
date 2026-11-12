"""
MudraForge Decompiler - Training Engine Module
Includes synthetic dataset generators, USVM RL environment, and dual-stream training pipelines.
"""

from compiler.training.dataset_generator import (
    generate_synthetic_dataset,
    create_comprehensive_dataset,
    tokenize_with_offsets,
)
from compiler.training.usvm_rl_environment import (
    StatutoryRlEnvironment,
)
from compiler.training.train_dual_stream import (
    train_supervised_dual_stream,
    train_grpo_logic_only,
    train_supervised_dual_stream as train_stage1_supervised,
    train_grpo_logic_only as train_stage2_grpo,
)

__all__ = [
    "generate_synthetic_dataset",
    "create_comprehensive_dataset",
    "tokenize_with_offsets",
    "StatutoryRlEnvironment",
    "train_supervised_dual_stream",
    "train_grpo_logic_only",
    "train_stage1_supervised",
    "train_stage2_grpo",
]
