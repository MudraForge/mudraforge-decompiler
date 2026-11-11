"""
Universal Statutory Virtual Machine (USVM) runtime bridge.
Communicates with @mudraforge/statutory over persistent Node.js stdio IPC.
"""

from .usvm_evaluator import CanonicalUsvmEvaluator

__all__ = ["CanonicalUsvmEvaluator"]
