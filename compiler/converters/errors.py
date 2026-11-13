"""Shared exceptions for AST -> target-language lowering."""


class CatalaLoweringError(ValueError):
    """Raised when an AST construct has no faithful Catala translation."""
