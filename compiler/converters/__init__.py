"""
MudraForge Decompiler - Target Code Emitters & Transpilers
Provides bridges from Declarative ASTs (OKF v0.2) to external formal legal DSLs (e.g., Catala).
"""

from compiler.converters.catala_emitter import (
    emit_catala_code,
    save_catala_file,
    CatalaEmitter,
)

__all__ = [
    "emit_catala_code",
    "save_catala_file",
    "CatalaEmitter",
]
