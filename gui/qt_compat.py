"""Qt compatibility layer: prefer PyQt6, fall back to PySide6.

This module exposes QtWidgets, QtCore, QtGui and a helper to check which binding is used.
"""
try:
    import PyQt6.QtWidgets as QtWidgets
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
    BINDING = 'PyQt6'
except Exception:
    try:
        import PySide6.QtWidgets as QtWidgets
        import PySide6.QtCore as QtCore
        import PySide6.QtGui as QtGui
        BINDING = 'PySide6'
    except Exception:
        raise ImportError('Neither PyQt6 nor PySide6 is installed')

def binding_name():
    return BINDING
