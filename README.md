Spite Analysis — minimal Python environment

This repository contains a small Python package implementing the rules of the
Spite Analysis card game (for RL simulation and testing).

Files added:
- `spite/` — package containing `cards.py` and `game.py`
- `tests/` — pytest test to run a short smoke test

How to run tests (Windows PowerShell):

```powershell
python -m pip install -U pip && pip install pytest
pytest -q
```

The implementation is intentionally minimal and focused on core turn flow,
legal actions, wild card handling, and stockpile win condition. It's ready
to be extended with observation wrappers, richer policies, and RL training.
