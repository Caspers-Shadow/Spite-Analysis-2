"""
StatsManager – persistent per-matchup game statistics.

Stores every game result indexed by matchup key (e.g. "Heuristic vs Random").
Saves automatically to a JSON file so history survives across sessions.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any


MATCHUP_TYPES = [
    "Human vs Random",
    "Human vs Heuristic",
    "Human vs RL (DQN)",
    "Heuristic vs Random",
    "Heuristic vs Heuristic",
    "Heuristic vs RL (DQN)",
    "Random vs RL (DQN)",
    "RL (DQN) vs RL (DQN)",
]

_DEFAULT_PATH = "game_stats.json"


class StatsManager:
    """
    Singleton-friendly stats store.  Create one instance in main_window.py
    and pass it around.

    Data shape
    ----------
    self.data = {
        "Heuristic vs Random": [
            {"winner": 0, "turns": 43, "ts": 1700000000.0},
            ...
        ],
        ...
    }
    """

    def __init__(self, save_path: str = _DEFAULT_PATH):
        self.save_path = save_path
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ── Write ──────────────────────────────────────────────────────────────────
    def record(self, p0_label: str, p1_label: str,
               winner: Optional[int], turns: int):
        """
        Record a completed game.

        p0_label / p1_label – display names like "Heuristic", "Random",
                               "RL (DQN)", or "Human".
        winner              – 0 or 1 (player index), None for draw/timeout.
        """
        key = f"{p0_label} vs {p1_label}"
        if key not in self.data:
            self.data[key] = []
        self.data[key].append({
            "winner": winner,
            "turns":  turns,
            "ts":     time.time(),
        })
        self._save()

    # ── Read ───────────────────────────────────────────────────────────────────
    def matchups(self) -> List[str]:
        """Return all matchup keys that have at least one game recorded."""
        return [k for k, v in self.data.items() if v]

    def games(self, key: str) -> List[Dict[str, Any]]:
        return self.data.get(key, [])

    def summary(self, key: str) -> Dict[str, Any]:
        gs = self.games(key)
        if not gs:
            return {"games": 0}
        total   = len(gs)
        wins0   = sum(1 for g in gs if g["winner"] == 0)
        wins1   = sum(1 for g in gs if g["winner"] == 1)
        draws   = total - wins0 - wins1
        avg_t   = sum(g["turns"] for g in gs) / total
        return {
            "games":   total,
            "wins_p0": wins0,
            "wins_p1": wins1,
            "draws":   draws,
            "win_pct_p0": round(wins0 / total * 100, 1),
            "win_pct_p1": round(wins1 / total * 100, 1),
            "avg_turns":  round(avg_t, 1),
        }

    def rolling_win_rate(self, key: str,
                         window: int = 20) -> Tuple[List[int], List[float]]:
        """Rolling win rate for player 0 over games in this matchup."""
        gs = self.games(key)
        if len(gs) < 2:
            return [], []
        xs, ys = [], []
        for i in range(1, len(gs) + 1):
            chunk = gs[max(0, i - window): i]
            wins  = sum(1 for g in chunk if g["winner"] == 0)
            xs.append(i)
            ys.append(wins / len(chunk))
        return xs, ys

    def all_summaries(self) -> Dict[str, Dict[str, Any]]:
        return {k: self.summary(k) for k in self.matchups()}

    # ── Persistence ────────────────────────────────────────────────────────────
    def _save(self):
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError:
            pass

    def _load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path) as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def clear(self):
        self.data = {}
        self._save()