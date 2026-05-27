import json
import time
from typing import Dict, Any
from ..game.game_state import GameState
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent


def run_selfplay(num_games: int = 100, seed: int = None, agents=None, headless: bool = True, callback=None) -> Dict[str, Any]:
    stats = {"games":0, "wins":[0,0], "avg_turns":[]}
    for g in range(num_games):
        env = GameState(seed=(seed+g) if seed is not None else None)
        state = env.reset()
        if agents is None:
            agents = [RandomAgent(seed), RandomAgent(seed+1 if seed else None)]
        while not env.is_terminal():
            cp = env.current_player
            action = agents[cp].act(env, cp)
            env.step(action)
            if callback:
                callback(env)
        stats["games"] += 1
        stats["wins"][env.winner] += 1
        stats["avg_turns"].append(env.turn_count)
    stats["win_rate"] = [stats["wins"][0]/stats["games"], stats["wins"][1]/stats["games"]]
    stats["mean_turns"] = sum(stats["avg_turns"])/len(stats["avg_turns"]) if stats["avg_turns"] else 0
    # write out
    fname = f"training_stats_{int(time.time())}.json"
    with open(fname, 'w') as f:
        json.dump(stats, f, indent=2)
    return stats
