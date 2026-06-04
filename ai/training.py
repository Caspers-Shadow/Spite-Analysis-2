"""
Training infrastructure – self-play loops, statistics, plots.

Key guarantee: run_game() ALWAYS returns a winner (never None).
If max_turns is reached, the player with the smaller stockpile wins.
"""

import json, os, time
from typing import List, Optional, Callable, Dict, Any, Tuple

from game.game_state import GameState, Action
from ai.agent import Agent
from utils.logger import get_logger

log = get_logger(__name__)


def run_game(
    agent0: Agent,
    agent1: Agent,
    seed: Optional[int] = None,
    max_turns: int = 3000,
    render_callback: Optional[Callable] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Run one complete game.  Returns a dict that always includes a
    non-None 'winner' key (0 or 1) — draws are impossible by design.
    """
    state = GameState(seed=seed, player_names=(agent0.name, agent1.name))
    agents = {0: agent0, 1: agent1}
    agent0.on_game_start(state)
    agent1.on_game_start(state)

    t0 = time.time()

    for _ in range(max_turns):
        if stop_callback and stop_callback():
            break

        if state.is_terminal():
            break

        pid   = state.current_player_idx
        agent = agents[pid]
        action = agent.choose_action(state)

        if action is None:
            # Fallback: force a discard so the game never stalls
            discards = state.get_discard_actions()
            if discards:
                action = discards[0]
            else:
                log.warning(f"Agent {agent.name} returned None and no discard available")
                break

        prev_stocks = [p.stockpile_size for p in state.players]
        state.apply_action(action)

        if render_callback:
            render_callback(state, action)

        if stop_callback and stop_callback():
            break

        if hasattr(agent, 'record_transition'):
            reward = _shaped_reward(state, pid, prev_stocks)
            agent.record_transition(state, reward, state.is_terminal())

    # ── Guarantee a winner – never return None ────────────────────────────────
    if not state.is_terminal():
        s0, s1 = state.players[0].stockpile_size, state.players[1].stockpile_size
        state.winner   = 0 if s0 <= s1 else 1
        state.game_over = True

    winner_id = state.winner   # guaranteed non-None

    agent0.on_game_end(state, winner_id)
    agent1.on_game_end(state, winner_id)

    return {
        'winner':               winner_id,
        'turns':                state.turn_number,
        'duration_s':           round(time.time() - t0, 4),
        'completed_sequences':  state.completed_sequences,
        'total_plays':          state.stats.get('total_plays', 0),
        'total_discards':       state.stats.get('total_discards', 0),
        'hand_refills':         state.hand_refills,
        'stockpile_remaining':  [p.stockpile_size for p in state.players],
    }


def _shaped_reward(state, player_id, prev_stocks):
    if state.is_terminal():
        return 1.0 if state.winner == player_id else -1.0
    delta = prev_stocks[player_id] - state.players[player_id].stockpile_size
    return delta * 0.2


class TrainingSession:
    def __init__(self, agent0, agent1, n_episodes=1000,
                 checkpoint_dir='checkpoints', checkpoint_freq=500,
                 seed=None):
        self.agent0 = agent0
        self.agent1 = agent1
        self.n_episodes = n_episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.base_seed = seed
        self.results: List[Dict[str, Any]] = []
        self.wins = [0, 0]
        self.episode = 0
        self.last_checkpoint_paths: List[str] = []
        self._stop_flag = False
        os.makedirs(checkpoint_dir, exist_ok=True)

    def run(self, callback=None):
        self._stop_flag = False
        for ep in range(self.n_episodes):
            if self._stop_flag:
                break
            seed = (self.base_seed + ep) if self.base_seed is not None else None
            result = run_game(
                self.agent0,
                self.agent1,
                seed=seed,
                stop_callback=lambda: self._stop_flag,
            )
            self.results.append(result)
            self.episode = ep + 1
            # winner is always 0 or 1, never None
            self.wins[result['winner']] += 1
            if callback:
                callback(self)
            if self.episode % self.checkpoint_freq == 0:
                self._save_checkpoints()
        self._save_checkpoints()

    def stop(self):
        self._stop_flag = True

    def get_summary(self):
        if not self.results:
            return {}
        total = len(self.results)
        wins_pct = [round(w / total * 100, 1) for w in self.wins]
        recent = self.results[-50:]
        recent_wins = [sum(1 for r in recent if r['winner'] == i) for i in range(2)]
        return {
            'episodes':       self.episode,
            'wins':           self.wins,
            'win_pct':        wins_pct,
            'recent_50_wins': recent_wins,
            'avg_turns':      round(sum(r['turns'] for r in self.results) / total, 1),
            'avg_duration':   round(sum(r['duration_s'] for r in self.results) / total, 4),
        }

    def get_win_rate_history(self, window=50):
        xs, ys = [], []
        for i in range(window, len(self.results)+1, max(1, window//5)):
            chunk = self.results[max(0, i-window):i]
            xs.append(i)
            ys.append(sum(1 for r in chunk if r['winner'] == 0) / len(chunk))
        return xs, ys

    def save_stats(self, path='training_stats.json'):
        data = {'summary': self.get_summary(), 'results': self.results[-500:]}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        log.info(f"Stats saved to {path}")

    def plot_stats(self, save_path=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not available")
            return
        if not self.results:
            return

        xs, ys = self.get_win_rate_history()
        turns  = [r['turns'] for r in self.results]
        smooth = [sum(turns[max(0,i-20):i+1])/min(i+1,20) for i in range(len(turns))]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Spite Analysis Training', fontsize=14, fontweight='bold')

        axes[0,0].plot(xs, ys, color='royalblue', linewidth=2)
        axes[0,0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[0,0].set_title(f'{self.agent0.name} Win Rate (rolling 50)')
        axes[0,0].set_ylim(0,1); axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(range(len(smooth)), smooth, color='green', linewidth=1.5)
        axes[0,1].set_title('Average Game Length'); axes[0,1].grid(True, alpha=0.3)

        if hasattr(self.agent0,'losses') and self.agent0.losses:
            axes[1,0].plot(self.agent0.losses, color='orange', alpha=0.7, linewidth=1)
            axes[1,0].set_title(f'{self.agent0.name} Loss'); axes[1,0].set_yscale('log')
        else:
            axes[1,0].bar([self.agent0.name, self.agent1.name],
                          self.wins, color=['royalblue','salmon'])
            axes[1,0].set_title('Total Wins')

        if hasattr(self.agent0,'episode_rewards') and self.agent0.episode_rewards:
            axes[1,1].plot(self.agent0.episode_rewards, color='purple', alpha=0.7)
            axes[1,1].set_title(f'{self.agent0.name} Episode Rewards')
        else:
            labels = [self.agent0.name, self.agent1.name]
            axes[1,1].bar(labels, self.wins, color=['royalblue','salmon'])
            axes[1,1].set_title('Wins Summary')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        else:
            plt.show(block=False)
        plt.close(fig)

    def _save_checkpoints(self):
        self.last_checkpoint_paths = []
        for i, agent in enumerate([self.agent0, self.agent1]):
            if hasattr(agent, 'save'):
                numbered_path = os.path.join(self.checkpoint_dir, f'agent{i}_ep{self.episode}.pt')
                agent.save(numbered_path)
                self.last_checkpoint_paths.append(numbered_path)

                try:
                    from ai.rl_agent import default_model_path
                    latest_path = default_model_path(i, self.checkpoint_dir)
                except Exception:
                    latest_path = os.path.join(self.checkpoint_dir, f'agent{i}_latest.pt')
                agent.save(latest_path)
                self.last_checkpoint_paths.append(latest_path)
