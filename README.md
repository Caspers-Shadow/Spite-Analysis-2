# Spite Analysis – Card Game & RL Platform

A fully playable implementation of **Spite Analysis** with:
- Human vs AI gameplay (PyQt6 GUI)
- AI vs AI self-play
- Reinforcement learning training infrastructure (DQN)
- Training statistics & matplotlib charts

---

## Quick Start

### 1. Install dependencies

```bash
pip install PyQt6 matplotlib numpy
# Optional: pip install torch  (for DQN / RL agent)
```

### 2. Run the GUI

```bash
python main.py
```

### 3. Headless training (no GUI)

```bash
python main.py --headless-train --episodes 500
```

---

## Project Structure

```
spite_analysis/
├── game/
│   ├── card.py          Card class (rank, suit, wild logic)
│   ├── deck.py          Double deck with reshuffle pool
│   ├── player.py        Hand, stockpile, discard piles
│   ├── game_state.py    Core game loop, actions, RL environment
│   └── rules.py         Stateless rule validation helpers
│
├── ai/
│   ├── agent.py         Abstract base class
│   ├── random_agent.py  Uniform-random baseline
│   ├── heuristic_agent.py  Priority-based rule agent
│   ├── rl_agent.py      DQN agent (PyTorch)
│   └── training.py      Self-play training loop + stats/plots
│
├── gui/
│   ├── main_window.py   App window, screens, navigation
│   ├── board_view.py    Interactive game board
│   ├── card_widget.py   QPainter card rendering with wild glow
│   ├── statistics_panel.py  Side panel: game info + AI metrics
│   └── controls.py      Reusable UI helpers
│
├── utils/
│   ├── config.py        Colours, sizes, timing constants
│   └── logger.py        Logging helper
│
├── main.py              Entry point
└── requirements.txt
```

---

## Game Rules Summary

| Element | Rule |
|---|---|
| Stockpile | 13 cards face-down; top card visible; emptying it wins |
| Hand | 5 cards drawn at start of each turn |
| Discard piles | 4 personal piles; only top card is playable |
| Building piles | 4 shared piles; A → Q sequence; cleared on Queen |
| Red King (wild) | Represents **Ace through Queen** (1-12) |
| Black King (wild) | Represents **2 through Queen** (2-12); cannot be Ace |
| Turn end | Must discard exactly one hand card → turn passes |

---

## AI Agents

| Agent | Description |
|---|---|
| `RandomAgent` | Uniform random legal action; ~40 % chance to stop playing early |
| `HeuristicAgent` | Scored priority: stockpile plays > combo setups > wild conservation |
| `RLAgent` | DQN with replay buffer & target network (requires PyTorch) |

---

## RL Environment Interface

```python
from game.game_state import GameState

state = GameState(seed=42)
obs   = state.get_observation()     # dict of observable features
acts  = state.get_valid_actions()   # list of Action objects
ok, winner = state.apply_action(acts[0])
done  = state.is_terminal()
reward = state.get_reward(player_id=0)
```

---

## Training Example

```python
from ai.heuristic_agent import HeuristicAgent
from ai.random_agent import RandomAgent
from ai.training import TrainingSession

session = TrainingSession(
    HeuristicAgent(0),
    RandomAgent(1),
    n_episodes=1000,
)
session.run()
session.save_stats("stats.json")
session.plot_stats(save_path="plot.png")
```

---

## Keyboard / UI Controls

| Action | How |
|---|---|
| Select a hand card | Click the card |
| Play to building pile | Click a building pile after selecting |
| Discard & end turn | Select hand card → click a discard pile |
| End turn (auto-discard) | Click "End Turn" in the side panel |
| Change AI speed | Drag the speed slider in the toolbar |

---

## Extending the Project

- **New agent**: subclass `ai/agent.py::Agent`, implement `choose_action(state)`
- **New reward**: modify `game_state.py::get_reward()` or `training.py::_shaped_reward()`
- **New screen**: add a `QWidget` subclass and register it in `MainWindow._stack`
- **MCTS**: use `state.copy()` for tree rollouts (full deep-copy supported)
