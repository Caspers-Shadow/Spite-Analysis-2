# ♠ Spite Analysis

> A fully digitised, AI-powered implementation of the local South African card game **Spite Analysis** — complete with a polished GUI, four AI difficulty levels, reinforcement learning training, and a live statistics dashboard.

---

## What is Spite Analysis?

Spite Analysis is a two-player turn-based card game played widely in South African homes. It is similar to the international game Skip-Bo but carries distinct local rules, including a unique **unlock mechanic** and a **hand-refill rule** not found in the original.

**Objective:** Be the first player to completely empty your personal Stockpile of 13 cards by building sequences from Ace → Queen on shared Building Piles in the centre.

---

## Features

| Feature | Description |
|---|---|
| 🎮 **Human vs AI** | Play against Heuristic, MCTS, DQN, or Random AI |
| 🤖 **AI vs AI** | Watch any two agents play each other in real time |
| 🎓 **Training Mode** | Train the DQN agent via self-play with live progress charts |
| 📊 **Statistics Dashboard** | Persistent win-rate trends, game length graphs, matchup comparisons |
| 📖 **Rules & Help** | Full in-app guide covering all rules, UI controls, and AI descriptions |
| 🃏 **Custom Card Renderer** | QPainter-drawn cards with wild-card glow effects |
| 🔒 **No Draws** | The game always produces a winner — no stalls or freezes |

---

## Quick Start

### 1. Prerequisites

- **Python 3.10 or newer**
- **Windows / macOS / Linux**

### 2. Clone the repository

```bash
git clone https://github.com/your-username/spite-analysis.git
cd spite-analysis
```

### 3. Create and activate a virtual environment *(recommended)*

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install PyQt6 matplotlib numpy
```

> **Optional — DQN training only:**
> ```bash
> pip install torch
> ```
> Without PyTorch the DQN agent behaves like a random agent. All other features work without it.

### 5. Run the game

```bash
python main.py
```

---

## Windows Long-Path Fix *(if PyQt6 install fails)*

If you see `[Errno 2] No such file or directory` during `pip install PyQt6`, Windows' default 260-character path limit is too short for some PyQt6 filenames. Fix it once with:

```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Restart your terminal and retry `pip install PyQt6`.

---

## How to Play

### Main Menu

| Button | What it does |
|---|---|
| 🧑 Human vs AI | Play against an AI opponent of your choice |
| 🤖 AI vs AI | Watch two AI agents play — adjust speed with the slider |
| 🎓 Start Training | Train the DQN agent and view live charts |
| 📊 View Statistics | Browse win-rate graphs for all recorded matchups |
| 📖 Rules & Help | Full game rules, UI guide, and AI descriptions |

---

### Playing the Game

#### Selecting and playing a card

1. **Click a card** in your hand, on your stockpile, or on a discard pile top — it highlights in yellow.
2. **Click a Building Pile** to play it there. Legal targets glow green.
3. Click the same card again to deselect it.

#### Ending your turn

Every turn must end with a discard. Two ways:

- **Select a hand card → click one of your four Discard Piles** (they glow green when a hand card is selected).
- **Click the ⏭ End Turn button** at the bottom of your play area — it auto-picks a card to discard.

#### AI Speed

Use the **AI Speed slider** in the top toolbar to control how fast the AI thinks — from Very Slow (great for watching strategy) to Instant.

---

## Game Rules

### Setup
- Two standard 52-card decks shuffled together (104 cards total).
- Each player receives **13 face-down Stockpile cards**; the top card is flipped face-up.
- Each player draws **5 cards** into their hand.
- Four empty **Building Pile slots** sit in the centre of the table.

### Turn Structure

1. **Draw** — draw until your hand holds 5 cards.
2. **Play** — play any number of legal cards from your hand, stockpile top, or discard pile tops onto Building Piles.
3. **Discard** — place exactly one hand card onto one of your four personal Discard Piles. **This ends your turn.**

> **Hand Refill Rule:** If you play all 5 hand cards during your turn, you immediately draw 5 fresh cards and your turn **continues** — you do not discard yet.

### Building Piles
- Sequences run **Ace → 2 → 3 → ... → Queen**.
- Any player may play on any pile.
- When a pile reaches Queen it is **cleared**; the slot can start again from Ace.
- Up to **4 simultaneous piles** in the centre.

### 🔓 The Unlock Rule

You may only play **non-Ace** cards onto Building Piles after you are **unlocked**.

You unlock when:

- **(a)** You personally play an Ace to start a Building Pile, **OR**
- **(b)** Your opponent has cumulatively started all 4 pile slots (placed 4+ lifetime Aces).

Playing an Ace is **always permitted** — it is how you unlock yourself.  
Your current unlock status is shown in the side panel during play.

### Wild Cards

| Card | Represents |
|---|---|
| 🔴 Red King | Any value from **Ace (1) to Queen (12)** |
| ⚫ Black King | Any value from **2 to Queen (12)** — cannot act as Ace |

Once played, a wild card's value is fixed permanently for that pile. Wild cards display a coloured glow stripe on their left edge.

### Winning

The first player to empty their **Stockpile** wins instantly.  
If the deck runs out and neither player can act, the player with the **smaller remaining Stockpile** wins.  
**Draws are impossible** — the game always produces a winner.

---

## AI Agents

| Agent | How it works | Strength | Training required? |
|---|---|---|---|
| **Random** | Picks a random legal action each step | ⭐ Baseline | No |
| **Heuristic** | Scores actions using hand-crafted priority rules | ⭐⭐⭐ Strong | No |
| **MCTS-15** | Runs 15 random game simulations per candidate action | ⭐⭐⭐⭐ Very strong | No |
| **MCTS-30** | Runs 30 simulations — stronger, slightly slower | ⭐⭐⭐⭐⭐ Strongest | No |
| **RL (DQN)** | Deep Q-Network trained via self-play reinforcement learning | ⭐ early → ⭐⭐⭐⭐ trained | Yes (10k+ episodes) |

### Recommended matchups

| Goal | Matchup |
|---|---|
| Learning the game | Human vs Heuristic |
| A real challenge | Human vs MCTS-15 |
| Watching strong AI | Heuristic vs MCTS-15 (AI vs AI mode) |
| Training DQN | DQN (P0) vs Heuristic (P1) — 5,000+ episodes |
| Generating stats fast | Heuristic vs Random — runs in seconds |

> **DQN note:** Without PyTorch, DQN behaves randomly. With PyTorch, allow at least 5,000 episodes before it shows meaningful improvement over random play. The win-rate chart in the training screen will show the learning curve.

---

## Training the DQN Agent

1. Click **🎓 Start Training** from the main menu.
2. Set **Player 0** to `RL (DQN)` — this is the **Learner** (🎓 badge shown).
3. Set **Player 1** to `Heuristic` — this is the **Training Opponent** (📋 badge shown).
4. Set **Episodes** to at least `1000`. Use `5000+` for meaningful DQN improvement.
5. Click **▶ Start Training** and watch the progress bar.

After training completes:
- Charts appear automatically showing win rate trend, game length, and final results.
- Results are pushed into **📊 View Statistics** — no file loading needed.
- Click **💾 Save JSON** to export a detailed results file.

> **MCTS training tip:** MCTS agents are slow per game. Use `MCTS-5` for training benchmarks or use `Heuristic` as the DQN's opponent for much faster training runs.

---

## Headless Training *(no GUI)*

Train from the command line without opening the window:

```bash
# 500 episodes, Heuristic vs Random
python main.py --headless-train --episodes 500

# With a fixed random seed for reproducibility
python main.py --headless-train --episodes 1000 --seed 42
```

Output files generated automatically:
- `training_stats.json` — full episode-by-episode results
- `training_plot.png` — win rate and game length charts

---

## Statistics Dashboard

Open **📊 View Statistics** from the main menu to explore all recorded matchup data.

| Tab | What it shows |
|---|---|
| **Win Rate Trend** | Rolling 20-game P0 win rate with green/red shading per matchup |
| **Game Length** | Histogram of turns per game, split by winner |
| **All Matchups** | Side-by-side win % and games played across every recorded combination |

All data is stored automatically in `game_stats.json`. It loads on startup, so your history survives between sessions. Click **🔄 Refresh** to pick up any new results, or **🗑 Clear All** to reset.

---

## Project Structure

```
spite-analysis/
│
├── game/
│   ├── card.py              Card class — rank, suit, wild-card logic
│   ├── deck.py              Double 104-card deck with reshuffle pool
│   ├── player.py            Hand, stockpile, four discard piles
│   ├── game_state.py        Core engine + RL environment interface
│   └── rules.py             Stateless validation helpers
│
├── ai/
│   ├── agent.py             Abstract base class for all agents
│   ├── random_agent.py      Uniform-random baseline
│   ├── heuristic_agent.py   Priority-scored rule-based agent
│   ├── mcts_agent.py        Monte Carlo rollout agent (configurable N)
│   ├── rl_agent.py          DQN agent with replay buffer + target network
│   └── training.py          Training loop, reward shaping, stats, plots
│
├── gui/
│   ├── main_window.py       App window — 5 screens, background AI threads
│   ├── board_view.py        Interactive game board with card selection
│   ├── card_widget.py       QPainter card renderer + wild-card glow
│   ├── statistics_panel.py  Side panel — live session stats + mini chart
│   ├── stats_view.py        Full statistics dashboard (3 chart tabs)
│   └── rules_screen.py      In-app rules & help (3 tabs)
│
├── utils/
│   ├── stats_manager.py     Persistent matchup statistics store
│   ├── config.py            Colours, sizes, delays, agent lists
│   └── logger.py            Structured console logging
│
├── main.py                  Entry point — GUI or --headless-train
├── requirements.txt         Python dependencies
└── README.md                You are here
```

---

## Extending the Project

### Adding a new AI agent

Create `ai/my_agent.py`:

```python
from ai.agent import Agent
from game.game_state import GameState, Action
from typing import Optional

class MyAgent(Agent):
    def __init__(self, player_id: int):
        super().__init__(player_id, name="MyAgent")

    def choose_action(self, state: GameState) -> Optional[Action]:
        actions = state.get_valid_actions()
        if not actions:
            return None
        # Your decision logic here
        return actions[0]
```

Then in `gui/main_window.py`:
1. Add `"MyAgent"` to `AGENT_PLAY_OPTIONS` and/or `AGENT_TRAIN_OPTIONS`.
2. Add a case to the `_make_agent()` factory function.

### Using the RL environment in your own code

```python
from game.game_state import GameState

state = GameState(seed=42)

while not state.is_terminal():
    obs     = state.get_observation()    # 32-feature dict
    actions = state.get_valid_actions()  # list of legal Action objects
    action  = actions[0]                 # replace with your policy

    ok, winner = state.apply_action(action)
    reward  = state.get_reward(player_id=0)  # +1 win, -1 loss, 0 ongoing

print(f"Winner: Player {state.winner}")
```

### Running a benchmark programmatically

```python
from ai.training import TrainingSession
from ai.heuristic_agent import HeuristicAgent
from ai.random_agent import RandomAgent

session = TrainingSession(HeuristicAgent(0), RandomAgent(1), n_episodes=200)
session.run()

summary = session.get_summary()
print(f"Wins: {summary['wins']}  |  Win %: {summary['win_pct']}")

session.save_stats("my_results.json")
session.plot_stats(save_path="my_plot.png")
```

---

## Requirements

```
PyQt6>=6.4.0
matplotlib>=3.6.0
numpy>=1.23.0
torch>=2.0.0    # optional — DQN training only
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'PyQt6'` | Activate your `.venv` first, then `pip install PyQt6` |
| PyQt6 install fails on Windows | Enable long paths — see [Windows Long-Path Fix](#windows-long-path-fix-if-pyqt6-install-fails) |
| `No module named 'torch'` | DQN requires PyTorch. `pip install torch`, or use Heuristic/MCTS instead |
| App says "not responding" during training | Normal — charts are being computed in background. Wait a few seconds |
| Charts empty in Statistics view | Click **🔄 Refresh** at the top right of the Statistics screen |
| MCTS takes too long per move | Switch to `MCTS-5` or `MCTS-15` instead of `MCTS-30` |
| `RuntimeError: wrapped C/C++ object of type QThread has been deleted` | Replace `gui/main_window.py` with the latest version |
| Console shows many "returned None" warnings | Replace `game/game_state.py` with the latest version |

---

## Acknowledgements

- Game rules based on the South African oral tradition of **Spite Analysis**, as played in local homes and social settings.
- AI architecture informed by: Mitchell (1997) *Machine Learning*; Mnih et al. (2015) *Human-level control through deep reinforcement learning*; Sutton & Barto (2018) *Reinforcement Learning: An Introduction*.
- Built with [PyQt6](https://pypi.org/project/PyQt6/), [matplotlib](https://matplotlib.org/), and optionally [PyTorch](https://pytorch.org/).

---

## Licence

This project is for educational and research purposes as part of the ITRI 616 course at the North-West University.

---

*Made with ♠ in South Africa.*
