"""
Spite Analysis – Entry Point
============================
Run this file to launch the application:

    python main.py

Optional flags:
    --headless-train   Run a quick headless training session and exit
    --seed N           Set global random seed
"""

import sys
import argparse
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))


def _parse_args():
    parser = argparse.ArgumentParser(description="Spite Analysis Card Game")
    parser.add_argument("--headless-train", action="store_true",
                        help="Run a quick AI self-play training session without GUI")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes for headless training (default: 200)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def run_gui():
    """Launch the full PyQt6 GUI application."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Spite Analysis")
    app.setStyle("Fusion")

    # Dark palette baseline (board handles its own colours)
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#2a2a3e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1e1e2e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e0e0f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#4a4a7a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e0e0f0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#7c6af7"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_headless_training(episodes: int = 200, seed=None):
    """Run a quick AI vs AI training session without any GUI."""
    from ai.heuristic_agent import HeuristicAgent
    from ai.random_agent import RandomAgent
    from ai.rl_agent import RLAgent
    from ai.training import TrainingSession

    print(f"\n{'='*55}")
    print("  Spite Analysis – Headless Training")
    print(f"{'='*55}")
    print(f"  Episodes : {episodes}")
    print(f"  Agent 0  : HeuristicAgent")
    print(f"  Agent 1  : RandomAgent")
    print(f"{'='*55}\n")

    agent0 = HeuristicAgent(0, seed=seed)
    agent1 = RandomAgent(1, seed=seed)

    session = TrainingSession(
        agent0, agent1,
        n_episodes=episodes,
        seed=seed,
        checkpoint_dir="checkpoints",
        checkpoint_freq=max(50, episodes // 4),
    )

    def progress_cb(sess: TrainingSession):
        if sess.episode % max(1, episodes // 10) == 0:
            s = sess.get_summary()
            wins = s.get('wins', [0, 0])
            pct = s.get('win_pct', [0.0, 0.0])
            print(f"  Ep {sess.episode:5d} | "
                  f"Heuristic: {wins[0]} ({pct[0]:.1f}%) | "
                  f"Random: {wins[1]} ({pct[1]:.1f}%) | "
                  f"Avg turns: {s.get('avg_turns', '?')}")

    session.run(callback=progress_cb)

    summary = session.get_summary()
    print(f"\n{'='*55}")
    print("  Training Complete")
    print(f"{'='*55}")
    wins = summary.get('wins', [0, 0])
    pct = summary.get('win_pct', [0.0, 0.0])
    print(f"  Heuristic wins : {wins[0]}  ({pct[0]:.1f}%)")
    print(f"  Random wins    : {wins[1]}  ({pct[1]:.1f}%)")
    print(f"  Avg turns/game : {summary.get('avg_turns', '?')}")
    print(f"{'='*55}\n")

    session.save_stats("training_stats.json")
    print("  Stats saved → training_stats.json")

    try:
        session.plot_stats(save_path="training_plot.png")
        print("  Plot saved  → training_plot.png")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


if __name__ == "__main__":
    args = _parse_args()

    if args.headless_train:
        run_headless_training(episodes=args.episodes, seed=args.seed)
    else:
        run_gui()
