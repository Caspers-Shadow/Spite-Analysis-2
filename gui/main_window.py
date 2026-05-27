from .qt_compat import QtWidgets, QtCore, QtGui, binding_name
from ..game.game_state import GameState
from ..ai.random_agent import RandomAgent
from ..ai.heuristic_agent import HeuristicAgent
from .board_view import BoardView
from .controls import ControlsPanel
from .statistics_panel import StatisticsPanel
from ..utils.logger import get_logger

log = get_logger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spite Analysis")
        self.resize(1200, 800)
        self.env = GameState()
        self.agent0 = None
        self.agent1 = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # left: board
        self.board = BoardView(self.env)
        layout.addWidget(self.board, 3)

        # right: controls + stats
        right = QtWidgets.QVBoxLayout()
        self.controls = ControlsPanel(self)
        self.stats = StatisticsPanel()
        right.addWidget(self.controls)
        right.addWidget(self.stats)
        layout.addLayout(right, 1)

        # wire controls
        self.controls.human_vs_ai.clicked.connect(self.start_human_vs_ai)
        self.controls.ai_vs_ai.clicked.connect(self.start_ai_vs_ai)
        self.controls.start_training.clicked.connect(self.start_training)

        # timer for AI thinking/auto-play
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ai_step)
        self.ai_speed_ms = 500

    def start_human_vs_ai(self):
        log.info("Starting Human vs AI")
        self.env.reset()
        self.board.repaint()
        self.agent0 = None  # human
        self.agent1 = HeuristicAgent()
        # If AI starts, start timer
        if self.env.current_player == 1:
            self.timer.start(self.ai_speed_ms)

    def start_ai_vs_ai(self):
        log.info("Starting AI vs AI")
        self.env.reset()
        self.board.repaint()
        self.agent0 = HeuristicAgent()
        self.agent1 = RandomAgent()
        self.timer.start(self.ai_speed_ms)

    def start_training(self):
        log.info("Start training clicked — hook into training pipeline")
        # For now just run a small headless batch and show stats
        from ..ai.training import run_selfplay
        stats = run_selfplay(num_games=50, seed=1)
        self.stats.update_stats(stats)

    def ai_step(self):
        if self.env.is_terminal():
            self.timer.stop()
            return
        cp = self.env.current_player
        agent = self.agent0 if cp == 0 else self.agent1
        if agent is None:
            # human turn — stop timer
            self.timer.stop()
            return
        action = agent.act(self.env, cp)
        self.env.step(action)
        self.board.repaint()
        self.stats.update_from_env(self.env)
