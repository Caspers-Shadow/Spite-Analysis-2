from gui.qt_compat import QtWidgets, QtCore, QtGui, binding_name
from game.game_state import GameState
from ai.random_agent import RandomAgent
from ai.heuristic_agent import HeuristicAgent
from gui.board_view import BoardView
from gui.controls import ControlsPanel
from gui.statistics_panel import StatisticsPanel
from utils.logger import get_logger

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
    self.controls.end_turn.clicked.connect(self.human_end_turn)

        # timer for AI thinking/auto-play
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.ai_step)
        self.ai_speed_ms = 500

    # selection state for human interactions
    self.selected_hand_index = None

    # connect board signals
    self.board.hand_clicked.connect(self.on_hand_clicked)
    self.board.build_clicked.connect(self.on_build_clicked)
    self.board.discard_clicked.connect(self.on_discard_clicked)

    # update controls status
    self.update_status()

    def start_human_vs_ai(self):
        log.info("Starting Human vs AI")
        self.env.reset()
        self.board.repaint()
        self.agent0 = None  # human
        self.agent1 = HeuristicAgent()
        # If AI starts, start timer
        if self.env.current_player == 1:
            self.timer.start(self.ai_speed_ms)
        self.board.update_view()
    self.update_status()

    def start_ai_vs_ai(self):
        log.info("Starting AI vs AI")
        self.env.reset()
        self.board.repaint()
        self.agent0 = HeuristicAgent()
        self.agent1 = RandomAgent()
        self.timer.start(self.ai_speed_ms)
    self.board.update_view()

    def start_training(self):
        log.info("Start training clicked — hook into training pipeline")
        # For now just run a small headless batch and show stats
        from ..ai.training import run_selfplay
        stats = run_selfplay(num_games=50, seed=1)
        self.stats.update_stats(stats)
        self.board.update_view()
    self.update_status()

    # --- human interaction handlers ---
    def on_hand_clicked(self, index: int):
        # select or deselect a hand card
        if index >= len(self.env.players[0].hand):
            return
        if self.selected_hand_index == index:
            self.selected_hand_index = None
        else:
            self.selected_hand_index = index
    # visual feedback via button enabled/disabled states
    self.board.selected_hand_index = self.selected_hand_index
    self.board.update_view()
    self.update_status()

    def on_build_clicked(self, build_index: int):
        # if a hand card selected, attempt to play it to build
        if self.selected_hand_index is not None:
            action = {"type":"PLAY_HAND","hand_index":self.selected_hand_index,"build_index":build_index}
            state, done, info = self.env.step(action)
            self.selected_hand_index = None
            self.board.selected_hand_index = None
            self.board.update_view()
            self.stats.update_from_env(self.env)
            self.update_status()
            return
        # otherwise, try to play top of stock to this build
        # only for human (player 0)
        ts = self.env.players[0].top_stock()
        if ts:
            action = {"type":"PLAY_STOCK","build_index":build_index}
            state, done, info = self.env.step(action)
            self.board.update_view()
            self.stats.update_from_env(self.env)
            self.update_status()

    def on_discard_clicked(self, discard_index: int):
        # if a hand card selected, discard it there
        if self.selected_hand_index is not None:
            action = {"type":"DISCARD","hand_index":self.selected_hand_index,"discard_index":discard_index}
            state, done, info = self.env.step(action)
            self.selected_hand_index = None
            self.board.selected_hand_index = None
            self.board.update_view()
            self.stats.update_from_env(self.env)
            self.update_status()
            # After discard, if AI turn, start timer
            if self.env.current_player != 0 and self.agent1 is not None:
                self.timer.start(self.ai_speed_ms)

    def human_end_turn(self):
        # End turn without discarding (not standard but useful)
        action = {"type":"END_TURN"}
        self.env.step(action)
        self.board.update_view()
        self.stats.update_from_env(self.env)
        if self.agent1 is not None:
            self.timer.start(self.ai_speed_ms)
        self.update_status()

    def update_status(self):
        cur = self.env.current_player
        s = f"Current: P{cur}"
        if cur == 0:
            s += " (Human)"
        else:
            s += " (AI)"
        # put into controls
        self.controls.status_label.setText(s)

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
