from gui.qt_compat import QtWidgets, QtGui, QtCore


class BoardView(QtWidgets.QWidget):
    """Interactive board: displays builds, player hand, discards as buttons."""

    build_clicked = QtCore.pyqtSignal(int)
    hand_clicked = QtCore.pyqtSignal(int)
    discard_clicked = QtCore.pyqtSignal(int)

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.setMinimumSize(600, 400)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Opponent info
        self.opponent_label = QtWidgets.QLabel()
        layout.addWidget(self.opponent_label)
        self.instruction_label = QtWidgets.QLabel('Select a hand card, then click a build to play or a discard to discard.')
        layout.addWidget(self.instruction_label)

        # Builds grid
        self.builds_layout = QtWidgets.QHBoxLayout()
        self.build_buttons = []
        for i in range(4):
            b = QtWidgets.QPushButton(f"Build {i}\n(empty)")
            b.setFixedSize(140, 80)
            b.clicked.connect(lambda _, idx=i: self.build_clicked.emit(idx))
            self.build_buttons.append(b)
            self.builds_layout.addWidget(b)
        layout.addLayout(self.builds_layout)

        # Player hand
        self.hand_layout = QtWidgets.QHBoxLayout()
        self.hand_buttons = []
        for i in range(5):
            hb = QtWidgets.QPushButton("")
            hb.setFixedSize(100, 140)
            hb.clicked.connect(lambda _, idx=i: self.hand_clicked.emit(idx))
            self.hand_buttons.append(hb)
            self.hand_layout.addWidget(hb)
        layout.addLayout(self.hand_layout)

        # Discards
        self.discard_layout = QtWidgets.QHBoxLayout()
        self.discard_buttons = []
        for i in range(4):
            db = QtWidgets.QPushButton(f"D{i}\n(empty)")
            db.setFixedSize(100, 80)
            db.clicked.connect(lambda _, idx=i: self.discard_clicked.emit(idx))
            self.discard_buttons.append(db)
            self.discard_layout.addWidget(db)
        layout.addLayout(self.discard_layout)

        self.update_view()

    def update_view(self):
        # Opponent summary
        opp = self.env.players[1]
        self.opponent_label.setText(f"Opponent: stock top {opp.top_stock()} | hand size {len(opp.hand)}")

        # builds
        for i, b in enumerate(self.env.builds):
            txt = ",".join([f"{c[0].rank}{c[0].suit}:{v}" for c, v in b]) if b else "(empty)"
            self.build_buttons[i].setText(f"Build {i}\n{txt}")
            self.build_buttons[i].setStyleSheet("")

        # highlight legal builds when a hand is selected
        sel = getattr(self, 'selected_hand_index', None)
        if sel is not None and sel < len(self.env.players[0].hand):
            card = self.env.players[0].hand[sel]
            for i in range(4):
                # check legality by introspecting rules via build content
                # a simple approximation: find required value
                top = self.env.builds[i][-1][1] if self.env.builds[i] else 0
                req = top + 1
                legal = False
                if card.rank == 'K':
                    legal = (card.is_red_king() and 1 <= req <= 12) or (card.is_black_king() and 2 <= req <= 12)
                else:
                    rv = card.rank_value()
                    legal = (rv == req)
                if legal:
                    self.build_buttons[i].setStyleSheet('background: #2a9d8f; color: white;')

        # hand
        hs = self.env.players[0].hand
        for i in range(5):
            if i < len(hs):
                self.hand_buttons[i].setText(str(hs[i]))
                self.hand_buttons[i].setEnabled(True)
                # highlight selected
                if getattr(self, 'selected_hand_index', None) == i:
                    self.hand_buttons[i].setStyleSheet('border: 2px solid yellow;')
                else:
                    self.hand_buttons[i].setStyleSheet('')
            else:
                self.hand_buttons[i].setText("")
                self.hand_buttons[i].setEnabled(False)

        # discards
        for i, pile in enumerate(self.env.players[0].discards):
            if pile:
                self.discard_buttons[i].setText(f"D{i}\n{pile[-1]}")
            else:
                self.discard_buttons[i].setText(f"D{i}\n(empty)")

