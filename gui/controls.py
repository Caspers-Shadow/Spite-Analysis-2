from gui.qt_compat import QtWidgets


class ControlsPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.human_vs_ai = QtWidgets.QPushButton("Human vs AI")
        self.ai_vs_ai = QtWidgets.QPushButton("AI vs AI")
    self.start_training = QtWidgets.QPushButton("Start Training")
        self.view_stats = QtWidgets.QPushButton("View Statistics")
    self.exit_button = QtWidgets.QPushButton("Exit")
    self.status_label = QtWidgets.QLabel("")
    layout.addWidget(self.human_vs_ai)
        layout.addWidget(self.ai_vs_ai)
    layout.addWidget(self.start_training)
    layout.addWidget(self.view_stats)
    layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.exit_button)

        self.exit_button.clicked.connect(lambda: QtWidgets.QApplication.quit())
