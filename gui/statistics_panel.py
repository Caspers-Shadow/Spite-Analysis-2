from .qt_compat import QtWidgets


class StatisticsPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("Statistics")
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.label)
        layout.addWidget(self.text)

    def update_stats(self, stats: dict):
        self.text.setText(str(stats))

    def update_from_env(self, env):
        s = env.get_state()
        out = f"Turn: {s['turn_count']} Current: P{s['current_player']}\n"
        out += f"Builds: {s['builds']}\n"
        out += f"Stock sizes: {s['stock_sizes']}\n"
        out += f"Discard tops: {s['discard_tops']}\n"
        self.text.setText(out)
