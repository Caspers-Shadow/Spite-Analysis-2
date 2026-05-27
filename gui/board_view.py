from .qt_compat import QtWidgets, QtGui, QtCore


class BoardView(QtWidgets.QWidget):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.setMinimumSize(600,400)

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        qp.fillRect(rect, QtGui.QColor(30,30,30))

        # Draw builds in center
        w = rect.width()
        h = rect.height()
        build_area = QtCore.QRect(50,50,w-100,150)
        qp.setPen(QtGui.QPen(QtGui.QColor('white')))
        qp.drawText(build_area, QtCore.Qt.AlignmentFlag.AlignCenter, "Building Piles")
        y = 100
        x = 100
        for i, b in enumerate(self.env.builds):
            txt = ",".join([f"{c[0].rank}{c[0].suit}:{v}" for c,v in b]) if b else "empty"
            qp.drawRect(x, y, 150, 80)
            qp.drawText(x+10, y+20, f"Build {i}: {txt}")
            x += 170

        # Opponent area
        qp.drawText(10, 20, f"Opponent (P{1}): Stock top {self.env.players[1].top_stock()} Hand size {len(self.env.players[1].hand)}")

        # Player area
        qp.drawText(10, h-60, f"You (P0): Stock top {self.env.players[0].top_stock()} Hand: {','.join([str(c) for c in self.env.players[0].hand])}")

        qp.end()
