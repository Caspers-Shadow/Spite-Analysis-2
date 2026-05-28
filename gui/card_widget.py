"""
CardWidget – renders a single playing card using QPainter.

Features
--------
- Full rank + suit display
- Red / black colouring
- Wild card glow border (red glow for Red Kings, dark glow for Black Kings)
- Highlighted (selected) state
- Legal-move indicator glow
- Face-down (card back) mode
- Click signal
"""

from typing import Optional

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRect, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics,
    QLinearGradient, QRadialGradient, QPainterPath,
)

from game.card import Card
from utils.config import (
    CARD_W, CARD_H, CARD_RADIUS,
    COLOR_CARD_FACE, COLOR_CARD_BACK,
    COLOR_RED_CARD, COLOR_BLACK_CARD,
    COLOR_HIGHLIGHT, COLOR_LEGAL_GLOW,
    COLOR_WILD_RED_GLOW, COLOR_WILD_BLACK_GLOW,
)


class CardWidget(QWidget):
    """A clickable card widget."""

    clicked = pyqtSignal(object)   # emits the Card (or None for empty pile)

    def __init__(
        self,
        card: Optional[Card] = None,
        face_up: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.card: Optional[Card] = card
        self.face_up: bool = face_up
        self.selected: bool = False
        self.legal_target: bool = False
        self.empty_label: str = ""   # e.g. "A?" for empty building pile

        self.setFixedSize(CARD_W, CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    # ── Setters ────────────────────────────────────────────────────────────────
    def set_card(self, card: Optional[Card], face_up: bool = True):
        self.card = card
        self.face_up = face_up
        self.update()

    def set_selected(self, value: bool):
        self.selected = value
        self.update()

    def set_legal_target(self, value: bool):
        self.legal_target = value
        self.update()

    # ── Events ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.card)
        super().mousePressEvent(event)

    # ── Painting ───────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(2, 2, CARD_W - 4, CARD_H - 4)

        if self.card is None:
            self._paint_empty(p, rect)
        elif not self.face_up:
            self._paint_back(p, rect)
        else:
            self._paint_face(p, rect)

        # Glow overlays
        if self.legal_target:
            self._paint_glow(p, QColor(COLOR_LEGAL_GLOW), 3)
        if self.selected:
            self._paint_glow(p, QColor(COLOR_HIGHLIGHT), 4)

        p.end()

    def _paint_empty(self, p: QPainter, rect: QRectF):
        """Render an empty pile placeholder."""
        p.setPen(QPen(QColor("#446644"), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor("#1a3a1a"))
        p.drawRoundedRect(rect, CARD_RADIUS, CARD_RADIUS)
        if self.empty_label:
            p.setPen(QColor("#557755"))
            font = QFont("Arial", 14, QFont.Weight.Bold)
            p.setFont(font)
            p.drawText(rect.toRect(), Qt.AlignmentFlag.AlignCenter, self.empty_label)

    def _paint_back(self, p: QPainter, rect: QRectF):
        """Render face-down card back."""
        # Shadow
        shadow_rect = rect.adjusted(2, 2, 2, 2)
        p.setBrush(QColor(0, 0, 0, 80))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(shadow_rect, CARD_RADIUS, CARD_RADIUS)

        # Back gradient
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0, QColor("#1a3c8a"))
        grad.setColorAt(0.5, QColor("#2550bb"))
        grad.setColorAt(1, QColor("#122a66"))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#0a1a4a"), 1))
        p.drawRoundedRect(rect, CARD_RADIUS, CARD_RADIUS)

        # Inner pattern
        inner = rect.adjusted(6, 6, -6, -6)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#3060cc"), 1))
        p.drawRoundedRect(inner, CARD_RADIUS - 2, CARD_RADIUS - 2)

        # Center symbol
        p.setPen(QColor("#4470dd"))
        p.setFont(QFont("Arial", 22))
        p.drawText(rect.toRect(), Qt.AlignmentFlag.AlignCenter, "🂠")

    def _paint_face(self, p: QPainter, rect: QRectF):
        """Render a face-up card."""
        card = self.card

        # Shadow
        shadow = rect.adjusted(2, 2, 2, 2)
        p.setBrush(QColor(0, 0, 0, 70))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(shadow, CARD_RADIUS, CARD_RADIUS)

        # Card face
        p.setBrush(QColor(COLOR_CARD_FACE))
        border_color = QColor("#cccccc")
        p.setPen(QPen(border_color, 1))
        p.drawRoundedRect(rect, CARD_RADIUS, CARD_RADIUS)

        # Wild glow border
        if card.is_wild:
            glow_color = QColor(COLOR_WILD_RED_GLOW) if card.is_red_wild else QColor(COLOR_WILD_BLACK_GLOW)
            for w in range(4, 0, -1):
                glow = QColor(glow_color)
                glow.setAlpha(60 + w * 30)
                p.setPen(QPen(glow, w + 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(w, w, -w, -w), CARD_RADIUS, CARD_RADIUS)

        text_color = QColor(COLOR_RED_CARD) if card.is_red else QColor(COLOR_BLACK_CARD)

        # Rank in top-left
        p.setPen(text_color)
        rank_font = QFont("Arial", 13, QFont.Weight.Bold)
        p.setFont(rank_font)
        p.drawText(QRect(5, 3, 28, 20), Qt.AlignmentFlag.AlignLeft, card.rank)

        # Suit in top-left below rank
        suit_font = QFont("Arial", 11)
        p.setFont(suit_font)
        p.drawText(QRect(5, 20, 28, 20), Qt.AlignmentFlag.AlignLeft, card.suit.symbol)

        # Centre large rank + suit
        center_font = QFont("Arial", 22, QFont.Weight.Bold)
        p.setFont(center_font)
        rank_display = card.rank
        if card.is_wild and card.wild_as is not None:
            rank_display = Card.VALUE_TO_RANK.get(card.wild_as, card.rank)

        center_rect = rect.adjusted(0, 20, 0, -20).toRect()
        p.drawText(center_rect, Qt.AlignmentFlag.AlignCenter, rank_display)

        suit_mid_font = QFont("Arial", 14)
        p.setFont(suit_mid_font)
        suit_rect = rect.adjusted(0, 36, 0, 0).toRect()
        p.drawText(suit_rect, Qt.AlignmentFlag.AlignCenter, card.suit.symbol)

        # Rank in bottom-right (rotated 180°)
        p.save()
        p.translate(rect.right() - 2, rect.bottom() - 2)
        p.rotate(180)
        p.setFont(rank_font)
        p.drawText(QRect(0, 0, 28, 20), Qt.AlignmentFlag.AlignLeft, card.rank)
        p.restore()

        # Wild label stripe on left edge
        if card.is_wild:
            stripe_color = QColor(COLOR_WILD_RED_GLOW if card.is_red_wild else "#888888")
            stripe_color.setAlpha(200)
            p.setBrush(stripe_color)
            p.setPen(Qt.PenStyle.NoPen)
            stripe = QRectF(rect.left() + 1, rect.top() + CARD_RADIUS,
                            5, rect.height() - 2 * CARD_RADIUS)
            p.drawRect(stripe)

    def _paint_glow(self, p: QPainter, color: QColor, width: int):
        """Paint a glowing border around the card."""
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(width, 0, -1):
            c = QColor(color)
            c.setAlpha(60 + (width - i) * 40)
            p.setPen(QPen(c, i * 2))
            p.drawRoundedRect(
                QRectF(i, i, CARD_W - 2 * i, CARD_H - 2 * i),
                CARD_RADIUS, CARD_RADIUS
            )
