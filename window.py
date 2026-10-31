"""The VanishCam window: live preview with overlays, and a settings panel."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import (
    QColor, QDesktopServices, QFont, QFontMetricsF, QIcon, QImage, QKeySequence, QPainter,
    QPainterPath, QPen, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractButton, QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

try:
    from .engine import APP_DIR, CALIBRATION_SECONDS, Engine, Status, camera_devices
    from .settings import MODE_INFO, MODES, RESOLUTIONS
except ImportError:
    from engine import APP_DIR, CALIBRATION_SECONDS, Engine, Status, camera_devices
    from settings import MODE_INFO, MODES, RESOLUTIONS


BACKGROUND = "#111113"
PANEL = "#18181b"
CARD = "#212125"
CARD_HOVER = "#29292e"
BORDER = "#2c2c31"
TEXT = "#f4f4f5"
MUTED = "#a1a1aa"
FAINT = "#71717a"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"
OBS_URL = "https://obsproject.com/download"

PHASE_LABELS = {
    "setup": "Set up",
    "calibrating": "Learning background",
    "ready": "Ready",
    "vanishing": "Vanishing",
    "vanished": "Vanished",
    "appearing": "Coming back",
}


def font(pixels: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    value = QFont()
    value.setPixelSize(max(8, int(round(pixels))))
    value.setWeight(weight)
    return value


def color(hex_value: str, alpha: int = 255) -> QColor:
    value = QColor(hex_value)
    value.setAlpha(alpha)
    return value


# ------------------------------------------------------------ engine thread

class EngineThread(QThread):
    """Runs the engine loop; the window always reads only the newest frame."""

    updated = Signal()

    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.lock = threading.Lock()
        self.latest: tuple[QImage | None, Status] | None = None
        self.toasts: list[str] = []
        self.pending = False
        self.display_size = (1280, 720)  # Physical pixels available for the preview.
        self.device_ratio = 1.0

    def run(self) -> None:
        engine = self.engine
        try:
            engine.open()
        except Exception as error:
            engine.error = str(error)
        while not self.isInterruptionRequested():
            try:
                output, status = engine.next_frame()
            except Exception as error:  # Keep the window alive and show what happened.
                engine.error = f"Something went wrong: {error}"
                output, status = None, engine.snapshot()
            image = self._to_image(output) if output is not None else None
            with self.lock:
                if image is None and self.latest is not None and not status.error:
                    image = self.latest[0]  # Keep the last picture while waiting.
                self.latest = (image, status)
                self.toasts.extend(status.toasts)
                notify = not self.pending
                self.pending = True
            if notify:
                self.updated.emit()
        engine.close()

    def _to_image(self, frame: np.ndarray) -> QImage:
        # Downscale here with area averaging: sharper than scaling while painting.
        width, height = self.display_size
        frame_height, frame_width = frame.shape[:2]
        scale = min(width / frame_width, height / frame_height)
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(frame_width * scale)),
                                       max(1, int(frame_height * scale))),
                               interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                       QImage.Format.Format_BGR888).copy()
        image.setDevicePixelRatio(self.device_ratio)
        return image

    def take(self) -> tuple[tuple[QImage | None, Status] | None, list[str]]:
        with self.lock:
            self.pending = False
            toasts, self.toasts = self.toasts, []
            return self.latest, toasts


# --------------------------------------------------------------- the preview

class VideoView(QWidget):
    start_requested = Signal()
    retry_requested = Signal()

    MARGIN = 18

    def __init__(self):
        super().__init__()
        self.image: QImage | None = None
        self.status: Status | None = None
        self.show_hud = True
        self.toast_text = ""
        self.toast_until = 0.0
        self.buttons: dict[str, QRectF] = {}
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(560, 360)

    def available(self) -> QRectF:
        margin = self.MARGIN
        return QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)

    def toast(self, text: str) -> None:
        self.toast_text = text
        self.toast_until = time.perf_counter() + 1.8

    def set_frame(self, image: QImage | None, status: Status) -> None:
        self.image = image
        self.status = status
        self.update()

    def video_rect(self) -> QRectF:
        area = self.available()
        if self.image is not None:
            ratio = self.image.width() / max(1, self.image.height())
        else:
            ratio = 16 / 9
        width = min(area.width(), area.height() * ratio)
        height = width / ratio
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)

    # -- painting

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing
                               | QPainter.RenderHint.SmoothPixmapTransform
                               | QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), QColor(BACKGROUND))
        target = self.video_rect()
        clip = QPainterPath()
        clip.addRoundedRect(target, 16, 16)
        painter.save()
        painter.setClipPath(clip)
        if self.image is not None:
            painter.drawImage(target, self.image)
        else:
            painter.fillRect(target, QColor("#0a0a0b"))
        painter.restore()
        self.buttons = {}
        self.scale = max(0.75, min(1.35, target.height() / 640))
        status = self.status
        if status is None:
            self._center_message(painter, target, "Starting VanishCam...", "")
            return
        if status.error:
            self._error(painter, target, status.error)
            return
        if self.image is None:
            self._center_message(painter, target, "Starting the camera...", status.waiting)
            return
        if status.phase == "setup":
            self._setup_card(painter, target)
        elif status.phase == "calibrating":
            self._countdown(painter, target, status)
        if self.show_hud:
            self._mode_pill(painter, target, status)
            self._virtual_pill(painter, target, status)
            if status.phase in ("ready", "vanished"):
                hint = self._hint(painter, target, status)
                if status.live is not None:
                    self._live_view(painter, target, status, hint)
            if status.debug:
                painter.setFont(font(12 * self.scale))
                painter.setPen(QColor(GREEN))
                painter.drawText(QPointF(target.left() + 16, target.bottom() - 14), status.debug)
        self._toast(painter, target)

    def s(self, value: float) -> float:
        return value * self.scale

    def _panel(self, painter: QPainter, rect: QRectF, alpha: int = 165, radius: float | None = None,
               fill: str = "#000000") -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color(fill, alpha))
        radius = rect.height() / 2 if radius is None else radius
        painter.drawRoundedRect(rect, radius, radius)

    def _text_width(self, text: str, value: QFont) -> float:
        return QFontMetricsF(value).horizontalAdvance(text)

    def _key_button(self, painter: QPainter, center: QPointF, key: str, label: str,
                    accent: str, name: str) -> None:
        """A clickable pill: [Key] Label."""
        label_font = font(self.s(16), QFont.Weight.DemiBold)
        key_font = font(self.s(13), QFont.Weight.DemiBold)
        key_width = self._text_width(key, key_font) + self.s(16)
        width = key_width + self._text_width(label, label_font) + self.s(40)
        height = self.s(46)
        rect = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
        self._panel(painter, rect, 255, fill=accent)
        key_rect = QRectF(rect.left() + self.s(9), rect.top() + self.s(9), key_width,
                          height - self.s(18))
        self._panel(painter, key_rect, 60, self.s(6), "#000000")
        painter.setPen(QColor("#111111"))
        painter.setFont(key_font)
        painter.drawText(key_rect, Qt.AlignmentFlag.AlignCenter, key)
        painter.setFont(label_font)
        painter.drawText(rect.adjusted(key_width + self.s(18), 0, 0, 0),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        self.buttons[name] = rect

    def _center_message(self, painter, target, title, body) -> None:
        painter.setPen(QColor(TEXT))
        painter.setFont(font(self.s(20), QFont.Weight.DemiBold))
        painter.drawText(target.adjusted(0, -self.s(30), 0, -self.s(30)),
                         Qt.AlignmentFlag.AlignCenter, title)
        if body:
            painter.setPen(QColor(MUTED))
            painter.setFont(font(self.s(14)))
            painter.drawText(target.adjusted(self.s(80), self.s(30), -self.s(80), self.s(30)),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, body)

    def _card(self, painter, target, width, height) -> QRectF:
        card = QRectF(target.center().x() - width / 2, target.center().y() - height / 2,
                      width, height)
        self._panel(painter, card, 205, self.s(20), "#0d0d0f")
        return card

    def _setup_card(self, painter: QPainter, target: QRectF) -> None:
        accent = MODE_INFO[self.status.mode].accent
        card = self._card(painter, target, min(target.width() - 40, self.s(520)), self.s(220))
        painter.setPen(QColor(TEXT))
        painter.setFont(font(self.s(24), QFont.Weight.DemiBold))
        painter.drawText(QRectF(card.left(), card.top() + self.s(26), card.width(), self.s(34)),
                         Qt.AlignmentFlag.AlignCenter, "Capture your background")
        painter.setPen(QColor(MUTED))
        painter.setFont(font(self.s(15)))
        painter.drawText(
            QRectF(card.left() + self.s(36), card.top() + self.s(68), card.width() - self.s(72),
                   self.s(60)),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            f"When the countdown starts, step out of the frame for {CALIBRATION_SECONDS:g} "
            "seconds so VanishCam can learn what is behind you.")
        self._key_button(painter, QPointF(card.center().x(), card.bottom() - self.s(50)),
                         "Space", "Start countdown", accent, "start")

    def _countdown(self, painter: QPainter, target: QRectF, status: Status) -> None:
        painter.fillRect(target, color("#000000", 120))
        accent = MODE_INFO[status.mode].accent
        center = target.center()
        painter.setPen(QColor(TEXT))
        painter.setFont(font(self.s(30), QFont.Weight.DemiBold))
        painter.drawText(QRectF(target.left(), center.y() - self.s(130), target.width(), self.s(44)),
                         Qt.AlignmentFlag.AlignCenter, "Step out of the frame")
        painter.setPen(QColor(accent))
        painter.setFont(font(self.s(120), QFont.Weight.Bold))
        number = str(max(1, int(np.ceil(status.countdown or 0))))
        painter.drawText(QRectF(target.left(), center.y() - self.s(80), target.width(), self.s(150)),
                         Qt.AlignmentFlag.AlignCenter, number)
        painter.setPen(QColor(MUTED))
        painter.setFont(font(self.s(15)))
        painter.drawText(QRectF(target.left(), center.y() + self.s(76), target.width(), self.s(30)),
                         Qt.AlignmentFlag.AlignCenter, "Learning your background...")

    def _mode_pill(self, painter: QPainter, target: QRectF, status: Status) -> None:
        info = MODE_INFO[status.mode]
        title_font = font(self.s(14), QFont.Weight.DemiBold)
        phase_font = font(self.s(14))
        phase = PHASE_LABELS.get(status.phase, status.phase)
        width = (self._text_width(info.title, title_font) + self._text_width(phase, phase_font)
                 + self.s(58))
        rect = QRectF(target.left() + self.s(16), target.top() + self.s(16), width, self.s(34))
        self._panel(painter, rect)
        painter.setBrush(QColor(info.accent))
        painter.drawEllipse(QPointF(rect.left() + self.s(17), rect.center().y()), self.s(4.5), self.s(4.5))
        painter.setPen(QColor(TEXT))
        painter.setFont(title_font)
        title_rect = rect.adjusted(self.s(30), 0, 0, 0)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter, info.title)
        painter.setPen(QColor(MUTED))
        painter.setFont(phase_font)
        painter.drawText(title_rect.adjusted(self._text_width(info.title, title_font) + self.s(12), 0, 0, 0),
                         Qt.AlignmentFlag.AlignVCenter, phase)

    def _virtual_pill(self, painter: QPainter, target: QRectF, status: Status) -> None:
        if not status.virtual_device:
            return
        text = f"Live in {status.virtual_device}"
        value = font(self.s(13))
        width = self._text_width(text, value) + self.s(44)
        rect = QRectF(target.right() - self.s(16) - width, target.top() + self.s(16), width, self.s(34))
        self._panel(painter, rect)
        painter.setBrush(QColor(GREEN))
        painter.drawEllipse(QPointF(rect.left() + self.s(17), rect.center().y()), self.s(4.5), self.s(4.5))
        painter.setPen(QColor(TEXT))
        painter.setFont(value)
        painter.drawText(rect.adjusted(self.s(30), 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, text)

    def _hint(self, painter: QPainter, target: QRectF, status: Status) -> QRectF:
        info = MODE_INFO[status.mode]
        if status.blocked:
            # After an effect the sign only counts again once you have lowered it.
            text = "Lower your hand, then show the sign again"
        elif status.phase == "ready":
            text = info.hint
        else:
            text = "Do the same sign to come back, or press V"
        value = font(self.s(16), QFont.Weight.Medium)
        width = self._text_width(text, value) + self.s(48)
        height = self.s(44) + (self.s(12) if status.hold > 0 else 0)
        rect = QRectF(target.center().x() - width / 2, target.bottom() - self.s(20) - height,
                      width, height)
        self._panel(painter, rect, 170, self.s(22))
        painter.setPen(QColor(TEXT))
        painter.setFont(value)
        painter.drawText(QRectF(rect.left(), rect.top(), rect.width(), self.s(44)),
                         Qt.AlignmentFlag.AlignCenter, text)
        if status.hold > 0:
            bar = QRectF(rect.left() + self.s(24), rect.top() + self.s(42), rect.width() - self.s(48),
                         self.s(5))
            self._panel(painter, bar, 255, None, "#3f3f46")
            done = QRectF(bar.left(), bar.top(), bar.width() * status.hold, bar.height())
            self._panel(painter, done, 255, None, info.accent)
        return rect

    def _live_view(self, painter: QPainter, target: QRectF, status: Status, hint: QRectF) -> None:
        """You, small, in a corner while you are invisible, so you can aim the sign.

        Only the preview draws it; the virtual camera never gets it. The frame
        lights up in the effect's color when the sign is recognized.
        """
        live = np.ascontiguousarray(status.live)
        image = QImage(live.data, live.shape[1], live.shape[0], live.strides[0],
                       QImage.Format.Format_BGR888)
        width = min(target.width() * 0.24, self.s(300))
        height = width * live.shape[0] / live.shape[1]
        rect = QRectF(target.right() - self.s(16) - width, hint.top() - self.s(12) - height,
                      width, height)
        radius = self.s(12)
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.save()
        painter.setClipPath(clip)
        painter.drawImage(rect, image)
        painter.restore()
        if status.sign and not status.blocked:
            edge = QColor(MODE_INFO[status.mode].accent)
        elif status.blocked:
            edge = QColor(RED)  # Distinct from every mode's accent.
        else:
            edge = color("#ffffff", 90)
        painter.setPen(QPen(edge, self.s(2.5) if status.sign or status.blocked else self.s(1.5)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, radius, radius)
        label_font = font(self.s(12), QFont.Weight.Medium)
        text = "Only you see this"
        label = QRectF(rect.left() + self.s(8), rect.top() + self.s(8),
                       self._text_width(text, label_font) + self.s(18), self.s(24))
        self._panel(painter, label, 170)
        painter.setPen(QColor(TEXT))
        painter.setFont(label_font)
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _toast(self, painter: QPainter, target: QRectF) -> None:
        left = self.toast_until - time.perf_counter()
        if left <= 0 or not self.toast_text:
            return
        value = font(self.s(15), QFont.Weight.Medium)
        width = self._text_width(self.toast_text, value) + self.s(40)
        rect = QRectF(target.center().x() - width / 2, target.top() + self.s(66), width, self.s(38))
        painter.setOpacity(min(1.0, left / 0.3))
        self._panel(painter, rect, 200, None, "#0d0d0f")
        painter.setPen(QColor(TEXT))
        painter.setFont(value)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.toast_text)
        painter.setOpacity(1.0)

    def _error(self, painter: QPainter, target: QRectF, message: str) -> None:
        card = self._card(painter, target, min(target.width() - 40, self.s(560)), self.s(240))
        painter.setPen(QColor(TEXT))
        painter.setFont(font(self.s(22), QFont.Weight.DemiBold))
        painter.drawText(QRectF(card.left(), card.top() + self.s(26), card.width(), self.s(32)),
                         Qt.AlignmentFlag.AlignCenter, "The camera is not available")
        painter.setPen(QColor(MUTED))
        painter.setFont(font(self.s(14)))
        painter.drawText(QRectF(card.left() + self.s(32), card.top() + self.s(66),
                                card.width() - self.s(64), self.s(100)),
                         Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, message)
        self._key_button(painter, QPointF(card.center().x(), card.bottom() - self.s(44)),
                         "R", "Try again", "#e4e4e7", "retry")

    def mouseMoveEvent(self, event) -> None:
        over = any(rect.contains(event.position()) for rect in self.buttons.values())
        self.setCursor(Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:
        for name, rect in self.buttons.items():
            if rect.contains(event.position()):
                (self.start_requested if name == "start" else self.retry_requested).emit()


# -------------------------------------------------------------- panel widgets

class Switch(QAbstractButton):
    def __init__(self, checked: bool = False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setFixedSize(40, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.accent = "#ffffff"

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.accent if self.isChecked() else "#3f3f46"))
        painter.drawRoundedRect(QRectF(0, 0, 40, 24), 12, 12)
        painter.setBrush(QColor("#111113" if self.isChecked() else "#e4e4e7"))
        x = 28 if self.isChecked() else 12
        painter.drawEllipse(QPointF(x, 12), 8.5, 8.5)


class KeyBadge(QLabel):
    def __init__(self, key: str):
        super().__init__(key)
        self.setObjectName("key")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(f"Shortcut: {key}")


class ModeCard(QAbstractButton):
    def __init__(self, number: int, mode: str):
        super().__init__()
        self.number = number
        self.info = MODE_INFO[mode]
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(80)
        self.setToolTip(f"{self.info.description}  Shortcut: {number}")

    def sizeHint(self) -> QSize:
        return QSize(300, 80)

    def enterEvent(self, event) -> None:
        self.update()

    def leaveEvent(self, event) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        accent = self.info.accent
        checked = self.isChecked()
        painter.setPen(QPen(QColor(accent), 1.5) if checked else Qt.PenStyle.NoPen)
        painter.setBrush(color(accent, 26) if checked else
                         QColor(CARD_HOVER if self.underMouse() else CARD))
        painter.drawRoundedRect(rect, 12, 12)
        badge = QRectF(14, 14, 24, 24)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(accent) if checked else QColor("#36363c"))
        painter.drawRoundedRect(badge, 7, 7)
        painter.setPen(QColor("#111113" if checked else TEXT))
        painter.setFont(font(13, QFont.Weight.DemiBold))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(self.number))
        left = 50
        width = rect.width() - left - 12
        painter.setPen(QColor(TEXT))
        painter.setFont(font(15, QFont.Weight.DemiBold))
        painter.drawText(QRectF(left, 12, width, 22), Qt.AlignmentFlag.AlignVCenter, self.info.title)
        painter.setPen(QColor(accent))
        painter.setFont(font(12.5, QFont.Weight.Medium))
        painter.drawText(QRectF(left, 34, width, 18), Qt.AlignmentFlag.AlignVCenter, self.info.gesture)
        painter.setPen(QColor(MUTED))
        description_font = font(12)
        painter.setFont(description_font)
        text = QFontMetricsF(description_font).elidedText(
            self.info.description, Qt.TextElideMode.ElideRight, width)
        painter.drawText(QRectF(left, 53, width, 18), Qt.AlignmentFlag.AlignVCenter, text)


class Combo(QComboBox):
    """Dropdown with a drawn chevron; refreshes its items right before opening."""

    about_to_open = Signal()

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)

    def showPopup(self) -> None:
        self.about_to_open.emit()
        super().showPopup()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(MUTED), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        x, y = self.width() - 18, self.height() / 2
        painter.drawLine(QPointF(x - 4, y - 2), QPointF(x, y + 2))
        painter.drawLine(QPointF(x, y + 2), QPointF(x + 4, y - 2))


STYLE = f"""
QMainWindow, QWidget#central {{ background: {BACKGROUND}; }}
QScrollArea#panel {{ background: {PANEL}; border: none; border-left: 1px solid {BORDER}; }}
QWidget#panelBody {{ background: {PANEL}; }}
QLabel {{ color: {TEXT}; font-size: 13px; background: transparent; }}
QLabel#title {{ font-size: 22px; font-weight: 600; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#section {{ color: {FAINT}; font-size: 11px; font-weight: 600; letter-spacing: 1px;
                  padding-top: 14px; }}
QLabel#note {{ color: {MUTED}; font-size: 12px; }}
QLabel#key {{ color: #d4d4d8; background: #26262b; border: 1px solid #3a3a40; border-radius: 5px;
              font-size: 11px; font-weight: 600; padding: 1px 6px; min-width: 10px; }}
QComboBox {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 9px;
             padding: 0 30px 0 12px; font-size: 13px; }}
QComboBox:hover {{ border-color: #3f3f46; background: {CARD_HOVER}; }}
QComboBox::drop-down {{ border: none; width: 0px; }}
QComboBox::down-arrow {{ image: none; width: 0px; height: 0px; }}
QComboBox QAbstractItemView {{ background: #222226; color: {TEXT}; border: 1px solid {BORDER};
    selection-background-color: #3a3a41; selection-color: {TEXT}; outline: none; padding: 4px; }}
QPushButton#primary {{ border: none; border-radius: 11px; min-height: 42px; }}
QPushButton#primary:disabled {{ background: #2a2a2f; }}
QSlider {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: #34343a; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
                              background: #f4f4f5; }}
QScrollBar:vertical {{ width: 8px; background: transparent; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: #3a3a40; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: #222226; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
"""


# ------------------------------------------------------------------ window

class MainWindow(QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.settings = engine.settings
        self.thread = EngineThread(engine)
        self.devices = []
        self.last_phase = ""
        self.virtual_note_state = None
        self.setWindowTitle("VanishCam")
        self.setWindowIcon(QIcon(str(APP_DIR / "assets" / "vanish_icon.ico")))
        self.setStyleSheet(STYLE)
        self.resize(1440, 860)
        self.setMinimumSize(1000, 620)

        central = QWidget(objectName="central")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.video = VideoView()
        self.video.start_requested.connect(self.start_calibration)
        self.video.retry_requested.connect(lambda: self.engine.post("retry"))
        layout.addWidget(self.video, 1)
        self.panel = self._build_panel()
        layout.addWidget(self.panel)
        self.setCentralWidget(central)
        self.panel.setVisible(self.settings.panel_visible)
        self._shortcuts()
        self.apply_accent()

        self.thread.updated.connect(self.on_update)
        self.thread.start()

    # -- panel

    def _section(self, layout: QVBoxLayout, text: str) -> None:
        label = QLabel(text.upper(), objectName="section")
        layout.addWidget(label)

    def _row(self, layout: QVBoxLayout, text: str, control: QWidget, key: str | None = None,
             stacked: bool = False) -> None:
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(QLabel(text))
        header.addStretch(1)
        if key:
            header.addWidget(KeyBadge(key))
        if stacked:
            layout.addLayout(header)
            layout.addWidget(control)
        else:
            header.addWidget(control)
            layout.addLayout(header)

    def _build_panel(self) -> QScrollArea:
        panel = QScrollArea(objectName="panel")
        panel.setWidgetResizable(True)
        panel.setFixedWidth(348)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(objectName="panelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(22, 22, 22, 24)
        layout.setSpacing(10)

        layout.addWidget(QLabel("VanishCam", objectName="title"))
        layout.addWidget(QLabel("Vanish from your webcam with a hand sign.", objectName="subtitle"))

        self._section(layout, "Effect")
        self.mode_group = QButtonGroup(self)
        self.mode_cards = {}
        for number, mode in enumerate(MODES, start=1):
            card = ModeCard(number, mode)
            card.setChecked(mode == self.settings.mode)
            card.clicked.connect(lambda checked=False, mode=mode: self.select_mode(mode))
            self.mode_group.addButton(card)
            self.mode_cards[mode] = card
            layout.addWidget(card)

        self._section(layout, "Background")
        self.capture_button = QPushButton(objectName="primary")
        self.capture_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.capture_button.setCursor(Qt.CursorShape.PointingHandCursor)
        button_layout = QHBoxLayout(self.capture_button)
        button_layout.setContentsMargins(16, 0, 12, 0)
        self.capture_label = QLabel("Capture background")
        self.capture_label.setStyleSheet("color: #111113; font-size: 14px; font-weight: 600;")
        button_layout.addWidget(self.capture_label)
        button_layout.addStretch(1)
        space = QLabel("Space")
        space.setStyleSheet("color: #111113; background: rgba(0,0,0,0.12); border-radius: 5px;"
                            "font-size: 11px; font-weight: 600; padding: 2px 7px;")
        button_layout.addWidget(space)
        for child in (self.capture_label, space):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.capture_button.clicked.connect(self.start_calibration)
        layout.addWidget(self.capture_button)
        self.background_note = QLabel(objectName="note")
        self.background_note.setWordWrap(True)
        layout.addWidget(self.background_note)

        self._section(layout, "Camera")
        self.camera_combo = Combo()
        self.camera_combo.about_to_open.connect(self.refresh_cameras)
        self.camera_combo.activated.connect(self.on_camera_chosen)
        self._row(layout, "Camera", self.camera_combo, "C", stacked=True)
        self.resolution_combo = Combo()
        for width, height in RESOLUTIONS:
            self.resolution_combo.addItem(f"{height}p   ({width} x {height})", (width, height))
        index = self.resolution_combo.findData((self.settings.width, self.settings.height))
        self.resolution_combo.setCurrentIndex(max(0, index))
        self.resolution_combo.activated.connect(self.on_resolution_chosen)
        self._row(layout, "Resolution", self.resolution_combo, stacked=True)
        self.mirror_switch = Switch(self.settings.mirror)
        self.mirror_switch.toggled.connect(lambda on: self.engine.post("mirror", on))
        self._row(layout, "Mirror preview", self.mirror_switch, "M")

        self._section(layout, "Output")
        self.virtual_switch = Switch(self.settings.virtual_camera)
        self.virtual_switch.toggled.connect(lambda on: self.engine.post("virtual", on))
        self._row(layout, "Virtual camera", self.virtual_switch, "O")
        self.virtual_note = QLabel(objectName="note")
        self.virtual_note.setWordWrap(True)
        self.virtual_note.setOpenExternalLinks(False)
        self.virtual_note.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
        layout.addWidget(self.virtual_note)
        self.sound_switch = Switch(self.settings.sound)
        self.sound_switch.toggled.connect(lambda on: self.engine.post("sound", on))
        self._row(layout, "Sound", self.sound_switch, "S")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.settings.volume * 100))
        self.volume_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume_slider.valueChanged.connect(lambda value: self.engine.post("volume", value / 100))
        self._row(layout, "Volume", self.volume_slider, stacked=True)
        self.reappear_switch = Switch(self.settings.gesture_reappear)
        self.reappear_switch.toggled.connect(self.on_reappear)
        self._row(layout, "Same sign brings you back", self.reappear_switch, "G")
        self.hold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_slider.setRange(5, 20)
        self.hold_slider.setValue(int(round(self.settings.goku_hold_seconds * 10)))
        self.hold_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hold_value = QLabel(objectName="note")
        self.hold_slider.valueChanged.connect(self.on_hold_changed)
        self._row(layout, "Goku hold time", self.hold_value, stacked=False)
        layout.addWidget(self.hold_slider)
        self.on_hold_changed(self.hold_slider.value(), post=False)

        self._section(layout, "Shortcuts")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        shortcuts = (("V", "Vanish / come back"), ("Space", "Capture background"),
                     ("1-4", "Switch effect"), ("H", "Hide hints on the preview"),
                     ("Tab", "Hide this panel"), ("F", "Full screen"), ("Q", "Quit"))
        for row, (key, text) in enumerate(shortcuts):
            badge = KeyBadge(key)
            grid.addWidget(badge, row, 0, Qt.AlignmentFlag.AlignLeft)
            label = QLabel(text, objectName="note")
            grid.addWidget(label, row, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        panel.setWidget(body)
        self.refresh_cameras()
        return panel

    def _shortcuts(self) -> None:
        bindings = {
            "Space": self.start_calibration,
            "V": lambda: self.engine.post("toggle_vanish"),
            "R": self.retry_or_return,
            "C": self.next_camera,
            "M": self.mirror_switch.toggle,
            "O": self.virtual_switch.toggle,
            "S": self.sound_switch.toggle,
            "G": self.reappear_switch.toggle,
            "H": self.toggle_hud,
            "Tab": self.toggle_panel,
            "F": self.toggle_fullscreen,
            "Q": self.close,
            "Esc": self.escape,
        }
        for number, mode in enumerate(MODES, start=1):
            bindings[str(number)] = lambda mode=mode: self.select_mode(mode)
        self.shortcut_objects = []
        for key, action in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(action)
            self.shortcut_objects.append(shortcut)

    # -- actions

    def select_mode(self, mode: str) -> None:
        self.mode_cards[mode].setChecked(True)
        if mode != self.settings.mode:
            self.engine.post("mode", mode)
            self.settings.mode = mode
            self.apply_accent()

    def apply_accent(self) -> None:
        accent = MODE_INFO[self.settings.mode].accent
        self.capture_button.setStyleSheet(
            f"QPushButton#primary {{ background: {accent}; }}"
            f"QPushButton#primary:hover {{ background: {QColor(accent).lighter(108).name()}; }}"
            "QPushButton#primary:disabled { background: #2a2a2f; }")
        for slider in (self.volume_slider, self.hold_slider):
            slider.setStyleSheet(f"QSlider::sub-page:horizontal {{ background: {accent};"
                                 "border-radius: 2px; }")
        for switch in (self.mirror_switch, self.virtual_switch, self.sound_switch,
                       self.reappear_switch):
            switch.accent = accent
            switch.update()

    def start_calibration(self) -> None:
        if self.last_phase != "calibrating":
            self.engine.post("calibrate")

    def retry_or_return(self) -> None:
        if self.engine.error:
            self.engine.post("retry")
        elif self.last_phase == "vanished":
            self.engine.post("toggle_vanish")

    def refresh_cameras(self) -> None:
        self.devices = camera_devices()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for device in self.devices:
            self.camera_combo.addItem(device.name, device.index)
        if not self.devices:
            self.camera_combo.addItem("No camera found", -1)
        index = self.camera_combo.findData(self.settings.camera)
        if index < 0 and self.devices:
            self.camera_combo.addItem(f"Camera {self.settings.camera} (not connected)",
                                      self.settings.camera)
            index = self.camera_combo.count() - 1
        self.camera_combo.setCurrentIndex(max(0, index))
        self.camera_combo.blockSignals(False)

    def on_camera_chosen(self, row: int) -> None:
        index = self.camera_combo.itemData(row)
        if index is not None and index >= 0 and index != self.settings.camera:
            self.settings.camera = index
            self.engine.post("camera", index)
            self.video.toast(self.camera_combo.itemText(row))

    def next_camera(self) -> None:
        self.refresh_cameras()
        if len(self.devices) < 2:
            self.video.toast("No other camera found")
            return
        row = (self.camera_combo.currentIndex() + 1) % len(self.devices)
        self.camera_combo.setCurrentIndex(row)
        self.on_camera_chosen(row)

    def on_resolution_chosen(self, row: int) -> None:
        width, height = self.resolution_combo.itemData(row)
        if (width, height) != (self.settings.width, self.settings.height):
            self.engine.post("resolution", width, height)

    def on_reappear(self, on: bool) -> None:
        self.settings.gesture_reappear = on

    def on_hold_changed(self, value: int, post: bool = True) -> None:
        seconds = value / 10
        self.hold_value.setText(f"{seconds:.1f} s")
        if post:
            self.engine.post("goku_hold", seconds)

    def toggle_hud(self) -> None:
        self.settings.hud = self.video.show_hud = not self.video.show_hud
        self.video.toast("Hints on" if self.video.show_hud else "Hints hidden")

    def toggle_panel(self) -> None:
        self.settings.panel_visible = not self.panel.isVisible()
        self.panel.setVisible(self.settings.panel_visible)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def escape(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.close()

    # -- updates from the engine

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display_size()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_display_size()

    def _update_display_size(self) -> None:
        ratio = self.devicePixelRatioF()
        area = self.video.available()
        self.thread.device_ratio = ratio
        self.thread.display_size = (max(1, int(area.width() * ratio)),
                                    max(1, int(area.height() * ratio)))

    def on_update(self) -> None:
        latest, toasts = self.thread.take()
        if latest is None:
            return
        image, status = latest
        for text in toasts:
            self.video.toast(text)
        self.video.show_hud = self.settings.hud
        self.video.set_frame(image, status)
        if status.phase != self.last_phase:
            self.last_phase = status.phase
            self._update_background_note(status.phase)
        note = (self.settings.virtual_camera, status.virtual_device, status.virtual_error)
        if note != self.virtual_note_state:
            self.virtual_note_state = note
            self._update_virtual_note(*note)

    def _update_background_note(self, phase: str) -> None:
        capturing = phase == "calibrating"
        self.capture_button.setEnabled(not capturing)
        self.capture_label.setText("Capturing..." if capturing else
                                   "Capture background" if phase == "setup" else
                                   "Recapture background")
        self.background_note.setText(
            "Not captured yet. Start the countdown, then step out of the frame."
            if phase == "setup" else
            f"Stay out of the frame for {CALIBRATION_SECONDS:g} seconds." if capturing else
            "Captured. Recapture if the camera moves or the light changes.")

    def _update_virtual_note(self, enabled: bool, device: str | None, error: str) -> None:
        if not enabled:
            text = "Off. Only this window shows the effect."
        elif device:
            text = (f'<span style="color:{GREEN}">●</span> Sending to {device}. '
                    "Pick it as your camera in Zoom, Meet or Teams.")
        elif error:
            text = (f'<span style="color:{AMBER}">●</span> No virtual camera found. '
                    f'<a style="color:{TEXT}" href="{OBS_URL}">Install OBS Studio</a>, click '
                    "Start Virtual Camera once, then Stop it and come back.")
        else:
            text = "Starts with the camera."
        self.virtual_note.setText(text)

    def closeEvent(self, event) -> None:
        self.thread.requestInterruption()
        self.thread.wait(5000)
        super().closeEvent(event)
