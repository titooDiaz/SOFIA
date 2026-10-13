"""
Gemini-style edge glow for Linux.

A soft, multi-color gradient hugs all four edges of the screen, flows around
the border, and gently pulses, like the effect shown when Gemini activates on
a phone. When the program starts, the glow fades in.

Why PyQt5 instead of tkinter?
    tkinter on Linux has no per-pixel transparency, so a "transparent" window
    renders as solid black. Qt supports real translucent windows, as long as a
    compositor is running (GNOME, KDE, Cinnamon, Budgie... have one built in;
    on XFCE enable "Compositing"; on i3/bspwm run `picom`).

Install:
    pip install PyQt5
    (or on Debian/Ubuntu: sudo apt install python3-pyqt5)

Run:
    python3 gemini_glow.py

Stop:
    Ctrl+C in the terminal (the overlay never takes mouse or keyboard input).
"""

import math
import os
import signal
import sys
import time

# ------------------------- SETTINGS -------------------------
GLOW_DEPTH = 120        # how far the glow reaches toward the center (px)
LAYERS = 28             # number of stacked layers that build the soft falloff
LAYER_ALPHA = 0.055     # opacity of each layer (higher = brighter glow)
CORNER_RADIUS = 36      # roundness of the inner edge of the glow (px)
ROTATION_SPEED = 55     # degrees per second the colors flow around the screen
PULSE_SPEED = 2.0       # speed of the breathing effect
PULSE_AMOUNT = 0.18     # strength of the breathing effect (0 = none)
INTRO_SECONDS = 1.4     # fade-in duration when the program starts
FPS = 60
ALL_SCREENS = False     # True = draw on every monitor, False = primary only
DURATION = None         # auto-quit after N seconds (None = run until Ctrl+C)

# On Wayland, always-on-top and click-through are unreliable for normal
# windows, so run through XWayland instead. Set to False if you prefer native.
FORCE_XWAYLAND = True

# On X11, bypass the window manager so the overlay sits at exactly (0, 0),
# above panels, and is never moved or decorated by the WM.
BYPASS_WINDOW_MANAGER = True

# Gemini-like palette: blue -> violet -> pink -> orange (loops back to blue)
COLORS = [
    "#4285F4",  # blue
    "#9B72CB",  # violet
    "#D96570",  # pink / red
    "#F6A56B",  # warm orange
    "#4285F4",  # blue again so the loop is seamless
]
# ------------------------------------------------------------

if FORCE_XWAYLAND and os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QPainter,
                         QPainterPath)
from PyQt5.QtWidgets import QApplication, QWidget


def hex_to_rgb(value):
    """Convert '#rrggbb' to an (r, g, b) tuple."""
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


COLORS_RGB = [hex_to_rgb(c) for c in COLORS]


def ease_out_cubic(x):
    """Smooth deceleration curve used for the fade-in."""
    return 1 - (1 - x) ** 3


class GlowOverlay(QWidget):
    def __init__(self, screen):
        super().__init__()

        flags = (Qt.FramelessWindowHint
                 | Qt.WindowStaysOnTopHint
                 | Qt.Tool                        # keep it out of the taskbar
                 | Qt.WindowTransparentForInput)  # click-through
        if BYPASS_WINDOW_MANAGER and sys.platform.startswith("linux"):
            flags |= Qt.X11BypassWindowManagerHint
        self.setWindowFlags(flags)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # QWidget.setScreen() only exists in Qt6. In Qt5, placing the window
        # with the screen's global geometry puts it on the right monitor.
        self.setGeometry(screen.geometry())

        self.t0 = time.time()
        self.rings = self.build_rings(self.width(), self.height())

        # Gradient stop positions spread evenly over [0, 1]
        n = len(COLORS_RGB)
        self.stop_positions = [i / (n - 1) for i in range(n)]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / FPS))

    # ------------------------------------------------------------
    def build_rings(self, w, h):
        """
        Pre-compute the ring-shaped paths once.
        Each ring is the full screen rectangle minus a rounded inner
        rectangle. Stacking many semi-transparent rings makes the glow
        strongest at the edge and fade smoothly toward the center.
        Returns a list of (path, opacity), outermost-reaching ring first.
        """
        outer = QPainterPath()
        outer.addRect(QRectF(0, 0, w, h))

        rings = []

        # Soft body of the glow: evenly spaced depths, same opacity each
        for i in range(LAYERS, 0, -1):
            depth = GLOW_DEPTH * i / LAYERS
            rings.append((self.make_ring(outer, w, h, depth), LAYER_ALPHA))

        # Thin, brighter rims right on the edge for that "lit border" look
        rings.append((self.make_ring(outer, w, h, 9), 0.22))
        rings.append((self.make_ring(outer, w, h, 3), 0.45))
        return rings

    @staticmethod
    def make_ring(outer, w, h, depth):
        inner = QPainterPath()
        radius = min(CORNER_RADIUS + depth * 0.6, min(w, h) / 2 - depth)
        radius = max(radius, 0)
        inner.addRoundedRect(
            QRectF(depth, depth, w - 2 * depth, h - 2 * depth), radius, radius
        )
        return outer.subtracted(inner)

    # ------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Clear the whole window to fully transparent first
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        t = time.time() - self.t0
        intro = ease_out_cubic(min(1.0, t / INTRO_SECONDS)) if INTRO_SECONDS > 0 else 1.0
        pulse = 1.0 + PULSE_AMOUNT * math.sin(t * PULSE_SPEED)
        angle = (t * ROTATION_SPEED) % 360

        center = QPointF(self.width() / 2, self.height() / 2)

        for path, opacity in self.rings:
            alpha = max(0.0, min(1.0, opacity * intro * pulse))
            gradient = QConicalGradient(center, angle)
            for pos, (r, g, b) in zip(self.stop_positions, COLORS_RGB):
                gradient.setColorAt(pos, QColor(r, g, b, int(255 * alpha)))
            painter.fillPath(path, QBrush(gradient))

        painter.end()


def main():
    # Let Ctrl+C kill the program immediately
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    screens = app.screens() if ALL_SCREENS else [app.primaryScreen()]
    overlays = []
    for screen in screens:
        overlay = GlowOverlay(screen)
        overlay.show()
        overlays.append(overlay)  # keep references so they aren't garbage collected

    if DURATION:
        QTimer.singleShot(int(DURATION * 1000), app.quit)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()