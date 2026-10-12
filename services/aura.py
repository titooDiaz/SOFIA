"""
Aura estilo Cortana con tkinter.
- Aparece abajo a la derecha de la pantalla, siempre encima de otras ventanas.
- Se agranda y se mueve más cuanto más fuerte hablas por el micrófono.
- Cierra con clic derecho, tecla Esc, o Ctrl+C en la consola.

Instalar dependencias:
    pip install sounddevice numpy
"""

import math
import sys
import time
import tkinter as tk

import numpy as np
import sounddevice as sd

SIZE = 340             # tamaño de la ventana (px)
MARGIN_X = 20          # separación del borde derecho
MARGIN_Y = 60          # separación del borde inferior (deja espacio a la barra de tareas)
GAIN = 14.0            # sensibilidad del micrófono (súbela si no reacciona mucho)
BASE_RADIUS = 62       # radio del aura en silencio
FPS = 60
TRANSPARENT = "#ffffff"  # color que se vuelve transparente (Windows)



class Aura:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        bg = TRANSPARENT
        print(sys.platform)
        if sys.platform == "win32":
            self.root.attributes("-transparentcolor", TRANSPARENT)
        elif sys.platform == "darwin":
            self.root.attributes("-transparent", True)
            bg = "systemTransparent"

        self.root.config(bg=bg)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = sw - SIZE - MARGIN_X
        y = sh - SIZE - MARGIN_Y
        self.root.geometry(f"{SIZE}x{SIZE}+{x}+{y}")

        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE,
                                bg=bg, highlightthickness=0)
        self.canvas.pack()

        self.root.bind("<Button-3>", lambda e: self.close())
        self.root.bind("<Escape>", lambda e: self.close())

        self.raw_level = 0.0   # volumen instantáneo del micrófono
        self.level = 0.0       # volumen suavizado (0..1)
        self.t0 = time.time()

        self.stream = None
        try:
            self.stream = sd.InputStream(channels=1, callback=self.audio_cb,
                                         blocksize=1024)
            self.stream.start()
        except Exception as e:
            print(f"No se pudo abrir el micrófono: {e}")

        self.running = True
        self.animate()

    def audio_cb(self, indata, frames, t, status):
        rms = float(np.sqrt(np.mean(indata ** 2)))
        self.raw_level = min(1.0, rms * GAIN)

    def blob(self, cx, cy, radius, amp, speed, phase, lobes):
        t = time.time() - self.t0
        pts = []
        for deg in range(0, 360, 6):
            a = math.radians(deg)
            wobble = (
                math.sin(lobes * a + t * speed + phase)
                + 0.6 * math.sin((lobes + 2) * a - t * speed * 1.3 + phase)
                + 0.3 * math.sin((lobes + 5) * a + t * speed * 0.7)
            )
            r = radius + amp * wobble
            pts.extend((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    @staticmethod
    def mix(c1, c2, k):
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(r1 + (r2 - r1) * k)
        g = int(g1 + (g2 - g1) * k)
        b = int(b1 + (b2 - b1) * k)
        return f"#{r:02x}{g:02x}{b:02x}"

    def animate(self):
        if not self.running:
            return

        target = self.raw_level
        if target > self.level:
            self.level += (target - self.level) * 0.45
        else:
            self.level += (target - self.level) * 0.08
        lv = self.level

        c = self.canvas
        c.delete("all")
        cx = cy = SIZE / 2
        t = time.time() - self.t0
        
        breath = math.sin(t * 1.6) * 2.5
        radius = BASE_RADIUS + breath + lv * 48
        amp = 3 + lv * 20

        for i in range(7, 0, -1):
            rr = radius + 12 + i * 7 + lv * 10 * (i / 7)
            col = self.mix(TRANSPARENT, "#f6d8ff", 0.12 + (7 - i) * 0.09)
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                          fill=col, outline="")

        rings = [
            ("#f0d6f3", 3 + lv * 3, radius + 6, amp * 1.0, 1.7, 0.0, 3),
            ("#fad6ff", 3 + lv * 3, radius + 2, amp * 1.2, -1.3, 2.0, 4),
            ("#d7beff", 2 + lv * 2, radius - 2, amp * 0.9, 2.1, 4.0, 5),
            ("#ffffff", 1 + lv * 1.5, radius + 4, amp * 0.7, 1.1, 1.0, 3),
        ]
        for color, width, r, a, sp, ph, lobes in rings:
            pts = self.blob(cx, cy, r, a, sp, ph, lobes)
            c.create_polygon(pts, smooth=True, fill="", outline=color, width=width)

        # Núcleo con degradado (círculos concéntricos)
        core = radius * 0.78
        steps = 14
        for i in range(steps):
            k = i / (steps - 1)
            rr = core * (1 - k * 0.92)
            col = self.mix("#b97cff", "#ae69fd", k ** 1.4)
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                          fill=col, outline="")

        self.root.after(int(1000 / FPS), self.animate)

    def close(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()


if __name__ == "__main__":
    Aura().run()