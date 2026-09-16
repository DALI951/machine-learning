"""LIVE training: watch the machine learn AND adapt to your speed changes.

The GA keeps running in a background thread, generation after generation.
The slider changes the BALL speed (0.5x - 3x). Move it and watch:
fitness drops, then climbs back as the population re-adapts.

Usage:
    python live.py [--seed S]

Keys: Q = quit
"""
import argparse
import datetime
import os
import threading
import tkinter as tk

import numpy as np

from pong_env import (COURT_W, COURT_H, PADDLE_W, PADDLE_H, BALL_R,
                      PADDLE_X, OPP_X, WIN_SCORE, PongGame, SingleGame)
from brain import genome_size, decide, decide_batch
from evolve import make_pop, select, POP

SCALE = 8
BG = "#0a0a0f"
FG = "#e5e7eb"
RED = "#ef4444"
DIM = "#3f3f46"
GREEN = "#22c55e"

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_log.txt")


class LiveTrain:
    def __init__(self, root, seed):
        self.root = root
        self.seed = seed
        self.lock = threading.Lock()
        self.stop = False
        self.speed_mult = 1.0
        self.gen = 0
        self.best_fit = 0.0
        self.avg_fit = 0.0
        self.wins = 0
        self.best_genome = None
        self.history = []          # (gen, best_fit)
        self.speed_changes = []    # (gen, speed) markers on the chart
        self.game = None
        self.last_sm = 1.0

        self.w = int(COURT_W * SCALE)
        self.h = int(COURT_H * SCALE)
        root.title("Pong learner — LIVE training (drag the speed slider)")
        root.configure(bg=BG)

        self.cv = tk.Canvas(root, width=self.w, height=self.h, bg=BG,
                            highlightthickness=0)
        self.cv.pack(padx=10, pady=(10, 0))

        self.chart = tk.Canvas(root, width=self.w, height=140, bg=BG,
                               highlightthickness=0)
        self.chart.pack(padx=10, pady=(6, 0))

        self.stats = tk.Label(root, text="", bg=BG, fg=FG,
                              font=("Consolas", 11))
        self.stats.pack(pady=(4, 0))

        # speed slider
        self.slider = tk.Scale(root, from_=0.5, to=3.0, resolution=0.1,
                               orient="horizontal", length=self.w,
                               label="BALL SPEED", bg=BG, fg=FG,
                               troughcolor=DIM, highlightthickness=0,
                               command=self._on_speed)
        self.slider.set(1.0)
        self.slider.pack(padx=10, pady=(0, 4))

        self.hint = tk.Label(root, text="drag the slider -> the machine must re-adapt",
                             bg=BG, fg=DIM, font=("Consolas", 9))
        self.hint.pack(pady=(0, 8))

        self.root.bind("<Key-q>", lambda e: self._quit())

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.root.after(16, self.tick)

    # ---- training thread ----
    def loop(self):
        rng = np.random.default_rng(self.seed)
        pop = make_pop(rng)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"\n=== LIVE {datetime.datetime.now().isoformat()} "
                      f"seed={self.seed} pop={POP} ===\n")
            while not self.stop:
                with self.lock:
                    sm = self.speed_mult
                game = PongGame(POP, rng, speed_mult=sm)
                while not game.done.all():
                    states = game.states()
                    moves = decide_batch(pop, states)
                    game.step(moves)
                fit, pts, opp, hits = (game.fitness(), game.points,
                                       game.opp_points, game.hits)
                best_i = int(np.argmax(fit))
                with self.lock:
                    self.gen += 1
                    self.best_fit = float(fit[best_i])
                    self.avg_fit = float(fit.mean())
                    self.wins = int((pts >= WIN_SCORE).sum())
                    self.best_genome = pop[best_i].copy()
                    self.history.append((self.gen, self.best_fit))
                    if len(self.history) > 3000:
                        self.history.pop(0)
                log.write(f"\n{datetime.datetime.now().isoformat()} "
                          f"gen={self.gen} speed={sm:.1f} "
                          f"best={self.best_fit:.1f} avg={self.avg_fit:.1f} "
                          f"wins={self.wins}")
                log.flush()
                pop = select(pop, fit, rng)

    # ---- UI thread ----
    def _on_speed(self, v):
        with self.lock:
            self.speed_mult = float(v)
            self.speed_changes.append((self.gen, float(v)))
            self.game = None  # restart the visual game at the new speed

    def _quit(self):
        self.stop = True
        self.root.destroy()

    def tick(self):
        with self.lock:
            sm = self.speed_mult
            genome = self.best_genome
            gen = self.gen
            best = self.best_fit
            avg = self.avg_fit
            wins = self.wins
        if genome is not None:
            if self.game is None or self.game.over():
                self.game = SingleGame(speed_mult=sm)
            for _ in range(2):
                if not self.game.over():
                    move = decide(genome, self.game.state())[0]
                    self.game.step(move)
        self.draw(gen, best, avg, wins, sm)
        self.root.after(16, self.tick)

    def draw(self, gen, best, avg, wins, sm):
        cv = self.cv
        cv.delete("all")
        g = self.game
        cx = self.w // 2
        cv.create_line(cx, 0, cx, self.h, fill=DIM, dash=(8, 8))
        if g is not None:
            lx = int(PADDLE_X * SCALE)
            rx = int(OPP_X * SCALE)
            cv.create_rectangle(lx, int(g.py * SCALE),
                                lx + int(PADDLE_W * SCALE),
                                int((g.py + PADDLE_H) * SCALE),
                                fill=FG, outline="")
            cv.create_rectangle(rx, int(g.oy * SCALE),
                                rx + int(PADDLE_W * SCALE),
                                int((g.oy + PADDLE_H) * SCALE),
                                fill=DIM, outline="")
            bx = int(g.bx * SCALE)
            by = int(g.by * SCALE)
            r = int(BALL_R * SCALE)
            cv.create_oval(bx - r, by - r, bx + r, by + r, fill=RED, outline="")
            cv.create_text(self.w // 2 - 40, 20, text=str(g.pts), fill=FG,
                           font=("Consolas", 18, "bold"))
            cv.create_text(self.w // 2 + 40, 20, text=str(g.opp), fill=DIM,
                           font=("Consolas", 18, "bold"))
        self.stats.config(
            text=f"gen {gen} | best {best:.1f} | avg {avg:.1f} | "
                 f"wins {wins}/{POP} | ball speed {sm:.1f}x")
        self._draw_chart()

    def _draw_chart(self):
        ch = self.chart
        ch.delete("all")
        with self.lock:
            hist = list(self.history)
            changes = list(self.speed_changes)
        if len(hist) < 2:
            ch.create_text(6, 6, anchor="nw", text="best fitness (live)",
                           fill=DIM, font=("Consolas", 9))
            return
        w, h = self.w, 140
        gens = [p[0] for p in hist]
        bests = [p[1] for p in hist]
        gmin, gmax = gens[0], gens[-1]
        bmin, bmax = min(bests), max(bests)
        if bmax - bmin < 1e-9:
            bmax = bmin + 1.0
        pts = []
        for g, b in zip(gens, bests):
            x = 4 + (g - gmin) / (gmax - gmin) * (w - 8)
            y = h - 8 - (b - bmin) / (bmax - bmin) * (h - 16)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            ch.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                           fill=RED, width=2)
        # speed-change markers (green vertical lines)
        for g, sm in changes:
            if gmin <= g <= gmax:
                x = 4 + (g - gmin) / (gmax - gmin) * (w - 8)
                ch.create_line(x, 4, x, h - 4, fill=GREEN, width=2)
        ch.create_text(6, 6, anchor="nw",
                       text="best fitness (live) — green lines = speed changes",
                       fill=DIM, font=("Consolas", 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    root = tk.Tk()
    LiveTrain(root, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()