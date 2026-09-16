"""LIVE training: watch the Flappy learner go from 0 AND adapt to your speed.

The GA keeps running in a background thread, generation after generation.
The slider changes the PIPE SPEED (0.5x - 3x). Move it and watch:
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

from flappy_env import (COURT_W, COURT_H, BIRD_X, BIRD_R, PIPE_W, GAP_H,
                        FlappyGame, SingleFlappy)
from brain import decide, decide_batch
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
        self.best_score = 0
        self.best_genome = None
        self.history = []          # (gen, best_fit)
        self.speed_changes = []    # (gen, speed) markers on the chart
        self.game = None

        self.w = int(COURT_W * SCALE)
        self.h = int(COURT_H * SCALE)
        root.title("Flappy learner — LIVE training (drag the pipe-speed slider)")
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

        # pipe-speed slider
        self.slider = tk.Scale(root, from_=0.5, to=3.0, resolution=0.1,
                               orient="horizontal", length=self.w,
                               label="PIPE SPEED", bg=BG, fg=FG,
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
                game = FlappyGame(POP, rng, speed_mult=sm)
                while not game.done.all():
                    states = game.states()
                    moves = decide_batch(pop, states)
                    game.step(moves)
                fit, score, alive = game.fitness(), game.score, game.alive
                best_i = int(np.argmax(fit))
                with self.lock:
                    self.gen += 1
                    self.best_fit = float(fit[best_i])
                    self.avg_fit = float(fit.mean())
                    self.best_score = int(score.max())
                    self.best_genome = pop[best_i].copy()
                    self.history.append((self.gen, self.best_fit))
                    if len(self.history) > 3000:
                        self.history.pop(0)
                log.write(f"\n{datetime.datetime.now().isoformat()} "
                          f"gen={self.gen} speed={sm:.1f} "
                          f"best={self.best_fit:.1f} avg={self.avg_fit:.1f} "
                          f"score={int(score.max())}")
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
            score = self.best_score
        if genome is not None:
            if self.game is None or self.game.done():
                self.game = SingleFlappy(speed_mult=sm)
            for _ in range(2):
                if not self.game.done():
                    move = decide(genome, self.game.state())[0]
                    self.game.step(move)
        self.draw(gen, best, avg, score, sm)
        self.root.after(16, self.tick)

    def draw(self, gen, best, avg, score, sm):
        cv = self.cv
        cv.delete("all")
        g = self.game
        # ground line
        gy = int((COURT_H - 1) * SCALE)
        cv.create_line(0, gy, self.w, gy, fill=DIM)
        if g is not None:
            # bird (red accent)
            bx = int(BIRD_X * SCALE)
            by = int(g.y * SCALE)
            r = int(BIRD_R * SCALE)
            cv.create_oval(bx - r, by - r, bx + r, by + r, fill=RED, outline="")
            # pipes
            for k in range(20):
                px = g._pipe_x(k)
                if -PIPE_W < px < COURT_W + PIPE_W:
                    x0 = int(px * SCALE)
                    x1 = int((px + PIPE_W) * SCALE)
                    gap_c = int(g.gaps[k] * SCALE)
                    half = int(GAP_H / 2.0 * SCALE)
                    cv.create_rectangle(x0, 0, x1, gap_c - half, fill=DIM, outline="")
                    cv.create_rectangle(x0, gap_c + half, x1, gy, fill=DIM, outline="")
            # score
            cv.create_text(self.w // 2, 20, text=str(g.score), fill=FG,
                           font=("Consolas", 18, "bold"))
        self.stats.config(
            text=f"gen {gen} | best {best:.1f} | avg {avg:.1f} | "
                 f"best score {score} | pipe speed {sm:.1f}x")
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
        for g, sm in changes:
            if gmin <= g <= gmax:
                x = 4 + (g - gmin) / (gmax - gmin) * (w - 8)
                ch.create_line(x, 4, x, h - 4, fill=GREEN, width=2)
        ch.create_text(6, 6, anchor="nw",
                       text="best fitness (live) — green lines = speed changes",
                       fill=DIM, font=("Consolas", 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = tk.Tk()
    LiveTrain(root, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()