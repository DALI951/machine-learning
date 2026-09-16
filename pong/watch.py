"""Watch the champion play Pong live + fitness chart from the log.

Usage:
    python watch.py [checkpoint.json]     (default: best.json)
    python watch.py --human [checkpoint.json]   (YOU play the right paddle)
Keys: R = restart game, Q = quit
Human mode: Up/Down arrows or W/S move your paddle (right side, red).
The champion plays its trained side (left, white). Tally counts forever.
"""
import json
import os
import re
import sys
import tkinter as tk

import numpy as np

from pong_env import (COURT_W, COURT_H, PADDLE_W, PADDLE_H, BALL_R,
                      PADDLE_X, OPP_X, WIN_SCORE, SingleGame)
from brain import decide

SCALE = 8  # 100x60 court -> 800x480 px
BG = "#0a0a0f"
FG = "#e5e7eb"
RED = "#ef4444"
DIM = "#3f3f46"


def load_genome(path):
    with open(path) as f:
        return np.array(json.load(f)["genome"])


def parse_log(path):
    """Return (gens, bests) from pong_log.txt for the fitness chart."""
    gens, bests = [], []
    if not os.path.exists(path):
        return gens, bests
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"gen=(\d+) best=([-\d.]+)", line)
            if m:
                gens.append(int(m.group(1)))
                bests.append(float(m.group(2)))
    return gens, bests


class Watch:
    def __init__(self, root, genome, ckpt_label, human=False):
        self.root = root
        self.genome = genome
        self.human = human
        self.human_move = 0
        self.champ_wins = 0
        self.human_wins = 0
        self.draws = 0
        self.result_timer = 0
        self.result_text = ""
        self.w = int(COURT_W * SCALE)
        self.h = int(COURT_H * SCALE)
        root.title("Pong learner — YOU vs the champion" if human
                   else "Pong learner — watch it play")
        root.configure(bg=BG)

        self.cv = tk.Canvas(root, width=self.w, height=self.h, bg=BG,
                            highlightthickness=0)
        self.cv.pack(padx=10, pady=(10, 0))

        self.chart = tk.Canvas(root, width=self.w, height=120, bg=BG,
                               highlightthickness=0)
        self.chart.pack(padx=10, pady=(6, 0))

        self.info = tk.Label(root, text="", bg=BG, fg=FG,
                             font=("Consolas", 11))
        self.info.pack(pady=(4, 8))

        self.ckpt_label = ckpt_label
        self.game = SingleGame()
        self.root.bind("<Key-r>", lambda e: self.game.reset())
        self.root.bind("<Key-q>", lambda e: root.destroy())
        if human:
            for k in ("<Up>", "<w>", "<W>"):
                self.root.bind(k, lambda e: self._set_human(-1))
            for k in ("<Down>", "<s>", "<S>"):
                self.root.bind(k, lambda e: self._set_human(1))
            for k in ("<KeyRelease-Up>", "<KeyRelease-Down>",
                      "<KeyRelease-w>", "<KeyRelease-W>",
                      "<KeyRelease-s>", "<KeyRelease-S>"):
                self.root.bind(k, lambda e: self._set_human(0))
        self.root.after(16, self.tick)
        self.root.after(1000, self.redraw_chart)

    def _set_human(self, v):
        self.human_move = v

    def tick(self):
        if self.game.over():
            if self.result_timer == 0:
                if self.game.pts >= WIN_SCORE:
                    self.champ_wins += 1
                    self.result_text = "CHAMPION WINS"
                elif self.game.opp >= WIN_SCORE:
                    self.human_wins += 1
                    self.result_text = "YOU WIN"
                else:
                    self.draws += 1
                    self.result_text = "DRAW (frame cap)"
            self.result_timer += 1
            if self.result_timer >= 90:
                self.game.reset()
                self.result_timer = 0
                self.result_text = ""
        else:
            move = decide(self.genome, self.game.state())[0]
            if self.human:
                self.game.step(move, opp_move=self.human_move)
            else:
                self.game.step(move)
        self.draw()
        self.root.after(16, self.tick)

    def draw(self):
        cv = self.cv
        cv.delete("all")
        g = self.game
        # center line
        cx = self.w // 2
        cv.create_line(cx, 0, cx, self.h, fill=DIM, dash=(8, 8))
        # paddles
        lx = int(PADDLE_X * SCALE)
        rx = int(OPP_X * SCALE)
        cv.create_rectangle(lx, int(g.py * SCALE),
                            lx + int(PADDLE_W * SCALE),
                            int((g.py + PADDLE_H) * SCALE),
                            fill=FG, outline="")
        cv.create_rectangle(rx, int(g.oy * SCALE),
                            rx + int(PADDLE_W * SCALE),
                            int((g.oy + PADDLE_H) * SCALE),
                            fill=RED if self.human else DIM, outline="")
        # ball (red accent)
        bx = int(g.bx * SCALE)
        by = int(g.by * SCALE)
        r = int(BALL_R * SCALE)
        cv.create_oval(bx - r, by - r, bx + r, by + r, fill=RED, outline="")
        # score
        cv.create_text(self.w // 2 - 40, 20, text=str(g.pts), fill=FG,
                       font=("Consolas", 18, "bold"))
        cv.create_text(self.w // 2 + 40, 20, text=str(g.opp),
                       fill=RED if self.human else DIM,
                       font=("Consolas", 18, "bold"))
        if self.human:
            self.info.config(
                text=f"{self.ckpt_label} | champion {self.champ_wins} - "
                     f"{self.human_wins} you ({self.draws} draws) | "
                     f"hits={g.hits} | arrows/W-S move your paddle")
        else:
            self.info.config(text=f"{self.ckpt_label} | hits={g.hits} | "
                                  f"frames={g.frames} | learner vs scripted AI")
        if self.result_text:
            cv.create_text(self.w // 2, self.h // 2, text=self.result_text,
                           fill=RED if "YOU" in self.result_text else FG,
                           font=("Consolas", 28, "bold"))

    def redraw_chart(self):
        ch = self.chart
        ch.delete("all")
        gens, bests = parse_log(LOG_PATH)
        if len(gens) >= 2:
            w, h = self.w, 120
            gmin, gmax = min(gens), max(gens)
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
            ch.create_text(6, 6, anchor="nw", text="best fitness per generation",
                           fill=DIM, font=("Consolas", 9))
        self.root.after(1000, self.redraw_chart)


LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pong_log.txt")


def main():
    args = sys.argv[1:]
    human = "--human" in args
    ckpt = next((a for a in args if not a.startswith("--")), "best.json")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt} — train first: python evolve.py --gens 300")
        sys.exit(1)
    genome = load_genome(ckpt)
    root = tk.Tk()
    Watch(root, genome, os.path.basename(ckpt), human=human)
    root.mainloop()


if __name__ == "__main__":
    main()