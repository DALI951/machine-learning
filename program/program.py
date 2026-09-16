"""The mini-language: a program is a list of (condition, action) rules.

The machine's genome IS code. Each frame, the first rule whose condition is
true fires. Evolution writes these programs — nothing here is hand-designed
strategy, just the alphabet the machine is allowed to write with.

Conditions read the raw game state (decoded from the normalized NN state):
  ball_x, ball_y, ball_vx, paddle_center_y
"""
import numpy as np

from game_env import COURT_W, COURT_H, MAX_SPEED

# --- the alphabet -----------------------------------------------------------

CONDS = {
    "BALL_ABOVE": lambda s: s["ball_y"] < s["paddle_y"],
    "BALL_BELOW": lambda s: s["ball_y"] > s["paddle_y"],
    "BALL_LEFT":  lambda s: s["ball_x"] < COURT_W / 2.0,
    "BALL_RIGHT": lambda s: s["ball_x"] >= COURT_W / 2.0,
    "COMING":     lambda s: s["ball_vx"] < 0.0,   # moving toward me (left paddle)
    "GOING":      lambda s: s["ball_vx"] >= 0.0,  # moving away
    "CLOSE":      lambda s: abs(s["ball_y"] - s["paddle_y"]) < 20.0,
    "ALWAYS":     lambda s: True,
}

ACTIONS = {"UP": -1.0, "DOWN": 1.0, "STAY": 0.0}

MIN_RULES = 1
MAX_RULES = 6
HARD_CAP = 12  # mutate() may never grow a program past this (anti-bloat)


def state_from_nn(s):
    """Decode the normalized 6-vector into raw values the language can read."""
    return {
        "ball_x": s[0] * COURT_W,
        "ball_y": s[1] * COURT_H,
        "ball_vx": s[2] * MAX_SPEED,
        "paddle_y": s[4] * COURT_H,
    }


def run(program, s):
    """First matching rule fires. Returns -1 (up), 1 (down) or 0 (stay)."""
    for cond, act in program:
        if CONDS[cond](s):
            return ACTIONS[act]
    return 0.0


def to_text(program):
    if not program:
        return "[empty]"
    return "  ".join(f"{c}->{a}" for c, a in program)


# --- evolution operators on code -------------------------------------------

def random_program(rng):
    n = int(rng.integers(MIN_RULES, MAX_RULES + 1))
    conds = list(CONDS)
    acts = list(ACTIONS)
    return [(conds[int(rng.integers(len(conds)))],
             acts[int(rng.integers(len(acts)))]) for _ in range(n)]


def mutate(prog, rng):
    """Edit the code: change a rule, add a rule, delete a rule, or swap two."""
    p = list(prog)
    op = rng.random()
    conds = list(CONDS)
    acts = list(ACTIONS)
    if op < 0.30 and p:
        i = int(rng.integers(len(p)))
        if rng.random() < 0.5:
            p[i] = (conds[int(rng.integers(len(conds)))], p[i][1])
        else:
            p[i] = (p[i][0], acts[int(rng.integers(len(acts)))])
    elif op < 0.50 and len(p) < HARD_CAP:
        p.insert(int(rng.integers(len(p) + 1)),
                 (conds[int(rng.integers(len(conds)))],
                  acts[int(rng.integers(len(acts)))]))
    elif op < 0.70 and len(p) > 1:
        del p[int(rng.integers(len(p)))]
    elif op < 0.85 and len(p) > 1:
        i, j = int(rng.integers(len(p))), int(rng.integers(len(p)))
        p[i], p[j] = p[j], p[i]
    return p


def crossover(a, b, rng):
    """Splice: first part of A + second part of B."""
    k1 = int(rng.integers(0, len(a) + 1))
    k2 = int(rng.integers(0, len(b) + 1))
    return a[:k1] + b[k2:]