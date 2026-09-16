"""Genetic programming: evolve PROGRAMS (not weights) that play Pong.

Same environment, same fitness, same GA loop as the NN version — the only
difference is the genome: a list of (condition, action) rules instead of a
weight vector. If it learns, the machine has written its own code.
"""
import argparse
import datetime
import json
import os

import numpy as np

from game_env import (PongGame, SingleGame, COURT_W, COURT_H, MAX_SPEED,
                      PADDLE_H, PADDLE_X)
from program import (random_program, mutate, crossover, run,
                     state_from_nn, to_text)

POP = 80
ELITE = 10
MUT_RATE = 0.15
CROSSOVER = 0.5
CKPT_DIR = "checkpoints"
LOG_FILE = "program_log.txt"


def evaluate(pop, rng):
    """All programs play simultaneously (vectorized physics, scalar brains)."""
    game = PongGame(len(pop), rng)
    while not game.done.all():
        moves = np.zeros(len(pop))
        for i, prog in enumerate(pop):
            s = state_from_nn(np.array([game.bx[i] / COURT_W,
                                        game.by[i] / COURT_H,
                                        game.bvx[i] / MAX_SPEED,
                                        game.bvy[i] / MAX_SPEED,
                                        (game.py[i] + PADDLE_H / 2.0) / COURT_H,
                                        (game.bx[i] - PADDLE_X) / COURT_W]))
            moves[i] = run(prog, s)
        game.step(moves)
    fit = game.fitness()
    # anti-bloat: small penalty per rule (selection pressure, keeps code readable)
    fit = fit - 0.5 * np.array([len(p) for p in pop])
    return fit, game.points, game.opp_points, game.hits


def save_checkpoint(gen, prog, fitness, pts, opp, hits, tag="ckpt"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = {
        "gen": gen,
        "fitness": float(fitness),
        "pts": int(pts),
        "opp": int(opp),
        "hits": int(hits),
        "timestamp": datetime.datetime.now().isoformat(),
        "program": prog,
        "text": to_text(prog),
    }
    with open(os.path.join(CKPT_DIR, f"gen_{gen:04d}.json"), "w") as f:
        json.dump(data, f)
    with open("best.json", "w") as f:
        json.dump(data, f)


def load_program(path):
    with open(path) as f:
        return json.load(f)["program"]


def verify(prog, games=100, rng=None):
    """Play `games` full games with the program. Returns (wins, losses, draws, avg_hits)."""
    rng = rng if rng is not None else np.random.default_rng()
    wins = losses = draws = 0
    hits = []
    for _ in range(games):
        g = SingleGame(rng)
        while not g.over():
            s = state_from_nn(g.state()[0])
            g.step(run(prog, s))
        hits.append(g.hits)
        if g.pts > g.opp:
            wins += 1
        elif g.pts < g.opp:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws, float(np.mean(hits))


def train(gens, seed):
    rng = np.random.default_rng(seed)
    pop = [random_program(rng) for _ in range(POP)]
    best_prog = None
    best_fit = -1e9
    best_pts = best_opp = best_hits = 0

    with open(LOG_FILE, "a") as log:
        log.write(f"\n=== run seed={seed} gens={gens} {datetime.datetime.now().isoformat()} ===\n")
        for g in range(1, gens + 1):
            fit, pts, opp, hits = evaluate(pop, rng)
            order = np.argsort(-fit)
            elites = [pop[i] for i in order[:ELITE]]
            if fit[order[0]] > best_fit:
                best_fit = float(fit[order[0]])
                best_prog = list(pop[order[0]])
                best_pts, best_opp, best_hits = int(pts[order[0]]), int(opp[order[0]]), int(hits[order[0]])
                save_checkpoint(g, best_prog, best_fit, best_pts, best_opp, best_hits)
            children = []
            while len(children) < POP - ELITE:
                a = pop[int(rng.integers(POP))]
                b = pop[int(rng.integers(POP))]
                if rng.random() < CROSSOVER:
                    child = crossover(a, b, rng)
                else:
                    child = list(a)
                if rng.random() < MUT_RATE:
                    child = mutate(child, rng)
                children.append(child)
            pop = elites + children
            line = (f"{datetime.datetime.now().isoformat()} gen={g} "
                    f"best={fit[order[0]]:.1f} avg={fit.mean():.1f} "
                    f"pts={int(pts[order[0]])} opp={int(opp[order[0]])} "
                    f"hits={int(hits[order[0]])} "
                    f"prog=[{to_text(pop[order[0]])}]")
            print(line)
            log.write(line + "\n")
    print(f"\nDONE. best fitness ever: {best_fit:.1f} "
          f"({best_pts}-{best_opp}, {best_hits} hits)")
    print(f"BEST PROGRAM: {to_text(best_prog)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", nargs="?", const="best.json", default=None)
    args = ap.parse_args()
    if args.verify:
        prog = load_program(args.verify)
        wins, losses, draws, avg_hits = verify(prog)
        print(f"VERIFY {args.verify}: {wins} wins / {losses} losses / {draws} draws "
              f"(avg {avg_hits:.1f} hits)")
        print(f"PROGRAM: {to_text(prog)}")
    else:
        train(args.gens, args.seed)


if __name__ == "__main__":
    main()