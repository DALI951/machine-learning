"""Evolutionary training for the Pong learner.

Only the learning loop lives here: selection, mutation, reward signal.
The strategy itself is NEVER scripted — it must emerge from evolution.

Usage:
    python evolve.py [--gens N] [--seed S] [--verify [checkpoint.json]]
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

from pong_env import PongGame, WIN_SCORE, play_one
from brain import genome_size, decide, decide_batch

POP = 80
ELITE = 10
MUT_RATE = 0.15
MUT_SIGMA = 0.25
CROSSOVER = 0.5
LOG_FILE = "pong_log.txt"
CKPT_DIR = "checkpoints"


def make_pop(rng):
    return rng.normal(0.0, 1.0, (POP, genome_size()))


def evaluate(pop, rng):
    game = PongGame(POP, rng)
    while not game.done.all():
        states = game.states()
        moves = decide_batch(pop, states)
        game.step(moves)
    return game.fitness(), game.points, game.opp_points, game.hits


def select(pop, fit, rng):
    order = np.argsort(-fit)
    elites = pop[order[:ELITE]].copy()
    parents = pop[order[:POP // 2]]
    children = []
    while len(children) < POP - ELITE:
        a, b = parents[rng.integers(0, len(parents), 2)]
        if rng.random() < CROSSOVER:
            mask = rng.random(genome_size()) < 0.5
            child = np.where(mask, a, b)
        else:
            child = a.copy()
        mut = rng.random(genome_size()) < MUT_RATE
        child[mut] += rng.normal(0.0, MUT_SIGMA, int(mut.sum()))
        children.append(child)
    return np.vstack([elites, np.array(children)])


def save_checkpoint(gen, genome, fitness, pts, opp, hits, wins, tag="ckpt"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = {
        "gen": gen,
        "fitness": float(fitness),
        "points": int(pts),
        "opp_points": int(opp),
        "hits": int(hits),
        "wins": int(wins),
        "timestamp": datetime.datetime.now().isoformat(),
        "genome": genome.tolist(),
    }
    path = os.path.join(CKPT_DIR, f"gen_{gen:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    with open("best.json", "w") as f:
        json.dump(data, f)
    return path


def load_genome(path):
    with open(path) as f:
        return np.array(json.load(f)["genome"])


def verify(genome, games=100, rng=None):
    """Play `games` full games vs the scripted AI. Returns (wins, losses, draws)."""
    if rng is None:
        rng = np.random.default_rng()
    wins = 0
    losses = 0
    draws = 0
    for _ in range(games):
        pts, opp, _ = play_one(genome, decide, rng)
        if pts >= WIN_SCORE:
            wins += 1
        elif opp >= WIN_SCORE:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


def train(gens, seed):
    rng = np.random.default_rng(seed)
    pop = make_pop(rng)
    best_fit_ever = -1e18
    best_genome = None
    best_pts = best_opp = best_hits = 0
    milestones = {"hit": False, "point": False, "win": False}

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== RUN {datetime.datetime.now().isoformat()} "
                  f"seed={seed} pop={POP} elite={ELITE} mut_rate={MUT_RATE} "
                  f"sigma={MUT_SIGMA} crossover={CROSSOVER} gens={gens} ===\n")
        for g in range(1, gens + 1):
            t0 = time.time()
            fit, pts, opp, hits = evaluate(pop, rng)
            best_i = int(np.argmax(fit))
            best_f = float(fit[best_i])
            avg = float(fit.mean())
            wins = int((pts >= WIN_SCORE).sum())

            if not milestones["hit"] and int(hits.max()) > 0:
                milestones["hit"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST BALL HIT (hits={int(hits.max())})")
            if not milestones["point"] and int(pts.max()) > 0:
                milestones["point"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST POINT SCORED")
            if not milestones["win"] and wins > 0:
                milestones["win"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST FULL WIN (11 pts)")

            if best_f > best_fit_ever:
                best_fit_ever = best_f
                best_genome = pop[best_i].copy()
                best_pts, best_opp, best_hits = int(pts[best_i]), int(opp[best_i]), int(hits[best_i])

            dt = time.time() - t0
            line = (f"{datetime.datetime.now().isoformat()} gen={g} "
                    f"best={best_f:.1f} avg={avg:.1f} "
                    f"pts={int(pts[best_i])} opp={int(opp[best_i])} hits={int(hits[best_i])} "
                    f"wins={wins} dt={dt:.2f}s")
            log.write("\n" + line)
            log.flush()
            print(line, flush=True)

            if g % 50 == 0:
                save_checkpoint(g, best_genome, best_fit_ever, best_pts, best_opp, best_hits, wins)

            pop = select(pop, fit, rng)

        save_checkpoint(gens, best_genome, best_fit_ever, best_pts, best_opp, best_hits, wins)
        log.write(f"\n=== END best_fit_ever={best_fit_ever:.1f} ===\n")
    print(f"\nDONE. best fitness ever: {best_fit_ever:.1f} "
          f"(pts={best_pts} opp={best_opp} hits={best_hits})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", nargs="?", const="best.json", default=None)
    args = ap.parse_args()

    if args.verify:
        genome = load_genome(args.verify)
        wins, losses, draws = verify(genome, games=100)
        print(f"VERIFY {args.verify}: {wins} wins / {losses} losses / {draws} draws "
              f"vs scripted AI (100 games)")
        print("NEVER LOSES" if losses == 0 and wins == 100 else "STILL LOSES")
        return

    train(args.gens, args.seed)


if __name__ == "__main__":
    main()