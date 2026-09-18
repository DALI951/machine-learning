"""Evolutionary training for the Flappy learner.

Only the learning loop lives here: selection, mutation, reward signal.
The flap strategy is NEVER scripted — it must emerge from evolution.

Usage:
    python evolve.py [--gens N] [--seed S] [--verify [checkpoint.json]]
"""
import argparse
import datetime
import json
import os
import time

import numpy as np

from flappy_env import FlappyGame, play_one
from brain import genome_size, decide, decide_batch

POP = 80
ELITE = 10
MUT_RATE = 0.15
MUT_SIGMA = 0.25
CROSSOVER = 0.5
LOG_FILE = "flappy_log.txt"
CKPT_DIR = "checkpoints"


def make_pop(rng):
    return rng.normal(0.0, 1.0, (POP, genome_size()))


def evaluate(pop, rng):
    game = FlappyGame(POP, rng)
    while not game.done.all():
        states = game.states()
        moves = decide_batch(pop, states)
        game.step(moves)
    return game.fitness(), game.score, game.alive


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


def save_best(gen, genome, fitness, score, alive):
    """Write best.json only (no checkpoint file). Binary-safe, atomic-ish."""
    data = {
        "gen": gen,
        "fitness": float(fitness),
        "score": int(score),
        "alive": int(alive),
        "timestamp": datetime.datetime.now().isoformat(),
        "genome": genome.tolist(),
    }
    with open("best.json", "w") as f:
        json.dump(data, f)


def save_checkpoint(gen, genome, fitness, score, alive, tag="ckpt"):
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = {
        "gen": gen,
        "fitness": float(fitness),
        "score": int(score),
        "alive": int(alive),
        "timestamp": datetime.datetime.now().isoformat(),
        "genome": genome.tolist(),
    }
    path = os.path.join(CKPT_DIR, f"gen_{gen:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    save_best(gen, genome, fitness, score, alive)
    return path


def load_genome(path):
    with open(path) as f:
        return np.array(json.load(f)["genome"])


def verify(genome, games=100, rng=None):
    """Play `games` full games. Returns (avg_score, best_score, avg_alive)."""
    if rng is None:
        rng = np.random.default_rng()
    scores = []
    alives = []
    for _ in range(games):
        score, alive = play_one(genome, decide, rng)
        scores.append(score)
        alives.append(alive)
    return (float(np.mean(scores)), int(np.max(scores)),
            float(np.mean(alives)))


def train(gens, seed):
    rng = np.random.default_rng(seed)
    pop = make_pop(rng)
    best_fit_ever = -1e18
    best_genome = None
    best_score = best_alive = 0
    milestones = {"survive": False, "pipe": False, "ten": False, "fifty": False}

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n=== RUN {datetime.datetime.now().isoformat()} "
                  f"seed={seed} pop={POP} elite={ELITE} mut_rate={MUT_RATE} "
                  f"sigma={MUT_SIGMA} crossover={CROSSOVER} gens={gens} ===\n")
        for g in range(1, gens + 1):
            t0 = time.time()
            fit, score, alive = evaluate(pop, rng)
            best_i = int(np.argmax(fit))
            best_f = float(fit[best_i])
            avg = float(fit.mean())
            best_s = int(score.max())
            best_a = int(alive.max())

            if not milestones["survive"] and best_a > 100:
                milestones["survive"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST BIRD SURVIVES 100+ FRAMES")
            if not milestones["pipe"] and best_s > 0:
                milestones["pipe"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST PIPE PASSED")
            if not milestones["ten"] and best_s >= 10:
                milestones["ten"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST 10 PIPES")
            if not milestones["fifty"] and best_s >= 50:
                milestones["fifty"] = True
                log.write(f"\nMILESTONE gen {g}: FIRST 50 PIPES")

            if best_f > best_fit_ever:
                best_fit_ever = best_f
                best_genome = pop[best_i].copy()
                best_score, best_alive = best_s, best_a

            dt = time.time() - t0
            line = (f"{datetime.datetime.now().isoformat()} gen={g} "
                    f"best={best_f:.1f} avg={avg:.1f} "
                    f"score={best_s} alive={best_a} "
                    f"dt={dt:.2f}s")
            log.write("\n" + line)
            log.flush()
            print(line, flush=True)

            if g % 50 == 0:
                save_checkpoint(g, best_genome, best_fit_ever, best_score, best_alive)

            pop = select(pop, fit, rng)

        save_checkpoint(gens, best_genome, best_fit_ever, best_score, best_alive)
        log.write(f"\n=== END best_fit_ever={best_fit_ever:.1f} ===\n")
    print(f"\nDONE. best fitness ever: {best_fit_ever:.1f} "
          f"(score={best_score} alive={best_alive})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", nargs="?", const="best.json", default=None)
    args = ap.parse_args()

    if args.verify:
        genome = load_genome(args.verify)
        avg_s, best_s, avg_a = verify(genome, games=100)
        print(f"VERIFY {args.verify}: avg score {avg_s:.1f} / best {best_s} "
              f"/ avg alive {avg_a:.1f} frames (100 games)")
        return

    train(args.gens, args.seed)


if __name__ == "__main__":
    main()