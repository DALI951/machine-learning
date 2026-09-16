# machine-learning

The lineage: machines that learn to play games by themselves — from zero, no hardcoded strategy.

Each folder is one learning experiment. Every game is a tiny neural network + genetic algorithm
(numpy only). The environment is written by hand; the *strategy is never scripted* — it must
emerge from evolution.

## Games

| Folder | Game | Learner controls | Learned result |
|--------|------|------------------|----------------|
| `pong/` | Pong vs scripted AI | paddle up/down | **100 wins / 0 losses / 0 draws** vs scripted AI (400 gens) |
| `flappy/` | Flappy Bird | flap or don't flap | in progress |

## How each folder works

- `*_env.py` — the world (physics, opponent, scoring). Nothing learned here.
- `brain.py` — tiny feed-forward NN (numpy). Genome = flat weight vector.
- `evolve.py` — the learning loop: evaluate population → select elites → crossover/mutate.
  `--verify` plays 100 fresh games against the checkpoint.
- `watch.py` — watch the best agent play.
- `live.py` — LIVE training window: watch it learn generation by generation, and drag the
  speed slider to change the environment and watch it re-adapt (green lines on the chart
  mark every speed change).

## Run it

```bash
cd pong        # or flappy
python evolve.py --gens 400 --seed 42   # train
python evolve.py --verify best.json     # test the champion
python live.py                          # watch it learn live
python watch.py best.json               # watch the champion play
```

## The creed

1. Nothing hardcoded — never script behavior into the agent. Only touch selection, reward,
   mutation, environment difficulty.
2. Instrument everything — every run writes a plain-text log with timestamps.
3. Meaningful progress > scripted success. Never fake results.
4. Watch for stagnation (no fitness movement, population collapse, reward hacking) and fix
   the pressure, not the learner.