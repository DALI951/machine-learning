"""Flappy Bird — vectorized environment for evolutionary learning.

The learner controls ONE thing: flap or don't flap. Gravity does the rest.
Nothing here is learned — this file is only the world. The brain must
figure out WHEN to flap by itself.

World: 100x60. Bird at x=25. Pipes move left. Gap is random per pipe.
Pipes are analytic: pipe k's x at frame t is a closed-form function,
so no pipe arrays are needed — just a pre-generated gap table.
"""
import numpy as np

COURT_W = 100.0
COURT_H = 60.0
BIRD_X = 25.0
BIRD_R = 2.0
GRAVITY = 0.25
FLAP_VY = -4.0
MAX_VY = 5.0
PIPE_W = 8.0
GAP_H = 20.0
PIPE_SPEED = 1.5
PIPE_SPACING = 45
MAX_PIPES = 80
MAX_FRAMES = 3000
GAP_MARGIN = 4.0  # gap center can't be too close to floor/ceiling


class FlappyGame:
    """Vectorized: a whole population plays simultaneously in numpy arrays."""

    def __init__(self, pop, rng, speed_mult=1.0):
        self.pop = pop
        self.rng = rng
        self.speed_mult = speed_mult
        self.reset()

    def reset(self):
        n = self.pop
        self.y = np.full(n, COURT_H / 2.0)
        self.vy = np.zeros(n)
        self.frames = 0
        self.alive = np.zeros(n, dtype=int)   # frames survived per agent
        self.score = np.zeros(n, dtype=int)   # pipes passed per agent
        self.done = np.zeros(n, dtype=bool)
        # pre-generate gap centers for every pipe each agent will ever see
        self.gaps = self.rng.uniform(
            GAP_H / 2.0 + GAP_MARGIN, COURT_H - GAP_H / 2.0 - GAP_MARGIN,
            (n, MAX_PIPES))

    def _pipe_x(self, k):
        """Left edge x of pipe k at the current frame (analytic)."""
        return (COURT_W + PIPE_W
                - (self.frames - k * PIPE_SPACING) * PIPE_SPEED * self.speed_mult)

    def _next_pipe(self):
        """(index, xs) of the next pipe not yet passed (same for all agents)."""
        ks = np.arange(MAX_PIPES)
        xs = self._pipe_x(ks)
        ahead = xs + PIPE_W >= BIRD_X
        k = int(np.argmax(ahead)) if ahead.any() else MAX_PIPES - 1
        return k, xs

    def step(self, moves):
        active = ~self.done
        # flap: moves > 0.5 -> set vy to flap velocity
        flap = active & (moves[:, 0] > 0.5)
        self.vy[flap] = FLAP_VY
        # gravity
        self.vy[active] += GRAVITY
        np.clip(self.vy, -MAX_VY, MAX_VY, out=self.vy)
        self.y[active] += self.vy[active]
        # ceiling clamp
        ceil = active & (self.y < BIRD_R)
        self.y[ceil] = BIRD_R
        self.vy[ceil] = 0.0
        # ground death
        ground = active & (self.y > COURT_H - BIRD_R)
        self.done[ground] = True
        # pipe collision (next pipe only — the only one that can hit)
        k, xs = self._next_pipe()
        px = xs[k]
        in_pipe_x = active & (BIRD_X + BIRD_R > px) & (BIRD_X - BIRD_R < px + PIPE_W)
        gap_lo = self.gaps[:, k] - GAP_H / 2.0
        gap_hi = self.gaps[:, k] + GAP_H / 2.0
        outside_gap = (self.y - BIRD_R < gap_lo) | (self.y + BIRD_R > gap_hi)
        self.done[in_pipe_x & outside_gap] = True
        # scoring: pipes whose right edge is left of the bird
        passed_count = int(np.sum(xs + PIPE_W < BIRD_X))
        self.score[active] = passed_count
        # survival time
        self.alive[active] += 1
        self.frames += 1
        self.done |= self.alive >= MAX_FRAMES

    def states(self):
        """Normalized NN inputs: y, vy, distance to next pipe, gap center."""
        k, xs = self._next_pipe()
        px = xs[k]
        gap_c = self.gaps[:, k]
        return np.column_stack([
            self.y / COURT_H,
            self.vy / MAX_VY,
            np.full(self.pop, (px - BIRD_X) / COURT_W),
            gap_c / COURT_H,
        ])

    def fitness(self):
        return self.alive + self.score * 100.0


class SingleFlappy:
    """Single bird for watching / verifying."""

    def __init__(self, rng=None, speed_mult=1.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.speed_mult = speed_mult
        self.reset()

    def reset(self):
        self.y = COURT_H / 2.0
        self.vy = 0.0
        self.frames = 0
        self.alive = 0
        self.score = 0
        self.over = False
        self.gaps = self.rng.uniform(
            GAP_H / 2.0 + GAP_MARGIN, COURT_H - GAP_H / 2.0 - GAP_MARGIN,
            MAX_PIPES)

    def _pipe_x(self, k):
        return (COURT_W + PIPE_W
                - (self.frames - k * PIPE_SPACING) * PIPE_SPEED * self.speed_mult)

    def _next_pipe(self):
        ks = np.arange(MAX_PIPES)
        xs = self._pipe_x(ks)
        ahead = xs + PIPE_W >= BIRD_X
        k = int(np.argmax(ahead)) if ahead.any() else MAX_PIPES - 1
        return k, xs

    def state(self):
        k, xs = self._next_pipe()
        px = xs[k]
        return np.array([[self.y / COURT_H, self.vy / MAX_VY,
                          (px - BIRD_X) / COURT_W, self.gaps[k] / COURT_H]])

    def step(self, move):
        if self.over:
            return
        if move > 0.5:
            self.vy = FLAP_VY
        self.vy += GRAVITY
        self.vy = min(max(self.vy, -MAX_VY), MAX_VY)
        self.y += self.vy
        if self.y < BIRD_R:
            self.y = BIRD_R
            self.vy = 0.0
        if self.y > COURT_H - BIRD_R:
            self.over = True
        k, xs = self._next_pipe()
        px = xs[k]
        if (BIRD_X + BIRD_R > px and BIRD_X - BIRD_R < px + PIPE_W
                and (self.y - BIRD_R < self.gaps[k] - GAP_H / 2.0
                     or self.y + BIRD_R > self.gaps[k] + GAP_H / 2.0)):
            self.over = True
        self.score = int(np.sum(xs + PIPE_W < BIRD_X))
        self.alive += 1
        self.frames += 1
        if self.alive >= MAX_FRAMES:
            self.over = True

    def done(self):
        return self.over


def play_one(genome, decide_fn, rng=None):
    """Play one full game with a brain. Returns (score, alive_frames)."""
    g = SingleFlappy(rng)
    while not g.done():
        move = decide_fn(genome, g.state())[0]
        g.step(move)
    return g.score, g.alive