"""Pong environment — physics + scripted opponent only. NOTHING here is learned.

The learner's behavior must emerge from evolution. This file only defines:
the court, the ball physics, the paddle mechanics, and a scripted opponent
that simply chases the ball (slower than the learner, so it is beatable).
"""
import numpy as np

COURT_W = 100.0
COURT_H = 60.0
PADDLE_W = 2.0
PADDLE_H = 12.0
BALL_R = 1.0
PADDLE_X = 4.0            # learner paddle x (left side)
OPP_X = COURT_W - 4.0     # opponent paddle x (right side)
BALL_SPEED0 = 3.0
MAX_SPEED = 6.0
SPEEDUP = 0.05            # ball speeds up 5% per paddle hit
PADDLE_SPEED = 2.5        # learner paddle speed
OPP_SPEED = 1.2           # scripted opponent speed (48% of learner -> beatable, scoring possible)
MAX_ANGLE = np.deg2rad(60.0)  # bounce angle depends on where the ball hits
WIN_SCORE = 11
MAX_FRAMES = 5000


class PongGame:
    """Vectorized: a whole population plays simultaneously in numpy arrays."""

    def __init__(self, pop, rng, speed_mult=1.0):
        self.pop = pop
        self.rng = rng
        self.speed_mult = speed_mult
        self.reset()

    def reset(self):
        n = self.pop
        self.bx = np.full(n, COURT_W / 2.0)
        self.by = np.full(n, COURT_H / 2.0)
        self.bvx = np.full(n, -BALL_SPEED0 * self.speed_mult)  # serve always toward the learner
        self.bvy = np.zeros(n)
        self.py = np.full(n, COURT_H / 2.0 - PADDLE_H / 2.0)
        self.oy = np.full(n, COURT_H / 2.0 - PADDLE_H / 2.0)
        self.points = np.zeros(n, dtype=int)
        self.opp_points = np.zeros(n, dtype=int)
        self.hits = np.zeros(n, dtype=int)
        self.frames = 0
        self.done = np.zeros(n, dtype=bool)

    def step(self, moves):
        active = ~self.done
        # learner paddles
        self.py += moves * PADDLE_SPEED
        np.clip(self.py, 0.0, COURT_H - PADDLE_H, out=self.py)
        # scripted opponent: chase the ball's y (slower than learner)
        target = self.by - PADDLE_H / 2.0
        diff = target - self.oy
        self.oy += np.clip(diff, -OPP_SPEED, OPP_SPEED)
        np.clip(self.oy, 0.0, COURT_H - PADDLE_H, out=self.oy)
        # ball
        self.bx += self.bvx
        self.by += self.bvy
        # wall bounce
        top = self.by < BALL_R
        bot = self.by > COURT_H - BALL_R
        self.bvy[top] = np.abs(self.bvy[top])
        self.bvy[bot] = -np.abs(self.bvy[bot])
        self.by[top] = BALL_R
        self.by[bot] = COURT_H - BALL_R
        # learner paddle hit (ball moving left, crossing the paddle plane)
        hit_l = (active & (self.bvx < 0)
                 & (self.bx - BALL_R <= PADDLE_X + PADDLE_W)
                 & (self.bx - BALL_R > PADDLE_X - 2.0)
                 & (self.by >= self.py) & (self.by <= self.py + PADDLE_H))
        # opponent paddle hit
        hit_r = (active & (self.bvx > 0)
                 & (self.bx + BALL_R >= OPP_X - PADDLE_W)
                 & (self.bx + BALL_R < OPP_X + 2.0)
                 & (self.by >= self.oy) & (self.by <= self.oy + PADDLE_H))
        if hit_l.any():
            off = (self.by[hit_l] - (self.py[hit_l] + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            ang = np.clip(off, -1.0, 1.0) * MAX_ANGLE
            sp = np.minimum(np.abs(self.bvx[hit_l]) * (1.0 + SPEEDUP),
                            MAX_SPEED * self.speed_mult)
            self.bvx[hit_l] = sp * np.cos(ang)
            self.bvy[hit_l] = sp * np.sin(ang)
            self.bx[hit_l] = PADDLE_X + PADDLE_W + BALL_R
            self.hits[hit_l] += 1
        if hit_r.any():
            off = (self.by[hit_r] - (self.oy[hit_r] + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            ang = np.clip(off, -1.0, 1.0) * MAX_ANGLE
            sp = np.minimum(np.abs(self.bvx[hit_r]) * (1.0 + SPEEDUP),
                            MAX_SPEED * self.speed_mult)
            self.bvx[hit_r] = -sp * np.cos(ang)
            self.bvy[hit_r] = sp * np.sin(ang)
            self.bx[hit_r] = OPP_X - PADDLE_W - BALL_R
        # scoring
        score_opp = active & (self.bx < -BALL_R)
        score_me = active & (self.bx > COURT_W + BALL_R)
        self.opp_points[score_opp] += 1
        self.points[score_me] += 1
        for m in (score_opp, score_me):
            if m.any():
                self._reset_ball(m)
        self.frames += 1
        self.done |= (self.points >= WIN_SCORE) | (self.opp_points >= WIN_SCORE) | (self.frames >= MAX_FRAMES)

    def _reset_ball(self, mask):
        n = int(mask.sum())
        self.bx[mask] = COURT_W / 2.0
        self.by[mask] = self.rng.uniform(BALL_R, COURT_H - BALL_R, n)
        self.bvx[mask] = -BALL_SPEED0 * self.speed_mult  # serve always toward the learner
        self.bvy[mask] = 0.0

    def fitness(self):
        win = np.where((self.points >= WIN_SCORE) & (self.opp_points < WIN_SCORE), 100.0, 0.0)
        return (self.points - self.opp_points) * 10.0 + self.hits + win

    def states(self):
        """(n, 6) normalized inputs for the NN."""
        s = np.empty((self.pop, 6))
        s[:, 0] = self.bx / COURT_W
        s[:, 1] = self.by / COURT_H
        s[:, 2] = self.bvx / MAX_SPEED
        s[:, 3] = self.bvy / MAX_SPEED
        s[:, 4] = (self.py + PADDLE_H / 2.0) / COURT_H
        s[:, 5] = (self.bx - PADDLE_X) / COURT_W
        return s


class SingleGame:
    """One game, frame by frame — used by the live watcher and by verify."""

    def __init__(self, rng=None, speed_mult=1.0):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.speed_mult = speed_mult
        self.reset()

    def reset(self):
        self.bx, self.by = COURT_W / 2.0, COURT_H / 2.0
        self.bvx = -BALL_SPEED0 * self.speed_mult
        self.bvy = 0.0
        self.py = COURT_H / 2.0 - PADDLE_H / 2.0
        self.oy = COURT_H / 2.0 - PADDLE_H / 2.0
        self.pts = 0
        self.opp = 0
        self.hits = 0
        self.frames = 0

    def state(self):
        return np.array([[self.bx / COURT_W, self.by / COURT_H,
                          self.bvx / MAX_SPEED, self.bvy / MAX_SPEED,
                          (self.py + PADDLE_H / 2.0) / COURT_H,
                          (self.bx - PADDLE_X) / COURT_W]])

    def step(self, move, opp_move=None):
        self.py += move * PADDLE_SPEED
        self.py = min(max(self.py, 0.0), COURT_H - PADDLE_H)
        if opp_move is None:
            target = self.by - PADDLE_H / 2.0
            d = target - self.oy
            self.oy += min(max(d, -OPP_SPEED), OPP_SPEED)
        else:
            self.oy += opp_move * PADDLE_SPEED
        self.oy = min(max(self.oy, 0.0), COURT_H - PADDLE_H)
        self.bx += self.bvx
        self.by += self.bvy
        if self.by < BALL_R:
            self.by = BALL_R
            self.bvy = abs(self.bvy)
        if self.by > COURT_H - BALL_R:
            self.by = COURT_H - BALL_R
            self.bvy = -abs(self.bvy)
        if (self.bvx < 0 and self.bx - BALL_R <= PADDLE_X + PADDLE_W
                and self.bx - BALL_R > PADDLE_X - 2.0
                and self.py <= self.by <= self.py + PADDLE_H):
            off = (self.by - (self.py + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            ang = np.clip(off, -1.0, 1.0) * MAX_ANGLE
            sp = min(abs(self.bvx) * (1.0 + SPEEDUP), MAX_SPEED * self.speed_mult)
            self.bvx = sp * np.cos(ang)
            self.bvy = sp * np.sin(ang)
            self.bx = PADDLE_X + PADDLE_W + BALL_R
            self.hits += 1
        if (self.bvx > 0 and self.bx + BALL_R >= OPP_X - PADDLE_W
                and self.bx + BALL_R < OPP_X + 2.0
                and self.oy <= self.by <= self.oy + PADDLE_H):
            off = (self.by - (self.oy + PADDLE_H / 2.0)) / (PADDLE_H / 2.0)
            ang = np.clip(off, -1.0, 1.0) * MAX_ANGLE
            sp = min(abs(self.bvx) * (1.0 + SPEEDUP), MAX_SPEED * self.speed_mult)
            self.bvx = -sp * np.cos(ang)
            self.bvy = sp * np.sin(ang)
            self.bx = OPP_X - PADDLE_W - BALL_R
        if self.bx < -BALL_R:
            self.opp += 1
            self._serve()
        if self.bx > COURT_W + BALL_R:
            self.pts += 1
            self._serve()
        self.frames += 1

    def _serve(self):
        self.bx, self.by = COURT_W / 2.0, COURT_H / 2.0
        self.bvx = -BALL_SPEED0 * self.speed_mult
        self.bvy = 0.0

    def over(self):
        return (self.pts >= WIN_SCORE or self.opp >= WIN_SCORE
                or self.frames >= MAX_FRAMES)


def play_one(genome, decide_fn, rng=None):
    """Play one full game with a brain. Returns (pts, opp, hits)."""
    g = SingleGame(rng)
    while not g.over():
        move = decide_fn(genome, g.state())[0]
        g.step(move)
    return g.pts, g.opp, g.hits