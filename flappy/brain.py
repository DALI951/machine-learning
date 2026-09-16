"""Tiny neural network for the Flappy learner.

4 inputs (bird y, bird vy, distance to next pipe, gap center)
-> 8 hidden (tanh) -> 1 output (flap if > 0.5).

The genome is just the flattened weights. Nothing about flapping
strategy lives here — only the matrix math.
"""
import numpy as np

N_IN = 4
N_HID = 8
N_OUT = 1


def genome_size():
    return N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT


def forward(w, states):
    """states: (n, N_IN) -> out: (n, N_OUT)."""
    W1 = w[:N_IN * N_HID].reshape(N_IN, N_HID)
    b1 = w[N_IN * N_HID:N_IN * N_HID + N_HID]
    W2 = w[N_IN * N_HID + N_HID:N_IN * N_HID + N_HID + N_HID * N_OUT].reshape(N_HID, N_OUT)
    b2 = w[-N_OUT:]
    h = np.tanh(states @ W1 + b1)
    return h @ W2 + b2


def forward_batch(pop, states):
    """pop: (P, genome_size), states: (P, N_IN) -> out: (P, N_OUT)."""
    P = pop.shape[0]
    W1 = pop[:, :N_IN * N_HID].reshape(P, N_IN, N_HID)
    b1 = pop[:, N_IN * N_HID:N_IN * N_HID + N_HID]
    W2 = pop[:, N_IN * N_HID + N_HID:N_IN * N_HID + N_HID + N_HID * N_OUT].reshape(P, N_HID, N_OUT)
    b2 = pop[:, -N_OUT:]
    h = np.tanh(np.einsum("pi,pio->po", states, W1) + b1)
    return np.einsum("po,poq->pq", h, W2) + b2


def decide(w, states):
    """Single decision: 1.0 = flap, 0.0 = don't. states: (1, N_IN)."""
    out = forward(w, states)
    return np.array([1.0 if out[0, 0] > 0.5 else 0.0])


def decide_batch(pop, states):
    """Whole population at once: (P, 1) of 0.0/1.0."""
    out = forward_batch(pop, states)
    return (out > 0.5).astype(float)