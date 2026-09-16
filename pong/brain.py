"""Tiny feed-forward neural network (numpy). Genome = flat weight vector.

Nothing about Pong strategy lives here — just matrix math. The weights are
evolved; the network learns what to do with them.
"""
import numpy as np

N_IN = 6
N_HID = 8
N_OUT = 2


def genome_size():
    return N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT


def unpack(w):
    i = 0
    W1 = w[i:i + N_IN * N_HID].reshape(N_IN, N_HID)
    i += N_IN * N_HID
    b1 = w[i:i + N_HID]
    i += N_HID
    W2 = w[i:i + N_HID * N_OUT].reshape(N_HID, N_OUT)
    i += N_HID * N_OUT
    b2 = w[i:i + N_OUT]
    return W1, b1, W2, b2


def forward(w, states):
    """states: (n, N_IN) -> out: (n, N_OUT)."""
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(states @ W1 + b1)
    return h @ W2 + b2


def decide(w, states, deadzone=0.1):
    """Two outputs (up, down). Move up if up > down + deadzone, etc."""
    out = forward(w, states)
    up = out[:, 0]
    dn = out[:, 1]
    moves = np.zeros(len(states), dtype=int)
    moves[up > dn + deadzone] = -1
    moves[dn > up + deadzone] = 1
    return moves


def forward_batch(pop, states):
    """pop: (P, genome_size), states: (P, N_IN) -> out: (P, N_OUT)."""
    P = pop.shape[0]
    W1 = pop[:, :N_IN * N_HID].reshape(P, N_IN, N_HID)
    b1 = pop[:, N_IN * N_HID:N_IN * N_HID + N_HID]
    W2 = pop[:, N_IN * N_HID + N_HID:N_IN * N_HID + N_HID + N_HID * N_OUT].reshape(P, N_HID, N_OUT)
    b2 = pop[:, -N_OUT:]
    h = np.tanh(np.einsum("pi,pio->po", states, W1) + b1)
    return np.einsum("po,poq->pq", h, W2) + b2


def decide_batch(pop, states, deadzone=0.1):
    out = forward_batch(pop, states)
    up = out[:, 0]
    dn = out[:, 1]
    moves = np.zeros(len(states), dtype=int)
    moves[up > dn + deadzone] = -1
    moves[dn > up + deadzone] = 1
    return moves