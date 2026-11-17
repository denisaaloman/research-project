"""model of lab 5-8: rerouting with switching penalties (mdp, policies, exact evaluation)."""
import numpy as np


def make_instance(paths, c0=1.0, delta=1.0):
    """paths: list of lists of link ids. returns a dict with path costs for every congestion vector."""
    links = sorted({e for p in paths for e in p})
    idx = {e: n for n, e in enumerate(links)}
    m = len(links)
    nx = 2 ** m
    bits = (np.arange(nx)[:, None] >> np.arange(m)[None, :]) & 1
    k = len(paths)
    cc = np.zeros((k, nx))
    for j, p in enumerate(paths):
        cols = [idx[e] for e in p]
        cc[j] = c0 * len(p) + delta * bits[:, cols].sum(axis=1)
    jac = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a, b = set(paths[i]), set(paths[j])
            jac[i, j] = len(a & b) / len(a | b)
    return {"k": k, "m": m, "nx": nx, "bits": bits, "cc": cc, "jac": jac, "paths": paths}


def chain_params(pi, rho):
    """two-state chain with stationary congestion probability pi and second eigenvalue rho."""
    pi = np.asarray(pi, dtype=float)
    rho = np.asarray(rho, dtype=float)
    p = pi * (1 - rho)
    q = (1 - pi) * (1 - rho)
    assert (p >= 0).all() and (p <= 1).all() and (q >= 0).all() and (q <= 1).all(), "rho out of range for this pi"
    return p, q


def transition(m, p, q):
    """transition matrix of the m independent link chains (bit e of x is link e)."""
    p = np.broadcast_to(np.asarray(p, dtype=float), (m,))
    q = np.broadcast_to(np.asarray(q, dtype=float), (m,))
    px = np.ones((1, 1))
    for e in range(m):
        one = np.array([[1 - p[e], p[e]], [q[e], 1 - q[e]]])
        px = np.kron(one, px)
    return px


def stationary(inst, pi):
    pi = np.broadcast_to(np.asarray(pi, dtype=float), (inst["m"],))
    return np.where(inst["bits"] == 1, pi[None, :], 1 - pi[None, :]).prod(axis=1)


def penalty(inst, s0, model):
    k = inst["k"]
    if model == "const":
        return s0 * (1 - np.eye(k))
    return s0 * (1 - inst["jac"])


def initial_dist(inst, pi):
    """initial path = path with the lowest expected cost (ties: lowest index); x is stationary."""
    lens = np.array([len(p) for p in inst["paths"]])
    i0 = int(np.argmin(lens))
    mu = np.zeros((inst["k"], inst["nx"]))
    mu[i0] = stationary(inst, pi)
    return mu


def value_iteration(cc, px, s, gamma, eps=1e-9):
    """returns the optimal value function and an optimal policy (ties: keep the current path)."""
    k, nx = cc.shape
    vv = np.zeros((k, nx))
    tol = eps * (1 - gamma) / (2 * gamma)
    while True:
        ww = cc + gamma * vv @ px.T
        new = (s[:, :, None] + ww[None, :, :]).min(axis=1)
        done = np.abs(new - vv).max() < tol
        vv = new
        if done:
            break
    ww = cc + gamma * vv @ px.T
    q = s[:, :, None] + ww[None, :, :]
    for i in range(k):
        q[i, i, :] -= 1e-9
    return vv, q.argmin(axis=1)


def evaluate(pol, cc, px, s, gamma):
    """exact discounted cost of a stationary policy: v = (i - gamma p) ^ -1 c."""
    k, nx = cc.shape
    ii, xx = np.meshgrid(np.arange(k), np.arange(nx), indexing="ij")
    jj = pol
    n = k * nx
    pp = np.zeros((n, n))
    rows = (ii * nx + xx).ravel()
    cols = jj.ravel()[:, None] * nx + np.arange(nx)[None, :]
    pp[rows[:, None], cols] = px[xx.ravel()]
    cpi = (s[ii, jj] + cc[jj, xx]).ravel()
    return np.linalg.solve(np.eye(n) - gamma * pp, cpi).reshape(k, nx)

def _jstar(cc, s):
    return np.argmin(cc[None] + 1e-9 * s[:, :, None], axis=1)   # (k, nx)

def policy_iteration(cc, px, s, gamma):
    k, nx = cc.shape
    pol = np.tile(np.arange(k)[:, None], (1, nx))
    ii, xx = np.meshgrid(np.arange(k), np.arange(nx), indexing="ij")
    for _ in range(1000):
        vv = evaluate(pol, cc, px, s, gamma)
        ww = cc + gamma * vv @ px.T
        q = s[:, :, None] + ww[None, :, :]
        qcur = q[ii, pol, xx]
        best = q.argmin(axis=1)
        qbest = q[ii, best, xx]
        new = np.where(qbest < qcur - 1e-9, best, pol)
        if (new == pol).all():
            return vv, pol
        pol = new
    return vv, pol


def pol_never(k, nx):
    return np.tile(np.arange(k)[:, None], (1, nx))


def pol_greedy(cc, s):
    k, nx = cc.shape
    return _jstar(cc, s)


def pol_threshold(cc, s, theta, alpha=0.0):
    k, nx = cc.shape
    ii = np.arange(k)[:, None]
    jstar = _jstar(cc, s)
    gain = cc - cc.min(axis=0)
    switching_penalty = s[ii, jstar]
    sw = gain > theta + alpha * switching_penalty
    return np.where(sw, jstar, ii)


def cost_of(vv, mu):
    return float((vv * mu).sum())


def best_threshold(cc, px, s, gamma, mu, thetas, alphas=(0.0,)):
    """tunes (theta, alpha) on a grid; returns the best cost and parameters."""
    best = (np.inf, None, None)
    for th in thetas:
        for al in alphas:
            pol = pol_threshold(cc, s, th, al)
            j = cost_of(evaluate(pol, cc, px, s, gamma), mu)
            if j < best[0] - 1e-12:
                best = (j, th, al)
    return best


def simulate(pol, inst, s, p, q, pi, i0, gamma, runs=1000, steps=300, window=5, seed=0):
    """monte carlo: discounted cost, switch rate and oscillation rate of a policy."""
    rng = np.random.default_rng(seed)
    m = inst["m"]
    cc = inst["cc"]
    weights = 2 ** np.arange(m)
    bits = (rng.random((runs, m)) < pi).astype(int)
    path = np.full(runs, i0)
    prev = np.full(runs, -1)
    last_t = np.full(runs, -10 ** 6)
    total = np.zeros(runs)
    nsw = np.zeros(runs)
    nosc = np.zeros(runs)
    for t in range(steps):
        x = bits @ weights
        act = pol[path, x]
        total += gamma ** t * (s[path, act] + cc[act, x])
        sw = act != path
        osc = sw & (act == prev) & (t - last_t <= window)
        nsw += sw
        nosc += osc
        prev = np.where(sw, path, prev)
        last_t = np.where(sw, t, last_t)
        path = act
        u = rng.random((runs, m))
        bits = np.where(bits == 0, (u < p).astype(int), (u >= q).astype(int))
    mean = total.mean()
    half = 1.96 * total.std(ddof=1) / np.sqrt(runs)
    return mean, half, nsw.sum() / (runs * steps), (nosc.sum() / max(nsw.sum(), 1))
