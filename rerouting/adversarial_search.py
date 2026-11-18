"""E5: adversarial search for the worst-case gap of the tuned static threshold (instance B).
usage: python adversarial_search.py [--model overlap|const] [--seeds 0 1 2]
Needs only model.py. Writes results/adversarial_<model>.csv."""
import os, argparse
import numpy as np
from multiprocessing import Pool
import model as md

gamma = 0.95
thetas = np.arange(0, 3.01, 0.25)
th0s = np.arange(0, 2.01, 0.5)
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
inst = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])   # P1={a,b,c} P2={a,b,d} P3={e,f,g}
m, cc = inst["m"], inst["cc"]
here = os.path.dirname(os.path.abspath(__file__))


def decode(v):
    """unit cube -> (pi_e in [0.05,0.95], valid rho_e, s0 in [0.25,4])."""
    pi = 0.05 + 0.9 * np.clip(v[:m], 0, 1)
    lo = np.maximum(1 - 1 / np.maximum(pi, 1 - pi), -0.95) + 1e-6   # smallest valid rho for this pi
    rho = lo + (0.98 - lo) * np.clip(v[m:2 * m], 0, 1)
    s0 = 0.25 + 3.75 * np.clip(v[2 * m], 0, 1)
    return pi, rho, s0


def gaps(v, model, with_oa=False):
    pi, rho, s0 = decode(v)
    p, q = md.chain_params(pi, rho)
    px = md.transition(m, p, q)
    s = md.penalty(inst, s0, model)
    mu = md.initial_dist(inst, pi)
    vv, _ = md.value_iteration(cc, px, s, gamma)
    jo = md.cost_of(vv, mu)
    jt, th, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
    r = {"th": (jt - jo) / jo, "theta": th}
    if with_oa:
        j2, t0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
        r.update(oa=(j2 - jo) / jo, theta0=t0, alpha=al)
    return r


def search(job):
    seed, model = job
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(250):                                   # random phase
        v = rng.random(2 * m + 1)
        pool.append((gaps(v, model)["th"], v))
    pool.sort(key=lambda x: -x[0])
    best = []
    for g, v in pool[:4]:                                  # hill climbing on the 4 worst
        sig = 0.15
        for it in range(120):
            w = np.clip(v + rng.normal(0, sig, v.shape) * (rng.random(v.shape) < 0.4), 0, 1)
            gw = gaps(w, model)["th"]
            if gw > g:
                g, v = gw, w
            if it % 40 == 39:
                sig *= 0.5
        best.append((g, v))
    best.sort(key=lambda x: -x[0])
    return seed, model, best[0], float(np.median([g for g, _ in pool]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="overlap", choices=["overlap", "const"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    with Pool() as P:
        res = P.map(search, [(sd, a.model) for sd in a.seeds])
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    out = os.path.join(here, "results", "adversarial_%s.csv" % a.model)
    with open(out, "w") as f:
        f.write("model,seed,gap_th_pct,gap_oa_pct,s0,theta,theta0,alpha,random_median_pct,"
                + ",".join("pi_%d" % e for e in range(m)) + "," + ",".join("rho_%d" % e for e in range(m)) + "\n")
        for seed, model, (g, v), med in res:
            pi, rho, s0 = decode(v)
            r = gaps(v, model, with_oa=True)
            f.write("%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s\n" % (
                model, seed, 100 * r["th"], 100 * r["oa"], s0, r["theta"], r["theta0"], r["alpha"], 100 * med,
                ",".join("%.3f" % x for x in pi), ",".join("%.3f" % x for x in rho)))
            print("%s seed %d: TH %.2f%%  OA %.2f%%  s0=%.2f  pi_c=%.2f pi_d=%.2f" % (
                model, seed, 100 * r["th"], 100 * r["oa"], s0, pi[2], pi[3]))
    print("written", out)
