"""E6: failure map of the tuned thresholds over (pi_c, s0/delta), instance B.
Link c is only in P1, link d only in P2; d has pi_d = 0.1, all other links pi = 0.3, rho = 0.5.
usage: python failure_map.py [--model overlap|const]
Writes results/failure_map_<model>.csv and, for the overlap penalty, results/failure_region_pi_s0.png."""
import os, argparse
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as md

gamma = 0.95
thetas = np.arange(0, 3.01, 0.25)
th0s = np.arange(0, 2.01, 0.5)
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
inst = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])
m, cc = inst["m"], inst["cc"]
pcs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
s0s = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
here = os.path.dirname(os.path.abspath(__file__))


def cell(job):
    pc, s0, model = job
    pi = np.full(m, 0.3); pi[2] = pc; pi[3] = 0.1
    p, q = md.chain_params(pi, np.full(m, 0.5))
    px = md.transition(m, p, q)
    s = md.penalty(inst, s0, model)
    mu = md.initial_dist(inst, pi)
    vv, _ = md.value_iteration(cc, px, s, gamma)
    jo = md.cost_of(vv, mu)
    jt, _, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
    j2, _, _ = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
    return 100 * (jt - jo) / jo, 100 * (j2 - jo) / jo


def heat(ax, g, title):
    ax.imshow(g, origin="lower", cmap="Reds", vmin=0, vmax=max(8, g.max()), aspect="auto")
    X, Y = np.meshgrid(range(len(pcs)), range(len(s0s)))
    if g.max() > 1:
        cs = ax.contour(X, Y, g, levels=[l for l in (1, 2, 5) if l < g.max()], colors="k",
                        linewidths=[0.8, 1.2, 2.2][:sum(l < g.max() for l in (1, 2, 5))],
                        linestyles=["dotted", "dashed", "solid"][:sum(l < g.max() for l in (1, 2, 5))])
        ax.clabel(cs, fmt=lambda v: "%d%%" % v, fontsize=7)
    for i in range(len(s0s)):
        for j in range(len(pcs)):
            ax.text(j, i, "%.1f" % g[i, j], ha="center", va="center", fontsize=7,
                    fontweight="bold" if g[i, j] > 5 else "normal")
    ax.set_xticks(range(len(pcs))); ax.set_xticklabels(pcs, fontsize=8)
    ax.set_yticks(range(len(s0s))); ax.set_yticklabels(s0s, fontsize=8)
    ax.set_xlabel(r"congestion probability $\pi_c$ of link c (only in $P_1$)")
    ax.set_ylabel(r"penalty $s_0/\delta$")
    ax.set_title(title, fontsize=9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="overlap", choices=["overlap", "const"])
    a = ap.parse_args()
    with Pool() as P:
        r = P.map(cell, [(pc, s0, a.model) for s0 in s0s for pc in pcs])
    gt = np.array([x[0] for x in r]).reshape(len(s0s), len(pcs))
    go = np.array([x[1] for x in r]).reshape(len(s0s), len(pcs))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "failure_map_%s.csv" % a.model), "w") as f:
        f.write("model,pi_c,s0,gap_th_pct,gap_oa_pct\n")
        for i, s0 in enumerate(s0s):
            for j, pc in enumerate(pcs):
                f.write("%s,%.2f,%.2f,%.3f,%.3f\n" % (a.model, pc, s0, gt[i, j], go[i, j]))
    print("%s: TH max %.2f%%, >5%%: %d/%d, >2%%: %d | OA max %.2f%%, >5%%: %d, >2%%: %d" % (
        a.model, gt.max(), (gt > 5).sum(), gt.size, (gt > 2).sum(), go.max(), (go > 5).sum(), (go > 2).sum()))
    if a.model == "overlap":
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        heat(axs[0], gt, "Static threshold")
        heat(axs[1], go, "Overlap-aware threshold")
        axs[1].set_ylabel("")
        fig.suptitle(r"Gap to the optimum (%), overlap penalty; $\pi_d = 0.1$, other links $\pi = 0.3$, $\rho = 0.5$"
                     "\ncontours: 1% (dotted), 2% (dashed), 5% (solid)", fontsize=9)
        fig.tight_layout()
        out = os.path.join(here, "results", "failure_region_pi_s0.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("written", out)
