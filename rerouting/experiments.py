"""case study: runs the experiments on the small instances and writes results/ (csv, figures)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as md

gamma = 0.95
pi = 0.3
eps = 0.05
s0s = [0.25, 0.5, 1.0, 2.0, 4.0]
rhos = [-0.4, 0.0, 0.5, 0.8, 0.95]
thetas = np.arange(0, 3.01, 0.25)
th0s = np.arange(0, 2.01, 0.5)
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
out = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out, exist_ok=True)

inst_a = md.make_instance([[0, 1], [2, 3]])                       # 2 disjoint paths
inst_b = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])      # 3 paths, two of them share 2 links


def scenario(inst, s0, rho, model, with_oa):
    p, q = md.chain_params(pi, rho)
    px = md.transition(inst["m"], p, q)
    s = md.penalty(inst, s0, model)
    mu = md.initial_dist(inst, pi)
    cc = inst["cc"]
    vv, popt = md.value_iteration(cc, px, s, gamma)
    jopt = md.cost_of(vv, mu)
    res = {"opt": jopt}
    res["ns"] = md.cost_of(md.evaluate(md.pol_never(inst["k"], inst["nx"]), cc, px, s, gamma), mu)
    res["gr"] = md.cost_of(md.evaluate(md.pol_greedy(cc, s), cc, px, s, gamma), mu)
    jt, th, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
    res["th"], res["theta"] = jt, th
    if with_oa:
        jo, th0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
        res["oa"], res["theta0"], res["alpha"] = jo, th0, al
    for key in ("ns", "gr", "th", "oa"):
        if key in res:
            res["gap_" + key] = (res[key] - jopt) / jopt
    return res, (px, s, mu, popt, p, q)


rows = []
for name, inst in (("a", inst_a), ("b", inst_b)):
    for rho in rhos:
        for s0 in s0s:
            for model in ("const", "overlap"):
                if name == "a" and model == "overlap":
                    continue
                res, _ = scenario(inst, s0, rho, model, with_oa=(name == "b" and model == "overlap"))
                rows.append((name, model, rho, s0, res))
                print(name, model, "rho=%.2f s0=%.2f" % (rho, s0),
                      " ".join("%s=%.2f%%" % (k[4:], 100 * v) for k, v in res.items() if k.startswith("gap_")))

with open(os.path.join(out, "gaps.csv"), "w") as f:
    f.write("instance,model,rho,s0,j_opt,gap_ns,gap_gr,gap_th,theta,gap_oa,theta0,alpha\n")
    for name, model, rho, s0, r in rows:
        f.write("%s,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.2f,%s,%s,%s\n" % (
            name, model, rho, s0, r["opt"], r["gap_ns"], r["gap_gr"], r["gap_th"], r["theta"],
            "%.4f" % r["gap_oa"] if "gap_oa" in r else "", "%.2f" % r["theta0"] if "theta0" in r else "",
            "%.2f" % r["alpha"] if "alpha" in r else ""))


def grid(name, model, key):
    g = np.zeros((len(rhos), len(s0s)))
    for n, rho in enumerate(rhos):
        for m_, s0 in enumerate(s0s):
            for nm, mo, r_, s_, res in rows:
                if nm == name and mo == model and r_ == rho and s_ == s0:
                    g[n, m_] = res[key]
    return g


def heat(ax, g, title):
    ax.imshow(100 * g, origin="lower", cmap="Reds", vmin=0, vmax=max(10, 100 * g.max()))
    for n in range(g.shape[0]):
        for m_ in range(g.shape[1]):
            ax.text(m_, n, "%.1f" % (100 * g[n, m_]), ha="center", va="center", fontsize=8,
                    fontweight="bold" if g[n, m_] > eps else "normal")
    ax.set_xticks(range(len(s0s)))
    ax.set_xticklabels(s0s)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels(rhos)
    ax.set_xlabel("penalty s0 / delta")
    ax.set_title(title, fontsize=9)


fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
heat(axs[0], grid("b", "const", "gap_th"), "static threshold, constant penalty")
heat(axs[1], grid("b", "overlap", "gap_th"), "static threshold, overlap penalty")
heat(axs[2], grid("b", "overlap", "gap_oa"), "overlap-aware threshold, overlap penalty")
axs[0].set_ylabel("persistence rho")
plt.tight_layout()
plt.savefig(os.path.join(out, "failure_maps.png"), dpi=200)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
heat(ax, grid("a", "const", "gap_th"), "2 disjoint paths, constant penalty: gap of threshold (%)")
ax.set_ylabel("persistence rho")
plt.tight_layout()
plt.savefig(os.path.join(out, "baseline_gap.png"), dpi=200)

# showcase: worst scenario of the static threshold under the overlap penalty
worst = max((r for r in rows if r[0] == "b" and r[1] == "overlap"), key=lambda r: r[4]["gap_th"])
_, _, rho, s0, _ = worst
res, (px, s, mu, popt, p, q) = scenario(inst_b, s0, rho, "overlap", True)
cc = inst_b["cc"]
i0 = int(np.argmax(mu.sum(axis=1)))
pols = {"opt": popt, "never-switch": md.pol_never(3, inst_b["nx"]), "greedy": md.pol_greedy(cc,s),
        "threshold": md.pol_threshold(cc, s, res["theta"]),
        "overlap-aware": md.pol_threshold(cc, s, res["theta0"], res["alpha"])}
with open(os.path.join(out, "showcase.csv"), "w") as f:
    f.write("# worst case: rho=%.2f s0=%.2f theta=%.2f theta0=%.2f alpha=%.2f\n" % (rho, s0, res["theta"], res["theta0"], res["alpha"]))
    f.write("policy,j_exact,gap_pct,j_mc,mc_halfwidth,switch_rate,oscillation_rate\n")
    for nm, pol in pols.items():
        je = md.cost_of(md.evaluate(pol, cc, px, s, gamma), mu)
        mean, half, swr, osr = md.simulate(pol, inst_b, s, p, q, pi, i0, gamma)
        gap = 100 * (je - res["opt"]) / res["opt"]
        f.write("%s,%.3f,%.2f,%.3f,%.3f,%.4f,%.3f\n" % (nm, je, gap, mean, half, swr, osr))
        print("showcase", nm, "J=%.3f gap=%.2f%% mc=%.3f+-%.3f sw=%.4f osc=%.3f" % (je, gap, mean, half, swr, osr))
print("worst case:", "rho", rho, "s0", s0, res)
