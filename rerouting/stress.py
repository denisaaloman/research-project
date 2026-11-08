"""stress test: links with different persistence and congestion probability (random scenarios)."""
import os
import numpy as np
import model as md

gamma = 0.95
thetas = np.arange(0, 3.01, 0.25)
th0s = np.arange(0, 2.01, 0.5)
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
inst = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])
rng = np.random.default_rng(1)
rows = []
for n in range(100):
    pi = rng.uniform(0.1, 0.6, inst["m"])
    rho = rng.uniform(-0.1, 0.95, inst["m"])
    s0 = rng.choice([0.5, 1.0, 2.0])
    p, q = md.chain_params(pi, rho)
    px = md.transition(inst["m"], p, q)
    mu = md.initial_dist(inst, pi)
    cc = inst["cc"]
    out = {"n": n, "s0": s0}
    for model in ("const", "overlap"):
        s = md.penalty(inst, s0, model)
        vv, _ = md.value_iteration(cc, px, s, gamma)
        jo = md.cost_of(vv, mu)
        jt, _, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
        out["th_" + model] = (jt - jo) / jo
        if model == "overlap":
            jv, th0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
            out["oa"] = (jv - jo) / jo
            out["pi"], out["rho"] = pi, rho
    rows.append(out)
res = os.path.join(os.path.dirname(__file__), "results")
with open(os.path.join(res, "stress.csv"), "w") as f:
    f.write("n,s0,gap_th_const,gap_th_overlap,gap_oa_overlap\n")
    for r in rows:
        f.write("%d,%.2f,%.4f,%.4f,%.4f\n" % (r["n"], r["s0"], r["th_const"], r["th_overlap"], r["oa"]))
a = lambda k: np.array([r[k] for r in rows])
for k in ("th_const", "th_overlap", "oa"):
    g = a(k)
    print(k, "mean %.2f%% median %.2f%% max %.2f%% above5%%: %d/100" % (100 * g.mean(), 100 * np.median(g), 100 * g.max(), (g > 0.05).sum()))
w = max(rows, key=lambda r: r["th_overlap"])
print("worst", w["n"], w["s0"], "th %.2f%% oa %.2f%%" % (100 * w["th_overlap"], 100 * w["oa"]), "pi", np.round(w["pi"], 2), "rho", np.round(w["rho"], 2))
print("oa better than th (>0.1pp):", int((a("th_overlap") - a("oa") > 0.001).sum()))

# detailed comparison on the worst scenario of the static threshold (overlap penalty)
pi, rho, s0 = w["pi"], w["rho"], w["s0"]
p, q = md.chain_params(pi, rho)
px = md.transition(inst["m"], p, q)
mu = md.initial_dist(inst, pi)
cc = inst["cc"]
s = md.penalty(inst, s0, "overlap")
vv, popt = md.value_iteration(cc, px, s, gamma)
jo = md.cost_of(vv, mu)
jt, th, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
jv, th0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
pols = {"optimal": popt, "never-switch": md.pol_never(3, inst["nx"]), "greedy": md.pol_greedy(cc),
        "threshold": md.pol_threshold(cc, s, th), "overlap-aware": md.pol_threshold(cc, s, th0, al)}
i0 = int(np.argmax(mu.sum(axis=1)))
with open(os.path.join(res, "stress_showcase.csv"), "w") as f:
    f.write("# scenario %d: s0=%.2f theta=%.2f theta0=%.2f alpha=%.2f\n" % (w["n"], s0, th, th0, al))
    f.write("policy,j_exact,gap_pct,j_mc,mc_halfwidth,switch_rate,oscillation_rate\n")
    for nm, pol in pols.items():
        je = md.cost_of(md.evaluate(pol, cc, px, s, gamma), mu)
        mean, half, swr, osr = md.simulate(pol, inst, s, p, q, pi, i0, gamma)
        f.write("%s,%.3f,%.2f,%.3f,%.3f,%.4f,%.3f\n" % (nm, je, 100 * (je - jo) / jo, mean, half, swr, osr))
print(open(os.path.join(res, "stress_showcase.csv")).read())
