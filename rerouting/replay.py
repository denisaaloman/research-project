"""trace-driven evaluation: estimate the chains from training traces, solve and tune, replay on test traces."""
import os
import numpy as np
import model as md

here = os.path.dirname(__file__)
gamma = 0.95
s0 = 2.0
thetas = np.arange(0, 3.01, 0.25)
th0s = np.arange(0, 2.01, 0.5)
alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
inst = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])
m = inst["m"]

raw = np.loadtxt(os.path.join(here, "data", "traces.csv"), delimiter=",", skiprows=1, dtype=int)
traces = [raw[raw[:, 0] == r][:, 2:] for r in range(raw[:, 0].max() + 1)]
train, test = traces[:5], traces[5:]

n = np.zeros((m, 2, 2))                      # transition counts per link, from the training traces only
for tr in train:
    for e in range(m):
        np.add.at(n[e], (tr[:-1, e], tr[1:, e]), 1)
p_hat = n[:, 0, 1] / n[:, 0].sum(axis=1)
q_hat = n[:, 1, 0] / n[:, 1].sum(axis=1)
pi_hat = p_hat / (p_hat + q_hat)
true = np.loadtxt(os.path.join(here, "data", "true_parameters.txt"), delimiter=",", skiprows=1)
print("max abs error of p: %.3f, of q: %.3f" % (np.abs(p_hat - true[:, 3]).max(), np.abs(q_hat - true[:, 4]).max()))

px = md.transition(m, p_hat, q_hat)
s = md.penalty(inst, s0, "overlap")
mu = md.initial_dist(inst, pi_hat)
cc = inst["cc"]
vv, popt = md.value_iteration(cc, px, s, gamma)
jt, th, _ = md.best_threshold(cc, px, s, gamma, mu, thetas)
jo, th0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)
pols = {"optimal": popt, "never-switch": md.pol_never(3, inst["nx"]), "greedy": md.pol_greedy(cc),
        "threshold": md.pol_threshold(cc, s, th), "overlap-aware": md.pol_threshold(cc, s, th0, al)}
print("tuned on training data: theta=%.2f theta0=%.2f alpha=%.2f" % (th, th0, al))

weights = 2 ** np.arange(m)


def replay(pol, bits, i0=0):
    """average cost per step and switches per step of a policy on one trace."""
    path, total, nsw = i0, 0.0, 0
    for row in bits:
        x = int(row @ weights)
        act = pol[path, x]
        total += s[path, act] + cc[act, x]
        nsw += act != path
        path = act
    return total / len(bits), nsw / len(bits)


res = {nm: [replay(pol, tr) for tr in test] for nm, pol in pols.items()}
opt = np.array([r[0] for r in res["optimal"]])
with open(os.path.join(here, "results", "replay.csv"), "w") as f:
    f.write("policy,avg_cost_per_step,std_err,gap_pct_vs_optimal,switches_per_step\n")
    for nm, r in res.items():
        c = np.array([x[0] for x in r])
        sw = np.mean([x[1] for x in r])
        f.write("%s,%.3f,%.3f,%.2f,%.4f\n" % (nm, c.mean(), c.std(ddof=1) / np.sqrt(len(c)), 100 * (c.mean() - opt.mean()) / opt.mean(), sw))
print(open(os.path.join(here, "results", "replay.csv")).read())
with open(os.path.join(here, "results", "estimation.csv"), "w") as f:
    f.write("link,p_true,p_hat,q_true,q_hat\n")
    for e in range(m):
        f.write("%d,%.3f,%.3f,%.3f,%.3f\n" % (e, true[e, 3], p_hat[e], true[e, 4], q_hat[e]))
