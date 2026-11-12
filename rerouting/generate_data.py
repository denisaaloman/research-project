"""creates the initial dataset: congestion traces of the 7 links of instance b (heterogeneous links)."""
import os
import numpy as np
import model as md

here = os.path.dirname(__file__)
pi = np.array([0.43, 0.21, 0.33, 0.10, 0.44, 0.53, 0.53])      # congestion probability of each link
rho = np.array([0.82, 0.11, 0.93, 0.48, 0.09, 0.00, -0.03])    # persistence of each link
p, q = md.chain_params(pi, rho)
runs, steps, m = 10, 5000, 7
rng = np.random.default_rng(7)
with open(os.path.join(here, "data", "traces.csv"), "w") as f:
    f.write("run,t," + ",".join("l%d" % e for e in range(m)) + "\n")
    for r in range(runs):
        bits = (rng.random(m) < pi).astype(int)
        for t in range(steps):
            f.write("%d,%d,%s\n" % (r, t, ",".join(str(b) for b in bits)))
            u = rng.random(m)
            bits = np.where(bits == 0, (u < p).astype(int), (u >= q).astype(int))
with open(os.path.join(here, "data", "true_parameters.txt"), "w") as f:
    f.write("link,pi,rho,p,q\n")
    for e in range(m):
        f.write("%d,%.2f,%.2f,%.4f,%.4f\n" % (e, pi[e], rho[e], p[e], q[e]))
print("written data/traces.csv:", runs, "runs x", steps, "steps x", m, "links")
