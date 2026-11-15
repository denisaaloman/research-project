"""builds a small fake dataset in the same format as the geant one (totem xml + topology.csv), only to test the code."""
import os
import numpy as np


def make(folder, nodes=8, steps=600, seed=3):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(folder, "tm"), exist_ok=True)
    edges = [(i, (i + 1) % nodes) for i in range(nodes)] + [(0, 4), (2, 6), (1, 5)]
    with open(os.path.join(folder, "topology.csv"), "w") as f:
        f.write("src,dst,capacity_kbps,weight\n")
        for a, b in edges:
            f.write("%d,%d,%d,%d\n" % (a, b, 10000000, 1 + (a + b) % 3))
    base = rng.uniform(1e5, 1e6, (nodes, nodes))
    for t in range(steps):
        day = 1 + 0.5 * np.sin(2 * np.pi * t / 96)
        mat = base * day * rng.lognormal(0, 0.3, (nodes, nodes))
        with open(os.path.join(folder, "tm", "IntraTM-%05d.xml" % t), "w") as f:
            f.write('<?xml version="1.0"?>\n<TrafficMatrixFile><info><date>2005-01-01T00:00:00</date></info><IntraTM ASID="1">\n')
            for a in range(nodes):
                f.write('<src id="%d">' % a + "".join('<dst id="%d">%.1f</dst>' % (b, mat[a, b]) for b in range(nodes) if b != a) + "</src>\n")
            f.write("</IntraTM></TrafficMatrixFile>\n")
