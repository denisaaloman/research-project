"""validation on real data: geant traffic matrices (totem xml) -> link states -> estimate -> solve -> replay.

usage: python real_data.py --tm data/geant/tm --topology data/geant/topology.csv [--src 1 --dst 7]
topology.csv: header src,dst,capacity_kbps,weight (one row per link, used in both directions).
"""
import argparse
import glob
import heapq
import os
import xml.etree.ElementTree as et
import numpy as np
import model as md


def local(tag):
    return tag.split("}")[-1]


def load_tms(folder):
    """reads one totem xml traffic matrix per file (sorted by name = time); returns (t, n, n) in kbps and node ids."""
    files = sorted(glob.glob(os.path.join(folder, "*.xml")))
    assert files, "no xml files in " + folder
    mats, ids = [], None
    for path in files:
        rows = {}
        for src in et.parse(path).getroot().iter():
            if local(src.tag) != "src":
                continue
            rows[src.get("id")] = {d.get("id"): float(d.text) for d in src if local(d.tag) == "dst"}
        if ids is None:
            allids = set(rows) | {d for r in rows.values() for d in r}
            ids = sorted(allids, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
        pos = {v: n for n, v in enumerate(ids)}
        mat = np.zeros((len(ids), len(ids)))
        for a, r in rows.items():
            for b, val in r.items():
                mat[pos[a], pos[b]] = val
        mats.append(mat)
    return np.array(mats), ids


def load_topology(path, ids):
    pos = {v: n for n, v in enumerate(ids)}
    links = []
    for line in open(path).read().strip().splitlines()[1:]:
        a, b, cap, w = line.split(",")[:4]
        links.append((pos[a], pos[b], float(cap), float(w)))
        links.append((pos[b], pos[a], float(cap), float(w)))
    return links


def shortest_routes(n, links):
    """link indices of the shortest path (by weight) between every ordered pair of nodes."""
    adj = {v: [] for v in range(n)}
    for li, (a, b, cap, w) in enumerate(links):
        adj[a].append((b, li, w))
    routes = {}
    for s in range(n):
        dist, prev = {s: 0.0}, {}
        heap = [(0.0, s)]
        while heap:
            dd, v = heapq.heappop(heap)
            if dd > dist.get(v, 1e18):
                continue
            for u, li, w in adj[v]:
                if dd + w < dist.get(u, 1e18) - 1e-12:
                    dist[u] = dd + w
                    prev[u] = (v, li)
                    heapq.heappush(heap, (dd + w, u))
        for d in range(n):
            if d == s or d not in dist:
                continue
            path, v = [], d
            while v != s:
                v, li = prev[v]
                path.append(li)
            routes[(s, d)] = path[::-1]
    return routes


def candidate_paths(n, links, s, d, k, max_links, max_hops=7):
    """k simple s-d paths in order of weight whose union has at most max_links links; None if not possible."""
    adj = {v: [] for v in range(n)}
    for li, (a, b, cap, w) in enumerate(links):
        adj[a].append((b, li, w))
    found = []

    def dfs(v, visited, path, wt):
        if v == d:
            found.append((wt, len(path), list(path)))
            return
        if len(path) >= max_hops:
            return
        for u, li, w in adj[v]:
            if u not in visited:
                visited.add(u)
                path.append(li)
                dfs(u, visited, path, wt + w)
                path.pop()
                visited.remove(u)

    dfs(s, {s}, [], 0.0)
    found.sort()
    chosen, union = [], set()
    for wt, hops, path in found:
        if len(union | set(path)) <= max_links:
            chosen.append(path)
            union |= set(path)
        if len(chosen) == k:
            break
    shared = any(set(a) & set(b) for i, a in enumerate(chosen) for b in chosen[i + 1:])
    return chosen if len(chosen) == k and shared else None


def main(tm, topology, src=None, dst=None, k=3, max_links=10, s0=2.0, quantile=0.75, split=0.7, gamma=0.95, out="results"):
    tms, ids = load_tms(tm)
    links = load_topology(topology, ids)
    n = len(ids)
    routes = shortest_routes(n, links)
    route = np.zeros((n * n, len(links)))
    for (a, b), path in routes.items():
        route[a * n + b, path] = 1.0
    load = tms.reshape(len(tms), n * n) @ route                        # kbps on every directed link, per interval
    util = load / np.array([l[2] for l in links])[None, :]
    if src is None:
        best = None
        for a in range(n):
            for b in range(n):
                if a != b and candidate_paths(n, links, a, b, k, max_links) and (best is None or tms[:, a, b].mean() > best[0]):
                    best = (tms[:, a, b].mean(), a, b)
        src, dst = best[1], best[2]
    else:
        src, dst = ids.index(str(src)), ids.index(str(dst))
    paths = candidate_paths(n, links, src, dst, k, max_links)
    assert paths, "no usable set of candidate paths for this pair"
    used = sorted({li for p in paths for li in p})
    local_id = {li: j for j, li in enumerate(used)}
    inst = md.make_instance([[local_id[li] for li in p] for p in paths])
    m = inst["m"]
    cut = int(split * len(tms))
    thr = np.quantile(util[:cut][:, used], quantile, axis=0)           # congested = above the link's own training quantile
    bits = (util[:, used] > thr[None, :]).astype(int)
    train, test = bits[:cut], bits[cut:]
    cnt = np.zeros((m, 2, 2))
    for e in range(m):
        np.add.at(cnt[e], (train[:-1, e], train[1:, e]), 1)
    p_hat = cnt[:, 0, 1] / np.maximum(cnt[:, 0].sum(axis=1), 1)
    q_hat = cnt[:, 1, 0] / np.maximum(cnt[:, 1].sum(axis=1), 1)
    pi_hat = p_hat / np.maximum(p_hat + q_hat, 1e-9)
    px = md.transition(m, p_hat, q_hat)
    s = md.penalty(inst, s0, "overlap")
    mu = md.initial_dist(inst, pi_hat)
    cc = inst["cc"]
    vv, popt = md.value_iteration(cc, px, s, gamma)
    thetas, th0s, alphas = np.arange(0, 3.01, 0.25), np.arange(0, 2.01, 0.5), [0.0, 0.5, 1.0, 1.5, 2.0]
    th = md.best_threshold(cc, px, s, gamma, mu, thetas)[1]
    th0, al = md.best_threshold(cc, px, s, gamma, mu, th0s, alphas)[1:]
    pols = {"optimal": popt, "never-switch": md.pol_never(inst["k"], inst["nx"]), "greedy": md.pol_greedy(cc,s),
            "threshold": md.pol_threshold(cc, s, th), "overlap-aware": md.pol_threshold(cc, s, th0, al)}
    weights = 2 ** np.arange(m)

    def replay(pol, trace):
        path, total, nsw = 0, 0.0, 0
        for row in trace:
            x = int(row @ weights)
            act = pol[path, x]
            total += s[path, act] + cc[act, x]
            nsw += act != path
            path = act
        return total / len(trace), nsw / len(trace)

    os.makedirs(out, exist_ok=True)
    res = {nm: replay(pol, test) for nm, pol in pols.items()}
    jopt = res["optimal"][0]
    with open(os.path.join(out, "real_replay.csv"), "w") as f:
        f.write("# pair %s->%s, %d paths, %d links, %d intervals (%d train / %d test), s0=%.2f, congested above the %.2f quantile\n" % (
            ids[src], ids[dst], k, m, len(tms), cut, len(tms) - cut, s0, quantile))
        f.write("# tuned: theta=%.2f theta0=%.2f alpha=%.2f\n" % (th, th0, al))
        f.write("policy,avg_cost_per_step,gap_pct_vs_optimal,switches_per_step\n")
        for nm, (c, sw) in res.items():
            f.write("%s,%.3f,%.2f,%.4f\n" % (nm, c, 100 * (c - jopt) / jopt, sw))
    print(open(os.path.join(out, "real_replay.csv")).read())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-links", type=int, default=10)
    ap.add_argument("--s0", type=float, default=2.0)
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.tm, a.topology, a.src, a.dst, a.k, a.max_links, a.s0, a.quantile, out=a.out)
