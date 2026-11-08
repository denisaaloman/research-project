"""automatic checks from section 3.9 (run with: python -m pytest)."""
import numpy as np
import model as md

gamma = 0.95
pi = 0.3
inst2 = md.make_instance([[0, 1], [2, 3]])
inst3 = md.make_instance([[0, 1, 2], [0, 1, 3], [4, 5, 6]])


def setup(inst, s0, rho, mod="const"):
    p, q = md.chain_params(pi, rho)
    px = md.transition(inst["m"], p, q)
    return px, md.penalty(inst, s0, mod), md.initial_dist(inst, pi), p, q


def test_zero_penalty_optimal_is_greedy():
    px, s, mu, _, _ = setup(inst2, 0.0, 0.5)
    vv, _ = md.value_iteration(inst2["cc"], px, s, gamma)
    jg = md.cost_of(md.evaluate(md.pol_greedy(inst2["cc"]), inst2["cc"], px, s, gamma), mu)
    assert abs(md.cost_of(vv, mu) - jg) < 1e-6


def test_huge_penalty_optimal_is_never_switch():
    px, s, mu, _, _ = setup(inst2, 1000.0, 0.5)
    vv, _ = md.value_iteration(inst2["cc"], px, s, gamma)
    jn = md.cost_of(md.evaluate(md.pol_never(2, inst2["nx"]), inst2["cc"], px, s, gamma), mu)
    assert abs(md.cost_of(vv, mu) - jn) < 1e-6


def test_value_iteration_equals_policy_iteration():
    for mod in ("const", "overlap"):
        px, s, mu, _, _ = setup(inst3, 1.0, 0.8, mod)
        v1, _ = md.value_iteration(inst3["cc"], px, s, gamma)
        v2, _ = md.policy_iteration(inst3["cc"], px, s, gamma)
        assert np.abs(v1 - v2).max() < 1e-6


def test_exact_matches_monte_carlo():
    px, s, mu, p, q = setup(inst3, 1.0, 0.8, "overlap")
    pol = md.pol_threshold(inst3["cc"], s, 1.0)
    je = md.cost_of(md.evaluate(pol, inst3["cc"], px, s, gamma), mu)
    i0 = int(np.argmax(mu.sum(axis=1)))
    mean, half, _, _ = md.simulate(pol, inst3, s, p, q, pi, i0, gamma, runs=4000)
    assert abs(mean - je) < 2 * half


def test_no_policy_beats_optimal():
    px, s, mu, _, _ = setup(inst3, 1.0, 0.5, "overlap")
    vv, _ = md.value_iteration(inst3["cc"], px, s, gamma)
    jo = md.cost_of(vv, mu)
    for th in np.arange(0, 3.1, 0.5):
        pol = md.pol_threshold(inst3["cc"], s, th)
        assert md.cost_of(md.evaluate(pol, inst3["cc"], px, s, gamma), mu) >= jo - 1e-6


def test_two_paths_optimal_has_threshold_structure():
    """for k = 2 and a constant penalty, the optimal decision depends on the gain only (a threshold)."""
    px, s, mu, _, _ = setup(inst2, 1.0, 0.8)
    cc = inst2["cc"]
    _, pol = md.value_iteration(cc, px, s, gamma)
    for i in range(2):
        gain = cc[i] - cc[1 - i]
        sw = pol[i] != i
        if sw.any() and (~sw).any():
            assert gain[sw].min() > gain[~sw].max() - 1e-9
