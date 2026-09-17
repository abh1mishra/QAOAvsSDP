"""Accuracy tests for src/qoaa_approach.py.   Run:  python test/test_qoaa_approach.py"""

import sys
from functools import reduce
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qoaa_approach import (  # noqa: E402  (import needs src/ on sys.path first)
    cost,
    define_graph,
    greedy_iterative,
    hamqaoa_circuit,
    heisenberg_hamiltonian,
    measured_energy,
    n_params,
    random_restarts,
)

EXACT = {"K33": -15.0, "C6": -2 * (2 + np.sqrt(13))}  # closed forms of λ_min(H)

# paper's Table II: exact ground-state parameters of the N = 6 ring, rows = layers (α, β, γ, δ)
TABLE_II_C6 = np.array([
    [0.4440, 0.5794, -1.5708, 0],
    [-0.8367, -0.7445, -0.7854, 0.7854],
    [1.4894, -1.2421, 1.1202, -1.5686],
    [1.5708, 1.0088, 1.0335, -1.9968],
    [-0.4696, 0, -0.0025, -1.3673],
    [-1.0117, -0.7854, -1.5708, 0.3109],
    [-0.1558, 0, -0.7854, 0.7854],
])


def exact_ground_energy(graph):
    n, edges, _ = define_graph(graph)
    return np.linalg.eigvalsh(heisenberg_hamiltonian(n, edges).to_matrix()).min()


def numpy_energy(graph, theta):
    """HamQAOA energy from an independent dense statevector simulation (no Qiskit gates)."""
    n, edges, s = define_graph(graph)
    zs = lambda b, q: 1 - 2 * ((b >> q) & 1)  # little-endian, as in Qiskit
    idx = range(2**n)
    A = np.array([sum(w * zs(b, u) * zs(b, v) for u, v, w in edges) for b in idx])
    C = np.array([sum(zs(b, q) for q in range(n)) for b in idx])
    D = np.array([sum(s[q] * zs(b, q) for q in range(n)) for b in idx])  # in the X basis
    Hn = reduce(np.kron, [np.array([[1, 1], [1, -1]]) / np.sqrt(2)] * n)
    psi = Hn[:, sum(((1 + s[q]) // 2) << q for q in range(n))].astype(complex)  # X eigenvalue -s_q
    for alpha, beta, gamma, delta in theta.reshape(-1, 4):
        psi = np.exp(-1j * alpha * A) * psi
        psi = Hn @ (np.exp(-1j * beta * C) * (Hn @ psi))
        psi = np.exp(-1j * gamma * C) * psi
        psi = Hn @ (np.exp(-1j * delta * D) * (Hn @ psi))
    H = heisenberg_hamiltonian(n, edges).to_matrix()
    return np.real(psi.conj() @ H @ psi)


def test_exact_diagonalization():
    for graph, value in EXACT.items():
        assert abs(exact_ground_energy(graph) - value) < 1e-10, graph


def test_circuit_against_numpy():
    rng = np.random.default_rng(0)
    for graph in EXACT:
        n, edges, s = define_graph(graph)
        for p in (1, 3):
            qc, H = hamqaoa_circuit(n, edges, s, p), heisenberg_hamiltonian(n, edges)
            for _ in range(3):
                theta = rng.uniform(-np.pi, np.pi, n_params(p))
                assert abs(cost(qc, H, theta) - numpy_energy(graph, theta)) < 1e-10, (graph, p)


def test_table_II_parameters():
    n, edges, s = define_graph("C6")
    qc = hamqaoa_circuit(n, edges, s, 7)
    E = cost(qc, heisenberg_hamiltonian(n, edges), TABLE_II_C6.flatten())
    assert E - EXACT["C6"] < 1e-5, E  # parameters are rounded to 4 decimals


def test_shot_measurement():
    rng = np.random.default_rng(1)
    for graph in EXACT:
        n, edges, s = define_graph(graph)
        qc, H = hamqaoa_circuit(n, edges, s, 2), heisenberg_hamiltonian(n, edges)
        theta = rng.uniform(-np.pi / 2, np.pi / 2, n_params(2))
        # 100k shots per basis: statistical error below ~0.03 per basis
        assert abs(measured_energy(qc, edges, theta) - cost(qc, H, theta)) < 0.1, graph


def test_greedy_K33():
    _, E, history = greedy_iterative("K33", p_max=10, n_restarts=20, target=EXACT["K33"], verbose=False)
    assert E - EXACT["K33"] < 1e-6, (E, history)


def test_greedy_C6():
    _, E, history = greedy_iterative("C6", p_max=10, n_restarts=20, target=EXACT["C6"], verbose=False)
    print(f"    C6 greedy: depth {history[-1][0]}, energies {[round(e, 6) for _, e in history]}")
    assert E - EXACT["C6"] < 1e-6, (E, history)


def test_random_restarts_C6():
    energies = np.array([E for _, E in random_restarts("C6", p=7, n_restarts=80, seed=0)])
    hits = np.sum(energies - EXACT["C6"] < 1e-6)
    print(f"    C6, p = 7: {hits} / 80 random starts reach the ground energy")
    assert hits >= 1, np.sort(energies)[:5]


if __name__ == "__main__":
    import time

    for test in [test_exact_diagonalization, test_circuit_against_numpy, test_table_II_parameters,
                 test_shot_measurement, test_greedy_K33, test_greedy_C6, test_random_restarts_C6]:
        t0 = time.time()
        test()
        print(f"PASS  {test.__name__}  ({time.time() - t0:.1f} s)")
