"""Quantum Max Cut / Heisenberg antiferromagnet on a 6-vertex bipartite graph with the
simplified HamQAOA of Kannan, King, Zhou (arXiv:2412.09221, Algorithm 2).

The ansatz and the measurements are done in Qiskit, the classical outer loop in scipy.

    |Θ⟩ = Π_{l=p..1} e^{-iδ_l D} e^{-iγ_l C} e^{-iβ_l B} e^{-iα_l A} ⊗_v |m_v⟩
    A = Σ_{u~v} Z_u Z_v,   B = Σ_v X_v,   C = Σ_v Z_v,   D = Σ_v s_v X_v

s ∈ {±1}^n is a classical max cut and ⊗_v |m_v⟩ is the ground state of D, i.e. |m_v⟩ is the
X eigenstate with eigenvalue -s_v (this convention reproduces the exact N = 6 ring parameters
of the paper's Table II).

Cost: Heisenberg energy ⟨H⟩, H = Σ_{u~v} w_uv (X_uX_v + Y_uY_v + Z_uZ_v), the same objective as in
sdp_approach.jl. The paper minimizes -H_QMC = (H - Σw)/2, which has the same minimizer.

Usage:
    python qoaa_approach.py --graph K33                                   # greedy iterative
    python qoaa_approach.py --graph C6 --strategy random --p 7 --restarts 80
"""

import argparse
import time
from multiprocessing import Pool

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


############################
##  1. Graph              ##
############################

def define_graph(name):
    """Return (n, edges, s): vertices 0..n-1, weighted edges (u, v, w), classical max cut s."""
    n = 6
    if name == "K33":
        A, B = [0, 1, 2], [3, 4, 5]
        edges = [(i, j, 1.0) for i in A for j in B]
    elif name == "C6":
        A, B = [0, 2, 4], [1, 3, 5]  # bipartition of the ring 0-1-2-3-4-5-0
        edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    else:
        raise ValueError(f"unknown graph {name}")
    # for a bipartite graph the bipartition cuts every edge
    s = [1 if v in A else -1 for v in range(n)]
    return n, edges, s


def heisenberg_hamiltonian(n, edges):
    """H = Σ_{u~v} w (XX + YY + ZZ) as a Qiskit observable (qubit q = vertex q)."""
    return SparsePauliOp.from_sparse_list(
        [(P, [u, v], w) for (u, v, w) in edges for P in ("XX", "YY", "ZZ")], num_qubits=n
    )


############################
##  2. Parameters         ##
############################

def n_params(p):
    """Θ = (α_1, β_1, γ_1, δ_1, α_2, ...): four angles per layer."""
    return 4 * p


############################
##  3. Ansatz circuit     ##
############################

def hamqaoa_circuit(n, edges, s, p):
    theta = ParameterVector("θ", n_params(p))
    qc = QuantumCircuit(n)
    # initial state ⊗_v |m_v⟩ = ground state of D: |−⟩ for s_v = +1, |+⟩ for s_v = -1
    for v in range(n):
        if s[v] == 1:
            qc.x(v)
        qc.h(v)
    # all terms within a driver commute, so every layer is implemented exactly
    for l in range(p):
        alpha, beta, gamma, delta = theta[4 * l : 4 * l + 4]
        for u, v, _ in edges:
            qc.rzz(2 * alpha, u, v)  # e^{-iα Z_u Z_v}
        for v in range(n):
            qc.rx(2 * beta, v)  # e^{-iβ X_v}
        for v in range(n):
            qc.rz(2 * gamma, v)  # e^{-iγ Z_v}
        for v in range(n):
            qc.rx(2 * s[v] * delta, v)  # e^{-iδ s_v X_v}
    return qc


############################
##  4. Cost function      ##
############################

estimator = StatevectorEstimator()


def cost(qc, observable, theta):
    """⟨Θ|H|Θ⟩ measured with the Estimator primitive (exact expectation, used by the optimizer)."""
    return float(estimator.run([(qc, observable, theta)]).result()[0].data.evs)


def measured_energy(qc, edges, theta, shots=100_000, seed=1234):
    """The same energy from finite-shot measurements: all XX, YY and ZZ terms are read off from
    three measurement settings (every qubit in the X, Y or Z basis)."""
    n = qc.num_qubits
    sampler = StatevectorSampler(seed=seed)
    bound = qc.assign_parameters(theta)
    energy = 0.0
    for basis in ("X", "Y", "Z"):
        mc = bound.copy()
        for q in range(n):
            if basis == "Y":
                mc.sdg(q)
            if basis in ("X", "Y"):
                mc.h(q)
        mc.measure_all()
        counts = sampler.run([mc], shots=shots).result()[0].data.meas.get_counts()
        for bits, c in counts.items():
            z = [1 - 2 * int(bits[n - 1 - q]) for q in range(n)]  # Qiskit bit strings are little-endian
            energy += c / shots * sum(w * z[u] * z[v] for (u, v, w) in edges)
    return energy


############################
##  5. Classical optimizer ##
############################

def minimize_energy(qc, observable, theta0):
    res = minimize(lambda th: cost(qc, observable, th), theta0, method="BFGS", options={"gtol": 1e-8})
    return res.x, res.fun


def _optimize_from_random(args):
    graph, p, seed = args
    n, edges, s = define_graph(graph)
    qc = hamqaoa_circuit(n, edges, s, p)
    rng = np.random.default_rng(seed)
    return minimize_energy(qc, heisenberg_hamiltonian(n, edges), rng.uniform(-np.pi / 2, np.pi / 2, n_params(p)))


def random_restarts(graph, p, n_restarts, seed=0, processes=None):
    """Random-initialization strategy: optimize all 4p parameters jointly from uniform starts in
    [-π/2, π/2)^{4p}; the restarts run in parallel. Returns [(θ, E), ...]."""
    with Pool(processes) as pool:
        return pool.map(_optimize_from_random, [(graph, p, seed + r) for r in range(n_restarts)])


def hessian(f, theta, h=1e-4):
    """Central finite-difference Hessian of f at theta."""
    k = len(theta)
    e = h * np.eye(k)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            hess[i, j] = hess[j, i] = (
                f(theta + e[i] + e[j]) - f(theta + e[i] - e[j]) - f(theta - e[i] + e[j]) + f(theta - e[i] - e[j])
            ) / (4 * h * h)
    return hess


def _optimize_transition_state(args):
    """Depth-p optimization started from the depth-(p-1) optimum with a zero layer inserted at
    `position`. The energy is unchanged there and the point is typically stationary, so BFGS started
    on it does not move: also start it a distance `step` along ± the lowest-curvature direction."""
    graph, p, theta_prev, position, step = args
    n, edges, s = define_graph(graph)
    qc, H = hamqaoa_circuit(n, edges, s, p), heisenberg_hamiltonian(n, edges)
    theta0 = np.insert(theta_prev, 4 * position, np.zeros(4))
    v = np.linalg.eigh(hessian(lambda th: cost(qc, H, th), theta0))[1][:, 0]
    return min((minimize_energy(qc, H, theta0 + sign * step * v) for sign in (0, 1, -1)), key=lambda r: r[1])


def greedy_iterative(graph, p_max, n_restarts, seed=0, step=0.1, target=None, tol=1e-6, verbose=True, processes=None):
    """Greedy-Iterative strategy (paper, Sec. III A and App. A.3, after Sack et al. [24]):
      p = 1: optimize from many uniform random starts in [-π/2, π/2)^4, keep the best;
      p-1 → p: insert a zero layer at each of the p positions of the best parameters (same energy,
      transition states of the depth-p landscape), leave them along the lowest-curvature direction,
      re-optimize all 4p parameters and keep the best.
    Stops early once the energy is within `tol` of `target` (if given)."""
    theta, E = min(random_restarts(graph, 1, n_restarts, seed, processes), key=lambda r: r[1])
    history = [(1, E)]
    if verbose:
        print(f"p = {1:2d}   ⟨H⟩ = {E:.10f}")
    for p in range(2, p_max + 1):
        if target is not None and E - target < tol:
            break
        with Pool(processes) as pool:
            candidates = pool.map(_optimize_transition_state, [(graph, p, theta, pos, step) for pos in range(p)])
        theta, E = min(candidates, key=lambda r: r[1])
        history.append((p, E))
        if verbose:
            print(f"p = {p:2d}   ⟨H⟩ = {E:.10f}")
    return theta, E, history


############################
##  6. Comparison         ##
############################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="K33", choices=["K33", "C6"])
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "random"])
    parser.add_argument("--p-max", type=int, default=10, help="maximal depth of the greedy iteration")
    parser.add_argument("--p", type=int, default=7, help="depth for the random strategy")
    parser.add_argument("--restarts", type=int, default=20, help="random starts (at p = 1 for greedy)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n, edges, s = define_graph(args.graph)
    H = heisenberg_hamiltonian(n, edges)
    W = sum(w for (_, _, w) in edges)
    E_exact = np.linalg.eigvalsh(H.to_matrix()).min()

    t0 = time.time()
    if args.strategy == "greedy":
        theta, E, history = greedy_iterative(args.graph, args.p_max, args.restarts, args.seed, target=E_exact)
        p = history[-1][0]
    else:
        p = args.p
        results = random_restarts(args.graph, p, args.restarts, args.seed)
        energies = np.array([E for _, E in results])
        theta, E = results[int(energies.argmin())]
        print(f"restarts within 1e-6 of the ground energy: {np.sum(energies - E_exact < 1e-6)} / {args.restarts}")

    qc = hamqaoa_circuit(n, edges, s, p)
    E_shots = measured_energy(qc, edges, theta)

    print(f"Graph                          {args.graph}  with {len(edges)} edges, cut s = {s}")
    print(f"Strategy                       {args.strategy}  ({time.time() - t0:.0f} s)")
    print(f"HamQAOA depth                  {p}  ({n_params(p)} parameters)")
    print(f"HamQAOA ⟨H⟩ (Estimator)        {cost(qc, H, theta):.10f}")
    print(f"HamQAOA ⟨H⟩ (100k shots/basis) {E_shots:.6f}")
    print(f"ED       λ_min(H)              {E_exact:.10f}")
    print(f"QMC value, Σ(1-σ·σ)/4 as in sdp_approach.jl:  HamQAOA {(W - E) / 4:.10f}   ED {(W - E_exact) / 4:.10f}")
    print(f"approximation ratio            {(W - E) / (W - E_exact):.10f}")


if __name__ == "__main__":
    main()
