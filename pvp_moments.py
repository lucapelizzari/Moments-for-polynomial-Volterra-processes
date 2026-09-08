"""Direct product-integration solver for scalar polynomial Volterra moments.

X_t = x0 + integral_0^t K(t-s) b(X_s) ds
         + integral_0^t K(t-s) sqrt(a(X_s)) dW_s,
b(x) = b0 + b1*x, a(x) = a0 + a1*x + a2*x*x.

No kernel approximation or Markovian lift is used by the moment solver.
Only the moment functions in the integrands are frozen at the left endpoint.
Fractional-kernel cell weights are evaluated from analytic primitives, in
float64 (including a Gauss hypergeometric function for off-diagonal products).

Implementation:
  * Symmetric maturity tuples, ranked in reverse-complement colex order.
  * One in-place state vector, descending moment orders, no second buffer.
  * Only the active prefix of each degree is visited at each time step.
  * Initial t=0 layer is constant and never stored.
  * Equal maturities are grouped with their exact multiplicities.
  * Blocked Numba prange, one unranking per block, O(1) deletion ranks.
  * No full tuple table, transition matrix, or transition-index array.

Memory for moment values: 8 * binom(M+p, p) bytes. There is also a packed
M*(M+1)/2 kernel-product table and small output/combinatorial arrays.
Worst-case tuple updates: sum_{q=1}^p binom(M+q, q+1), O(q^2) per tuple.
Symmetry does NOT remove the combinatorial growth in p and M.

The Monte Carlo helper is an INDEPENDENT time-discretized benchmark, not
an exact simulation of the moment scheme. It uses exact single-kernel cell
integrals and a piecewise-constant projection of Brownian white noise.
Its covariance on one cell is A_d*A_e/h, not the exact B_{d,e}; refine its
time grid separately. Antithetic standard errors are computed across pairs.

Requirements: Python >=3.10, numpy, scipy, numba.
Notebook additionally uses matplotlib, pandas, nbformat/Jupyter.

References:
  Abi Jaber, Cuchiero, Pelizzari, Pulido, Svaluto-Ferro (2024),
  Polynomial Volterra processes, Electronic Journal of Probability 29.
  https://arxiv.org/abs/2403.14251
  https://numba.readthedocs.io/en/stable/user/parallel.html
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from scipy.special import gammaln, hyp2f1

__version__ = "1.0.0"
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PolynomialSpec:
    """Constant initial value and the five polynomial coefficients.

    Existence/state-space admissibility is the caller's responsibility.
    The default a(x)=0.15+0.10*x+0.20*x^2 is positive on all of R.
    """
    x0: float = 0.7
    b0: float = 0.3
    b1: float = -0.8
    a0: float = 0.15
    a1: float = 0.10
    a2: float = 0.20

    def __post_init__(self) -> None:
        if not all(isfinite(float(v)) for v in
                   (self.x0, self.b0, self.b1, self.a0, self.a1, self.a2)):
            raise ValueError("Initial value and coefficients must be finite.")

    def coefficients(self) -> FloatArray:
        return np.array([self.b0, self.b1, self.a0, self.a1, self.a2], dtype=np.float64)

    def drift(self, x):
        return self.b0 + self.b1 * np.asarray(x)

    def variance(self, x):
        x = np.asarray(x)
        return self.a0 + x * (self.a1 + self.a2 * x)


@dataclass(frozen=True)
class KernelWeights:
    """Uniform-grid integrated weights, with one-based lags.

    A[d] = integral_{(d-1)h}^{dh} K(u) du, d=1,...,M; A[0]=0.
    B_{d,e} = integral_0^h K(d*h-r) K(e*h-r) dr.
    B is packed: B_{d,e}=B[d*(d-1)//2+e-1] for d>=e>=1.
    Call pair(d,e) or pair_matrix() outside the hot loop.
    """
    T: float
    n_steps: int
    A: FloatArray
    B: FloatArray
    name: str = "custom convolution kernel"

    def __post_init__(self) -> None:
        _validate_grid(self.n_steps, self.T)
        a = np.ascontiguousarray(self.A, dtype=np.float64)
        b = np.ascontiguousarray(self.B, dtype=np.float64)
        M = self.n_steps
        if a.shape != (M + 1,) or b.shape != (M * (M + 1) // 2,):
            raise ValueError("Incorrect weight shapes or packing.")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("Kernel weights must be finite.")
        if a[0] != 0.0:
            raise ValueError("A[0] is a sentinel and must equal zero.")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "B", b)

    @property
    def h(self) -> float:
        return self.T / self.n_steps

    @property
    def nbytes(self) -> int:
        return self.A.nbytes + self.B.nbytes

    def pair(self, d: int, e: int) -> float:
        if not (1 <= d <= self.n_steps and 1 <= e <= self.n_steps):
            raise IndexError("Lags must lie in 1,...,n_steps.")
        if d < e:
            d, e = e, d
        return float(self.B[d * (d - 1) // 2 + e - 1])

    def pair_matrix(self) -> FloatArray:
        """Expand B for diagnostics only; the solver never calls this."""
        M = self.n_steps
        out = np.empty((M, M), dtype=np.float64)
        for d in range(1, M + 1):
            row = self.B[d * (d - 1) // 2:d * (d + 1) // 2]
            out[d - 1, :d] = row
            out[:d, d - 1] = row
        return out


@dataclass(frozen=True)
class WorkEstimate:
    n_steps: int
    max_order: int
    state_entries: int
    state_bytes: int
    weight_bytes: int
    auxiliary_bytes: int
    tuple_updates: int
    ungrouped_contribution_bound: int

    @property
    def array_bytes(self) -> int:
        """Core persistent arrays only, not compiler/runtime/process memory."""
        return self.state_bytes + self.weight_bytes + self.auxiliary_bytes

    @property
    def array_mib(self) -> float:
        return self.array_bytes / 2**20


@dataclass(frozen=True)
class MomentResult:
    times: FloatArray
    moments: FloatArray  # shape (M+1, p+1); column 0 is one
    solve_seconds: float  # includes JIT on the first call
    estimate: WorkEstimate
    parallel: bool

    @property
    def terminal(self) -> FloatArray:
        return self.moments[-1]


@dataclass(frozen=True)
class MonteCarloResult:
    times: FloatArray
    moments: FloatArray
    standard_errors: FloatArray
    n_paths: int
    n_independent: int  # antithetic pairs when antithetic=True
    antithetic: bool
    seconds: float
    scheme: str = "cell-integrated Euler / projected Brownian noise"


# ---------------------------------------------------------------------------
# Weight construction: no kernel fitting, sampling grid, or SOE approximation.
# ---------------------------------------------------------------------------
def _validate_grid(n_steps: int, T: float) -> None:
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not isfinite(float(T)) or T <= 0:
        raise ValueError("T must be finite and positive.")


def _check_memory(nbytes: int, max_memory_gib: float | None, what: str) -> None:
    if max_memory_gib is not None:
        if not isfinite(float(max_memory_gib)) or max_memory_gib <= 0:
            raise ValueError("max_memory_gib must be positive or None.")
        if nbytes > max_memory_gib * 2**30:
            raise MemoryError(
                f"{what} needs about {nbytes / 2**30:.3f} GiB of core arrays, "
                f"exceeding the {max_memory_gib:.3f} GiB guard. "
                "Reduce n_steps/max_order or explicitly change max_memory_gib."
            )


def _unit_power_differences(x: FloatArray, exponent: float) -> FloatArray:
    """x**r-(x-1)**r for x>=1, avoiding cancellation for large x."""
    out = np.ones_like(x)
    mask = x > 1.0
    out[mask] = -(x[mask] ** exponent) * np.expm1(
        exponent * np.log1p(-1.0 / x[mask])
    )
    return out


def fractional_weights(
    n_steps: int, T: float = 1.0, H: float = 0.75,
    scale: float = 1.0, *, max_memory_gib: float | None = 2.0,
) -> KernelWeights:
    r"""Analytic product-integration weights for K(t)=scale*t^(H-1/2)/Gamma(H+1/2).

    H>0 is required for square integrability at zero. The notebook uses H=0.75.
    'Smooth' here means nonsingular (H>1/2), not necessarily C^1 at zero.

    Put beta=H-1/2, delta=d-e>0. An off-diagonal primitive is
      F(x,delta) = x^(beta+1)*(x+delta)^beta/(beta+1)
                  * hyp2f1(-beta,1,beta+2,x/(x+delta)).
    B[d,e] = scale^2*h^(2*beta+1)/Gamma(beta+1)^2
             * (F(e,delta)-F(e-1,delta)).
    Diagonal weights use a stable power difference, not this primitive.
    We evaluate analytic formulas in floating point; these are not a claim
    of exact arithmetic. Adaptive quadrature is used only in self-tests.
    """
    _validate_grid(n_steps, T)
    if not isfinite(float(H)) or H <= 0:
        raise ValueError("H must be finite and positive.")
    if not isfinite(float(scale)):
        raise ValueError("scale must be finite.")
    M = int(n_steps)
    _check_memory(8 * (M + 1 + M * (M + 1) // 2), max_memory_gib, "Weights")
    h = float(T) / M
    beta = float(H) - 0.5
    alpha = beta + 1.0
    ds = np.arange(1, M + 1, dtype=np.float64)
    A = np.zeros(M + 1, dtype=np.float64)
    A[1:] = (scale * h**alpha * np.exp(-gammaln(alpha + 1.0))
             * _unit_power_differences(ds, alpha))
    B = np.empty(M * (M + 1) // 2, dtype=np.float64)
    factor = scale**2 * h**(2.0 * H) * np.exp(-2.0 * gammaln(alpha))
    diagonal = factor * _unit_power_differences(ds, 2.0 * H) / (2.0 * H)
    if H == 0.5:
        B.fill(scale**2 * h)
    else:
        for d in range(1, M + 1):
            base = d * (d - 1) // 2
            B[base + d - 1] = diagonal[d - 1]
            if d == 1:
                continue
            e = np.arange(1, d, dtype=np.float64)
            delta = d - e
            upper = (e**alpha * (e + delta)**beta / alpha
                     * hyp2f1(-beta, 1.0, beta + 2.0, e / (e + delta)))
            lower = np.zeros_like(e)
            positive = e > 1.0
            x = e[positive] - 1.0
            c = delta[positive]
            lower[positive] = (x**alpha * (x + c)**beta / alpha
                               * hyp2f1(-beta, 1.0, beta + 2.0, x / (x + c)))
            B[base:base + d - 1] = factor * (upper - lower)
    if scale != 0 and (np.any(B <= 0) or not np.all(np.isfinite(B))):
        raise FloatingPointError("Analytic product-weight evaluation lost positivity/finite values.")
    return KernelWeights(float(T), M, A, B, f"fractional(H={H:g}, scale={scale:g})")


def weights_from_integrals(
    n_steps: int, T: float,
    integral_k: Callable[[float, float], float],
    integral_kk: Callable[[float, float, float], float],
    *, name: str = "custom convolution kernel", max_memory_gib: float | None = 2.0,
) -> KernelWeights:
    """Use a different convolution kernel WITHOUT changing the solver.

    Supply integral_k(lo,hi)=integral_lo^hi K(u) du and
    integral_kk(lo,hi,shift)=integral_lo^hi K(u)K(u+shift) du.
    The caller controls how accurately these integrals are evaluated.
    No kernel-value sampling or approximation is performed here.
    """
    _validate_grid(n_steps, T)
    M = int(n_steps)
    _check_memory(8 * (M + 1 + M * (M + 1) // 2), max_memory_gib, "Weights")
    h = T / M
    A = np.zeros(M + 1, dtype=np.float64)
    B = np.empty(M * (M + 1) // 2, dtype=np.float64)
    for d in range(1, M + 1):
        A[d] = integral_k((d - 1) * h, d * h)
        for e in range(1, d + 1):
            B[d * (d - 1) // 2 + e - 1] = integral_kk(
                (e - 1) * h, e * h, (d - e) * h
            )
    return KernelWeights(float(T), M, A, B, name)


# ---------------------------------------------------------------------------
# Symmetric indexing and operation counts.
# ---------------------------------------------------------------------------
def estimate_work(n_steps: int, max_order: int) -> WorkEstimate:
    """Inspect memory/iteration counts BEFORE allocating or compiling."""
    _validate_grid(n_steps, 1.0)
    if isinstance(max_order, (bool, np.bool_)) or not isinstance(max_order, (int, np.integer)) or max_order < 0:
        raise ValueError("max_order must be a nonnegative integer.")
    M, p = int(n_steps), int(max_order)
    states = comb(M + p, p)
    weight_bytes = 8 * (M + 1 + M * (M + 1) // 2)
    aux = 8 * ((M + p + 1) * (max(p, 2) + 1)
               + 2 * (p + 2) + (M + 1) * (p + 2))
    updates = sum(comb(M + q, q + 1) for q in range(1, p + 1))
    contributions = sum((q + q * (q - 1) // 2) * comb(M + q, q + 1)
                        for q in range(1, p + 1))
    return WorkEstimate(M, p, states, 8 * states, weight_bytes, aux,
                        updates, contributions)


def _binomial_table(M: int, p: int) -> IntArray:
    table = np.zeros((M + p + 1, max(p, 2) + 1), dtype=np.int64)
    limit = np.iinfo(np.int64).max
    for n in range(table.shape[0]):
        for k in range(min(n, table.shape[1] - 1) + 1):
            value = comb(n, k)
            if value > limit:
                raise OverflowError("Combinatorial indices exceed int64 capacity.")
            table[n, k] = value
    return table


@njit(cache=True, inline="always")
def _unrank_into(rank, q, max_u, choose, u):
    """Colex rank=sum_i C(u_i+i,i+1), with 0<=u_0<=...<=u_(q-1)."""
    rem = rank
    upper = max_u + q - 1
    for r in range(q, 0, -1):
        lo = r - 1
        hi = upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if choose[mid, r] <= rem:
                lo = mid
            else:
                hi = mid - 1
        u[r - 1] = lo - (r - 1)
        rem -= choose[lo, r]
        upper = lo - 1


@njit(cache=True, inline="always")
def _next_colex(u, q):
    for i in range(q - 1):
        if u[i] < u[i + 1]:
            u[i] += 1
            for j in range(i):
                u[j] = 0
            return
    u[q - 1] += 1
    for j in range(q - 1):
        u[j] = 0


@njit(cache=True)
def _update_block(state, offsets, choose, A, B, coeff, powers,
                  M, q, k, start, stop):
    """Update one independent contiguous colex block, without temporaries per tuple.

    Complement coordinates u_i=M-j_{q-i} are increasing. At time k the
    inserted maturity k becomes L=M-k, larger than all updated coordinates.
    Thus inserting it simply appends L. Source ranks lie in the untouched
    tail of this degree's active array. Lower degrees are still old because
    the outer loop visits degrees in descending order.
    """
    if start >= stop:
        return
    L = M - k
    u = np.empty(q, dtype=np.int64)
    _unrank_into(start, q, L - 1, choose, u)
    # Prefix sums for ranks after deleting zero, one, or two coordinates.
    pref0 = np.empty(q + 1, dtype=np.int64)
    pref1 = np.empty(q + 1, dtype=np.int64)
    pref2 = np.empty(q + 1, dtype=np.int64)
    groups = np.empty(q, dtype=np.int64)
    counts = np.empty(q, dtype=np.int64)
    off = offsets[q]
    off1 = offsets[q - 1]
    off2 = offsets[max(0, q - 2)]
    append1 = choose[L + q - 1, q]
    append2 = 0
    if q >= 2:
        append2 = choose[L + q - 2, q - 1]
    b0, b1, a0, a1, a2 = coeff
    has_b = b0 != 0.0 or b1 != 0.0
    has_a = q >= 2 and (a0 != 0.0 or a1 != 0.0 or a2 != 0.0)
    init_b = (b0 + b1 * powers[1]) * powers[q - 1]
    init_a = 0.0
    if q >= 2:
        init_a = (a0 + a1 * powers[1] + a2 * powers[1]**2) * powers[q - 2]

    for rank in range(start, stop):
        ng = 0
        i = 0
        while i < q:
            end = i + 1
            while end < q and u[end] == u[i]:
                end += 1
            groups[ng] = i
            counts[ng] = end - i
            ng += 1
            i = end

        if k > 0:
            pref0[0] = 0
            pref1[0] = 0
            pref2[0] = 0
            for i in range(q):
                pref0[i + 1] = pref0[i] + choose[u[i] + i, i + 1]
                pref1[i + 1] = pref1[i]
                pref2[i + 1] = pref2[i]
                if i >= 1:
                    pref1[i + 1] += choose[u[i] + i - 1, i]
                if i >= 2:
                    pref2[i + 1] += choose[u[i] + i - 2, i - 1]
            value = state[off + rank]
        else:
            value = powers[q]

        if has_b:
            for gi in range(ng):
                i = groups[gi]
                lag_i = L - u[i]
                if k == 0:
                    source = init_b
                else:
                    remove1 = pref0[i] + pref1[q] - pref1[i + 1]
                    source = (b0 * state[off1 + remove1]
                              + b1 * state[off + remove1 + append1])
                value += counts[gi] * A[lag_i] * source

        if has_a:
            for gi in range(ng):
                i = groups[gi]
                lag_i = L - u[i]
                # Same-maturity pairs, with binomial multiplicity.
                if counts[gi] >= 2:
                    j = i + 1
                    if k == 0:
                        source = init_a
                    else:
                        remove2 = (pref0[i] + pref1[j] - pref1[i + 1]
                                   + pref2[q] - pref2[j + 1])
                        source = (a0 * state[off2 + remove2]
                                  + a1 * state[off1 + remove2 + append2]
                                  + a2 * state[off + remove2 + append2 + append1])
                    multiplicity = counts[gi] * (counts[gi] - 1) // 2
                    value += multiplicity * B[lag_i * (lag_i + 1) // 2 - 1] * source
                # Different-maturity pairs; lag_i >= lag_j in this ordering.
                for gj in range(gi + 1, ng):
                    j = groups[gj]
                    lag_j = L - u[j]
                    if k == 0:
                        source = init_a
                    else:
                        remove2 = (pref0[i] + pref1[j] - pref1[i + 1]
                                   + pref2[q] - pref2[j + 1])
                        source = (a0 * state[off2 + remove2]
                                  + a1 * state[off1 + remove2 + append2]
                                  + a2 * state[off + remove2 + append2 + append1])
                    value += (counts[gi] * counts[gj]
                              * B[lag_i * (lag_i - 1) // 2 + lag_j - 1] * source)
        state[off + rank] = value
        if rank + 1 < stop:
            _next_colex(u, q)


@njit(cache=True, parallel=True)
def _update_degree_parallel(state, offsets, choose, A, B, coeff, powers,
                            M, q, k, n_active, block_size):
    nblocks = (n_active + block_size - 1) // block_size
    for block in prange(nblocks):
        # Avoid the unsigned prange induction variable propagating into ranks.
        start = np.int64(block) * block_size
        stop = min(start + block_size, n_active)
        _update_block(state, offsets, choose, A, B, coeff, powers,
                      M, q, k, start, stop)


@njit(cache=True)
def _solve_core(state, offsets, choose, A, B, coeff, powers, M, p,
                use_parallel, block_size, parallel_threshold, out):
    state[0] = 1.0
    for q in range(p + 1):
        out[0, q] = powers[q]
    for k in range(M):
        L = M - k
        # In-place evaluation of the SAME explicit frozen-state recursion.
        for q in range(p, 0, -1):
            n_active = choose[L + q - 1, q]
            if use_parallel and n_active >= parallel_threshold:
                _update_degree_parallel(state, offsets, choose, A, B, coeff,
                                        powers, M, q, k, n_active, block_size)
            else:
                _update_block(state, offsets, choose, A, B, coeff, powers,
                              M, q, k, 0, n_active)
            # u=(L-1,...,L-1), i.e. maturity (k+1,...,k+1), is last.
            out[k + 1, q] = state[offsets[q] + n_active - 1]
        out[k + 1, 0] = 1.0


def solve_moments(
    weights: KernelWeights, spec: PolynomialSpec,
    max_order: int = 4, *, parallel: bool = True,
    block_size: int = 512, parallel_threshold: int = 4096,
    max_memory_gib: float | None = 2.0,
) -> MomentResult:
    """Solve all raw moments 0,...,max_order at every grid time.

    The update is M^(k+1) = M^k + Q_k M^k. Q_k is applied as a structured
    sparse operator, NOT assembled into a matrix. Its scalar weights A,B
    have already been computed. No Python calls occur inside the time sweep.

    parallel=True parallelizes blocks within a degree; degrees and time
    remain sequential. This is necessary for the one-buffer implementation.
    Change Numba's thread count externally using numba.set_num_threads().

    Fixed allocation + shrinking active prefixes avoids reallocation and
    repeated copies. Dead tails are no longer iterated over, although the
    allocation is retained until completion. Only diagonal moment curves
    are returned, so the large work array is released on return.

    This is a left-point product-integration method, not an exact-in-time
    moment formula. Check grid convergence; no positivity projection or
    numerical-stability guarantee is imposed at a coarse grid.
    """
    M = weights.n_steps
    estimate = estimate_work(M, max_order)
    p = int(max_order)
    if block_size < 1 or not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be a positive integer.")
    if parallel_threshold < 1 or not isinstance(parallel_threshold, (int, np.integer)):
        raise ValueError("parallel_threshold must be a positive integer.")
    if estimate.state_entries > np.iinfo(np.int64).max:
        raise OverflowError("State index exceeds int64 capacity.")
    _check_memory(estimate.array_bytes, max_memory_gib, "Moment solver")
    choose = _binomial_table(M, p)
    offsets = np.zeros(p + 2, dtype=np.int64)
    for q in range(p + 1):
        offsets[q + 1] = offsets[q] + comb(M - 1 + q, q)
    state = np.empty(int(offsets[-1]), dtype=np.float64)
    out = np.empty((M + 1, p + 1), dtype=np.float64)
    # At least powers[1] must exist even if p=0.
    with np.errstate(over="raise", invalid="raise"):
        powers = np.power(float(spec.x0), np.arange(max(p, 1) + 1, dtype=np.int64))
    start = perf_counter()
    _solve_core(state, offsets, choose, weights.A, weights.B, spec.coefficients(),
                powers, M, p, bool(parallel), int(block_size),
                int(parallel_threshold), out)
    seconds = perf_counter() - start
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite moments: refine the grid/check coefficients and scale.")
    return MomentResult(np.linspace(0.0, weights.T, M + 1), out, seconds,
                        estimate, bool(parallel))


# ---------------------------------------------------------------------------
# Independent Monte Carlo, bounded memory, antithetic-pair standard errors.
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _simulate_batch(normals, A, h, x0, coeff, antithetic):
    units, M = normals.shape
    nsigns = 2 if antithetic else 1
    paths = np.empty((units, nsigns, M + 1), dtype=np.float64)
    inv_sqrt_h = 1.0 / np.sqrt(h)
    b0, b1, a0, a1, a2 = coeff
    for unit in prange(units):
        forcing = np.empty(M, dtype=np.float64)
        for side in range(nsigns):
            sign = 1.0 if side == 0 else -1.0
            x = x0
            paths[unit, side, 0] = x0
            for k in range(M):
                a = a0 + x * (a1 + a2 * x)
                if a < 0.0 or not np.isfinite(a):
                    for j in range(k + 1, M + 1):
                        paths[unit, side, j] = np.nan
                    break
                forcing[k] = (b0 + b1 * x + np.sqrt(a) * inv_sqrt_h
                              * sign * normals[unit, k])
                x = x0
                for j in range(k + 1):
                    x += A[k + 1 - j] * forcing[j]
                paths[unit, side, k + 1] = x
    return paths


def monte_carlo_moments(
    weights: KernelWeights, spec: PolynomialSpec, max_order: int = 4,
    *, n_paths: int = 40000, seed: int = 12345,
    batch_size: int = 512, antithetic: bool = True,
) -> MonteCarloResult:
    r"""Independent direct Euler/projection Monte Carlo on the supplied grid.

    X_n = x0 + sum_{k<n} A[n-k] *
                  (b(X_k) + sqrt(a(X_k))*xi_k/sqrt(h)).

    The xi_k are independent N(0,1). A uses exact deterministic K integrals.
    This projects Brownian white noise to constants on cells; it is NOT an
    exact sampler of the Gaussian integrals integral_cell K(T-r)dW_r.
    Its single-cell covariance is A[d]*A[e]/h, versus B[d,e] in the
    deterministic solver. Refine this MC grid independently.

    Antithetic xi/-xi trajectories are averaged BEFORE variance estimation;
    standard_errors use independent pairs, not the correlated individual
    paths. Batches use stable parallel-variance merging. No all-path history
    is retained; batch_size counts independent units (pairs, if antithetic).

    A negative a(X_k) raises an error; there is no silent clipping or
    reflection of the process. Bounded-state models may need a different
    Monte Carlo discretization, even when the moment solver is valid.
    """
    estimate_work(weights.n_steps, max_order)  # validates p without allocating
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 4:
        raise ValueError("n_paths must be an integer >=4.")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even with antithetic=True.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    p, M = int(max_order), weights.n_steps
    n_units = n_paths // 2 if antithetic else n_paths
    rng = np.random.default_rng(seed)
    means = np.zeros((M + 1, p + 1), dtype=np.float64)
    sumsquares = np.zeros_like(means)
    count = 0
    start = perf_counter()
    while count < n_units:
        nb = min(int(batch_size), n_units - count)
        normals = rng.standard_normal((nb, M))
        paths = _simulate_batch(normals, weights.A, weights.h,
                                float(spec.x0), spec.coefficients(), bool(antithetic))
        if not np.all(np.isfinite(paths)):
            raise ValueError("MC encountered a negative variance or a non-finite path; no clipping was applied.")
        power = np.ones_like(paths)
        with np.errstate(over="raise", invalid="raise"):
            for q in range(p + 1):
                if q:
                    power *= paths
                observations = power.mean(axis=1)  # pair means, or singleton
                block_mean = observations.mean(axis=0)
                centered = observations - block_mean
                block_ss = np.sum(centered * centered, axis=0)
                delta = block_mean - means[:, q]
                total = count + nb
                sumsquares[:, q] += block_ss + delta**2 * (count * nb / total)
                means[:, q] += delta * (nb / total)
        count += nb
    se = np.sqrt(np.maximum(sumsquares, 0.0) / (n_units - 1) / n_units)
    means[:, 0] = 1.0
    se[:, 0] = 0.0
    means[0, :] = np.power(float(spec.x0), np.arange(p + 1))
    se[0, :] = 0.0
    return MonteCarloResult(np.linspace(0.0, weights.T, M + 1), means, se,
                            int(n_paths), int(n_units), bool(antithetic),
                            perf_counter() - start)


# ---------------------------------------------------------------------------
# Transparent, independently formulated validation helpers.
# ---------------------------------------------------------------------------
def gaussian_exact_moments(times, max_order: int, *, H: float = 0.75,
                           x0: float = 0.7, b0: float = 0.0,
                           a0: float = 0.3, scale: float = 1.0) -> FloatArray:
    """Exact raw moments when b1=a1=a2=0 (Gaussian stochastic convolution)."""
    if H <= 0 or a0 < 0 or max_order < 0:
        raise ValueError("Require H>0, a0>=0, max_order>=0.")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or np.any(t < 0):
        raise ValueError("times must be a one-dimensional nonnegative array.")
    alpha = H + 0.5
    mean = x0 + b0 * scale * t**alpha * np.exp(-gammaln(alpha + 1))
    variance = a0 * scale**2 * t**(2 * H) * np.exp(-2 * gammaln(alpha)) / (2 * H)
    out = np.ones((t.size, max_order + 1), dtype=np.float64)
    if max_order >= 1:
        out[:, 1] = mean
    for q in range(2, max_order + 1):
        out[:, q] = mean * out[:, q - 1] + (q - 1) * variance * out[:, q - 2]
    return out


def fractional_mean_series(
    times, spec: PolynomialSpec, *, H: float = 0.75, scale: float = 1.0,
    tolerance: float = 2e-14, max_terms: int = 1000,
) -> FloatArray:
    r"""Independent mean reference from its convergent Mittag-Leffler series.

    For alpha=H+1/2, lambda=scale*b1, c=scale*(b0+b1*x0),
      E[X_t] = x0 + c * sum_{n>=1}
                         lambda^(n-1) t^(alpha*n)/Gamma(alpha*n+1).
    No moments or kernel weights from the numerical solver are used.
    This direct series is intended for moderate arguments, as in the
    notebook. A cancellation check rejects numerically ill-conditioned
    evaluations rather than returning an unreliable large-argument result.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or np.any(t < 0) or not np.all(np.isfinite(t)):
        raise ValueError("times must be a nonempty finite nonnegative vector.")
    if not isfinite(float(H)) or H <= 0 or not isfinite(float(scale)):
        raise ValueError("Require H>0 and finite H, scale.")
    if tolerance <= 0 or max_terms < 1:
        raise ValueError("Require tolerance>0 and max_terms>=1.")
    alpha = H + 0.5
    lam = scale * spec.b1
    c = scale * (spec.b0 + spec.b1 * spec.x0)
    out = np.full_like(t, float(spec.x0))
    if c == 0.0:
        return out
    if lam == 0.0:
        return out + c * t**alpha * np.exp(-gammaln(alpha + 1))
    log_t = np.full_like(t, -np.inf)
    np.log(t, out=log_t, where=t > 0)
    absolute_sum = np.zeros_like(t)
    compensation = np.zeros_like(t)
    sign = np.sign(c)
    consecutive_small = 0
    for n in range(1, max_terms + 1):
        log_magnitude = (np.log(abs(c)) + (n - 1) * np.log(abs(lam))
                         + alpha * n * log_t - gammaln(alpha * n + 1))
        with np.errstate(over="raise", invalid="raise"):
            magnitude = np.exp(log_magnitude)
        term = sign * magnitude
        absolute_sum += magnitude
        # Compensated summation of alternating terms.
        corrected = term - compensation
        updated = out + corrected
        compensation = (updated - out) - corrected
        out = updated
        if np.max(magnitude) <= tolerance * (1 + np.max(abs(out))):
            consecutive_small += 1
        else:
            consecutive_small = 0
        if consecutive_small >= 3:
            if np.finfo(float).eps * np.max(absolute_sum) > 1e-11 * (1 + np.max(abs(out))):
                raise FloatingPointError("Mean series suffers excessive cancellation at these arguments.")
            return out
        if lam < 0:
            sign = -sign
    raise RuntimeError("Mean series did not converge within max_terms.")


def classical_moment_matrix(spec: PolynomialSpec, max_order: int) -> FloatArray:
    """Only a K=1 validation reference, NOT the Volterra solver."""
    G = np.zeros((max_order + 1, max_order + 1), dtype=np.float64)
    for q in range(1, max_order + 1):
        pair = q * (q - 1) / 2
        G[q, q] = q * spec.b1 + pair * spec.a2
        G[q, q - 1] = q * spec.b0 + pair * spec.a1
        if q >= 2:
            G[q, q - 2] = pair * spec.a0
    return G


def dense_reference(weights: KernelWeights, spec: PolynomialSpec,
                    max_order: int = 3) -> FloatArray:
    """Slow independent full-tensor implementation for TINY test grids only."""
    from itertools import product
    M, p = weights.n_steps, max_order
    if sum((M + 1)**q for q in range(p + 1)) > 200000:
        raise ValueError("dense_reference is only for tiny regression tests.")
    tensors = [np.full((M + 1,) * q, spec.x0**q, dtype=np.float64)
               for q in range(p + 1)]
    out = np.ones((M + 1, p + 1), dtype=np.float64)
    out[0] = [spec.x0**q for q in range(p + 1)]
    for k in range(M):
        old = tensors
        new = [x.copy() for x in old]
        for q in range(1, p + 1):
            for J in product(range(k + 1, M + 1), repeat=q):
                value = float(old[q][J])
                for i in range(q):
                    rest = J[:i] + J[i + 1:]
                    value += weights.A[J[i] - k] * (
                        spec.b0 * old[q - 1][rest]
                        + spec.b1 * old[q][(k,) + rest])
                for i in range(q):
                    for j in range(i + 1, q):
                        rest = tuple(J[l] for l in range(q) if l != i and l != j)
                        value += weights.pair(J[i] - k, J[j] - k) * (
                            spec.a0 * old[q - 2][rest]
                            + spec.a1 * old[q - 1][(k,) + rest]
                            + spec.a2 * old[q][(k, k) + rest])
                new[q][J] = value
            out[k + 1, q] = new[q][(k + 1,) * q]
        tensors = new
    return out


def run_self_tests() -> list[dict]:
    """Run deterministic tests; raises AssertionError on failure.

    Includes independent dense tensors, serial/parallel identity, analytic
    weight integrals vs adaptive quadrature, K=1 explicit Euler closure,
    exact Gaussian first/second/third moments and fourth-moment refinement.
    """
    from itertools import combinations_with_replacement
    from scipy.integrate import quad
    records = []

    def record(name, err):
        records.append({"test": name, "max_abs_error": float(err), "passed": True})

    # Independently generate all sorted tuples and their combinatorial ranks.
    table = _binomial_table(7, 5)
    for q in range(1, 6):
        tuples = list(combinations_with_replacement(range(5), q))
        ranks = [sum(comb(v + i, i + 1) for i, v in enumerate(u)) for u in tuples]
        assert sorted(ranks) == list(range(len(tuples)))
        dest = np.empty(q, dtype=np.int64)
        for rank, u in zip(ranks, tuples):
            _unrank_into(rank, q, 4, table, dest)
            np.testing.assert_array_equal(dest, u)
    record("colex ranks and unranking (orders 1-5)", 0)

    max_weight_error = 0.0
    for H in (0.2, 0.5, 0.75, 1.2):
        w = fractional_weights(9, 1.3, H)
        beta = H - 0.5
        inv_gamma = np.exp(-gammaln(H + 0.5))
        for d, e in ((1, 1), (2, 1), (9, 1), (3, 2), (9, 8), (9, 9)):
            lo, hi, shift = (e - 1) * w.h, e * w.h, (d - e) * w.h
            ref = quad(lambda u: u**beta * (u + shift)**beta * inv_gamma**2,
                       lo, hi, epsabs=2e-12, epsrel=2e-12, limit=250)[0]
            np.testing.assert_allclose(w.pair(d, e), ref, rtol=2e-10, atol=2e-12)
            max_weight_error = max(max_weight_error, abs(w.pair(d, e) - ref))
    record("fractional B weights vs independent quadrature", max_weight_error)

    spec = PolynomialSpec()
    for H in (0.25, 0.75):
        w = fractional_weights(6, 1.0, H)
        ref = dense_reference(w, spec, 4)
        got = solve_moments(w, spec, 4, parallel=False).moments
        np.testing.assert_allclose(got, ref, rtol=2e-13, atol=2e-13)
        record(f"packed vs frozen full tensors, H={H}", np.max(abs(got - ref)))
        threaded = solve_moments(w, spec, 4, parallel=True,
                                  block_size=7, parallel_threshold=1).moments
        np.testing.assert_array_equal(got, threaded)
        record(f"in-place serial/parallel identity, H={H}", 0)

    w = fractional_weights(12, 0.8, 0.5)
    p = 5
    G = classical_moment_matrix(spec, p)
    ref = np.ones((13, p + 1))
    ref[0] = np.power(spec.x0, np.arange(p + 1))
    for k in range(12):
        ref[k + 1] = ref[k] + w.h * (G @ ref[k])
    got = solve_moments(w, spec, p, parallel=False).moments
    np.testing.assert_allclose(got, ref, rtol=3e-13, atol=3e-13)
    record("K=1 agrees with the explicit Euler moment-vector recursion", np.max(abs(got - ref)))

    gaussian = PolynomialSpec(x0=0.7, b0=0.0, b1=0.0, a0=0.3, a1=0.0, a2=0.0)
    errors = []
    for M in (12, 24):
        w = fractional_weights(M, 1.0, 0.75)
        got = solve_moments(w, gaussian, 4, parallel=False)
        exact = gaussian_exact_moments(got.times, 4, H=0.75)
        np.testing.assert_allclose(got.moments[:, :4], exact[:, :4], rtol=2e-12, atol=2e-12)
        errors.append(abs(got.terminal[4] - exact[-1, 4]))
    assert errors[1] < errors[0]
    record("Gaussian orders 0-3 exact; order 4 improves on grid refinement", errors[1])

    t = np.linspace(0.0, 1.0, 15)
    mean_ref = (-spec.b0 / spec.b1
                + (spec.x0 + spec.b0 / spec.b1) * np.exp(spec.b1 * t))
    mean_got = fractional_mean_series(t, spec, H=0.5)
    np.testing.assert_allclose(mean_got, mean_ref, rtol=1e-12, atol=1e-12)
    record("mean series at K=1 vs elementary exact mean", np.max(abs(mean_got - mean_ref)))

    # Degenerate cases: one cell, p=0, x0=0, zero kernel, deterministic forcing.
    for initial in (0.0, -0.3):
        spec0 = PolynomialSpec(x0=initial)
        w = fractional_weights(1, 1.0, 0.75)
        got = solve_moments(w, spec0, 4, parallel=False).moments
        ref = dense_reference(w, spec0, 4)
        np.testing.assert_allclose(got, ref, rtol=2e-13, atol=2e-13)
    z = solve_moments(fractional_weights(4, H=0.75, scale=0.0), spec, 4, parallel=False)
    np.testing.assert_allclose(z.moments, np.tile(np.power(spec.x0, np.arange(5)), (5, 1)))
    one = solve_moments(fractional_weights(3), spec, 0).moments
    np.testing.assert_array_equal(one, np.ones((4, 1)))
    record("one-cell, zero initial value, zero kernel, degree-zero edge cases", 0)
    return records


if __name__ == "__main__":
    import numba
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    for item in run_self_tests():
        print(f"PASS  {item['test']}: {item['max_abs_error']:.3e}")
    w = fractional_weights(48, T=1.0, H=0.75)
    answer = solve_moments(w, PolynomialSpec(), max_order=4)
    print("Terminal moments:", answer.terminal)
    print(f"Core arrays: {answer.estimate.array_mib:.2f} MiB")
    print(f"Solver call: {answer.solve_seconds:.3f}s (may include JIT)")
