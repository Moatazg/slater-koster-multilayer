"""
Multilayer tight-binding (5 d-orbitals × spin) with Slater–Koster-style hoppings,
exchange-split ferromagnetic layers, Green's functions, DOS, and Kubo conductivity.

This module mirrors a typical Mathematica notebook structure:
  1) Parameters & lattice
  2) Single-layer Hamiltonians (F and N) with magnetization
  3) Multilayer assembly (block-tridiagonal)
  4) Velocity operators
  5) Green's function & observables (DOS, layer/spin-resolved DOS)
  6) Kubo-Greenwood conductivity
  7) Band structure along a high-symmetry path

The implementation is deliberately modular and explicit. Where full Slater–Koster
(ddσ, ddπ, ddδ) tables would normally appear, we expose hooks that accept arbitrary
orbital hopping matrices per neighbor vector. A concise helper is provided to build
simple diagonal-orbital couplings; you can replace it with your full SK builder
if desired.

Default units set ħ = e = a0 = 1 unless otherwise noted.

Author: ChatGPT (Python port)
"""
from __future__ import annotations

import cmath
import math
import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# =============================
# 0) Utilities & constants
# =============================

complex_t = np.complex128
float_t = np.float64

I2: NDArray[complex_t] = np.eye(2, dtype=complex)

sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# =============================
# 1) Basis & projectors
# =============================

D_ORBITALS = ("xy", "yz", "zx", "x2y2", "z2")
N_DORB = len(D_ORBITALS)
N_SPIN = 2


def kron(a: NDArray[complex_t], b: NDArray[complex_t]) -> NDArray[complex_t]:
    return np.kron(a, b)


def unit_vec(theta: float, phi: float) -> Tuple[float, float, float]:
    """Spherical to Cartesian, theta = polar angle from +z, phi = azimuth."""
    st, ct = math.sin(theta), math.cos(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    return (st * cp, st * sp, ct)


def spin_dir_op(n: Tuple[float, float, float]) -> NDArray[complex_t]:
    nx, ny, nz = n
    return nx * sigma_x + ny * sigma_y + nz * sigma_z


def spin_projectors(n: Tuple[float, float, float]) -> Tuple[NDArray[complex_t], NDArray[complex_t]]:
    """Projectors onto spin ↑/↓ along direction n."""
    Sn = spin_dir_op(n)
    Pup = 0.5 * (I2 + Sn)
    Pdn = 0.5 * (I2 - Sn)
    return Pup, Pdn


# =============================
# 2) Lattice & neighbors
# =============================

@dataclass
class Neighbor:
    R: Tuple[float, float]  # in-plane vector (Rx, Ry) in units of a0
    T_orb: NDArray[complex_t]  # (5x5) orbital hopping for this neighbor


@dataclass
class Lattice:
    name: str  # 'square' or 'triangular'
    a0: float = 1.0
    # If you want to restrict the BZ integration domain, specify k-limits (in 1/a0 units)
    kx_min: float = -math.pi
    kx_max: float = math.pi
    ky_min: float = -math.pi
    ky_max: float = math.pi

    def k_grid(self, Nk: int) -> Tuple[NDArray[float_t], NDArray[float_t]]:
        kxs = np.linspace(self.kx_min, self.kx_max, Nk, endpoint=False)
        kys = np.linspace(self.ky_min, self.ky_max, Nk, endpoint=False)
        return kxs, kys

    def high_symmetry_path(self, n_per_segment: int = 100) -> Tuple[NDArray[float_t], NDArray[float_t], List[str], List[int]]:
        """Return kx,ky along a common path with tick labels & indices.
        For 'square': Γ(0,0) → X(π,0) → M(π,π) → Γ(0,0)
        For 'triangular' (hexagonal BZ approx): Γ(0,0) → K(4π/3a,0) → M(π,π/√3) → Γ
        (Here a0 set to 1; adjust if you change lattice units.)
        """
        if self.name.lower().startswith("sq"):
            G = (0.0, 0.0)
            X = (math.pi, 0.0)
            M = (math.pi, math.pi)
            pts = [G, X, M, G]
            labels = ["Γ", "X", "M", "Γ"]
        else:  # 'triangular'
            G = (0.0, 0.0)
            K = (4.0 * math.pi / 3.0, 0.0)
            M = (math.pi, math.pi / math.sqrt(3.0))
            pts = [G, K, M, G]
            labels = ["Γ", "K", "M", "Γ"]
        kxs: List[float] = []
        kys: List[float] = []
        ticks: List[int] = [0]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
                kxs.append((1 - t) * x0 + t * x1)
                kys.append((1 - t) * y0 + t * y1)
            ticks.append(len(kxs))
        return np.asarray(kxs), np.asarray(kys), labels, ticks


# =============================
# 3) Layer specifications
# =============================

@dataclass
class LayerParams:
    kind: str  # 'F' or 'N'
    onsite: NDArray[float_t]  # shape (5,) on-site energies for d-orbitals
    exchange: NDArray[float_t]  # shape (5,) exchange J per orbital (zero for N)
    inplane_neighbors: List[Neighbor]  # neighbors within this layer

    # Interlayer couplings (to nearest layer above/below). If None, no coupling.
    hop_up: Optional[NDArray[complex_t]] = None   # 5x5 orbital matrix
    hop_dn: Optional[NDArray[complex_t]] = None   # 5x5 orbital matrix


@dataclass
class ModelParams:
    lattice: Lattice
    layers: List[LayerParams]  # sequence of layers from bottom (0) to top (L-1)
    theta: float = 0.0  # magnetization polar angle (only used for F layers)
    phi: float = 0.0    # magnetization azimuth
    eta: float = 0.01   # broadening for Green's function
    ef: float = 0.0     # Fermi energy

    def n_dir(self) -> Tuple[float, float, float]:
        return unit_vec(self.theta, self.phi)


# =============================
# 4) Slater–Koster helper (minimal)
# =============================

def simple_dd_hopping_matrix(t_diag: float, t_off: float = 0.0) -> NDArray[complex_t]:
    """Return a (5x5) orbital hopping matrix with diagonal = t_diag and off-diagonal = t_off.
    This is a placeholder for full ddσ, ddπ, ddδ Slater–Koster construction.
    Replace as needed with a function that builds the 5×5 using direction cosines.
    """
    H = np.full((N_DORB, N_DORB), t_off, dtype=complex)
    np.fill_diagonal(H, t_diag)
    return H


# =============================
# 5) Single-layer H(k), velocities
# =============================

def layer_h_inplane(kx: float, ky: float, layer: LayerParams) -> NDArray[complex_t]:
    """In-plane Fourier sum H_inplane(k) = Σ_R [ T(R) e^{ik·R} + h.c. if not included separately ].
    Convention: provide neighbors only for one orientation; this function adds the Hermitian conjugate.
    """
    H_orb = np.zeros((N_DORB, N_DORB), dtype=complex)
    for nb in layer.inplane_neighbors:
        Rx, Ry = nb.R
        phase = cmath.exp(1j * (kx * Rx + ky * Ry))
        H_orb += nb.T_orb * phase + nb.T_orb.conj().T * phase.conjugate()
    # Spin identity
    return kron(H_orb, I2)


def layer_onsite_and_exchange(layer: LayerParams, n_dir: Tuple[float, float, float]) -> NDArray[complex_t]:
    eps = np.asarray(layer.onsite, dtype=float)
    J = np.asarray(layer.exchange, dtype=float)
    H_eps = kron(np.diag(eps), I2)
    if layer.kind.upper() == "F" and np.any(np.abs(J) > 0):
        Sn = spin_dir_op(n_dir)
        H_ex = kron(np.diag(J), Sn)
    else:
        H_ex = np.zeros_like(H_eps, dtype=complex)
    return H_eps + H_ex


def layer_block(kx: float, ky: float, layer: LayerParams, n_dir: Tuple[float, float, float]) -> NDArray[complex_t]:
    return layer_h_inplane(kx, ky, layer) + layer_onsite_and_exchange(layer, n_dir)


def layer_velocity_operators(kx: float, ky: float, layer: LayerParams) -> Tuple[NDArray[complex_t], NDArray[complex_t]]:
    """Analytical vα from ∂/∂kα of Fourier sum: vα = i Σ_R Rα [ T(R) e^{ik·R} − h.c. ] (note Hermiticity).
    """
    vx_orb = np.zeros((N_DORB, N_DORB), dtype=complex)
    vy_orb = np.zeros((N_DORB, N_DORB), dtype=complex)
    for nb in layer.inplane_neighbors:
        Rx, Ry = nb.R
        phase = cmath.exp(1j * (kx * Rx + ky * Ry))
        # derivative brings factor i*Rα; Hermitian companion is −i*Rα
        vx_orb += 1j * Rx * (nb.T_orb * phase - nb.T_orb.conj().T * phase.conjugate())
        vy_orb += 1j * Ry * (nb.T_orb * phase - nb.T_orb.conj().T * phase.conjugate())
    return kron(vx_orb, I2), kron(vy_orb, I2)


# =============================
# 6) Stack assembly
# =============================

def assemble_stack(
    kx: float,
    ky: float,
    mp: ModelParams,
) -> Tuple[NDArray[complex_t], NDArray[complex_t], NDArray[complex_t]]:
    """Build full H(k), vx(k), vy(k) for the multilayer stack.
    Layers are coupled only to nearest neighbors with layer.hop_up / hop_dn (orbital 5×5 each).
    """
    L = len(mp.layers)
    dS = N_DORB * N_SPIN
    dim = L * dS
    H = np.zeros((dim, dim), dtype=complex)
    vx = np.zeros_like(H)
    vy = np.zeros_like(H)

    n_dir = mp.n_dir()

    def sl(l: int) -> slice:
        i0 = l * dS
        return slice(i0, i0 + dS)

    # Diagonal blocks & velocities
    for l, layer in enumerate(mp.layers):
        H_block = layer_block(kx, ky, layer, n_dir)
        vx_block, vy_block = layer_velocity_operators(kx, ky, layer)
        H[sl(l), sl(l)] = H_block
        vx[sl(l), sl(l)] = vx_block
        vy[sl(l), sl(l)] = vy_block

    # Off-diagonal interlayer couplings (k-independent)
    for l, layer in enumerate(mp.layers):
        if layer.hop_up is not None and (l + 1) < L:
            T_up = kron(layer.hop_up, I2)
            H[sl(l), sl(l + 1)] += T_up
            H[sl(l + 1), sl(l)] += T_up.conj().T
        if layer.hop_dn is not None and (l - 1) >= 0:
            T_dn = kron(layer.hop_dn, I2)
            H[sl(l), sl(l - 1)] += T_dn
            H[sl(l - 1), sl(l)] += T_dn.conj().T

    return H, vx, vy


# =============================
# 7) Green's function & spectral objects
# =============================

def greens_function(E: float, H: NDArray[complex_t], eta: float) -> NDArray[complex_t]:
    dim = H.shape[0]
    z = (E + 1j * eta) * np.eye(dim, dtype=complex)
    return np.linalg.inv(z - H)


def spectral_function(E: float, H: NDArray[complex_t], eta: float) -> NDArray[complex_t]:
    G = greens_function(E, H, eta)
    # A = -2 Im G^R (physicists use various conventions). We'll use A = -1/π Im G for DOS convenience.
    return -(1.0 / math.pi) * G.imag


# =============================
# 8) Projectors (layer, spin)
# =============================

def layer_projector(L: int, target: int) -> NDArray[complex_t]:
    dS = N_DORB * N_SPIN
    dim = L * dS
    P = np.zeros((dim, dim), dtype=complex)
    i0 = target * dS
    P[i0 : i0 + dS, i0 : i0 + dS] = np.eye(dS, dtype=complex)
    return P


def spin_op_embedded(L: int, S: NDArray[complex_t]) -> NDArray[complex_t]:
    """Embed a 2×2 spin operator into orbital⊗spin for each layer (block-diagonal)."""
    dS = N_DORB * N_SPIN
    dim = L * dS
    S_emb = np.zeros((dim, dim), dtype=complex)
    for l in range(L):
        i0 = l * dS
        S_layer = kron(np.eye(N_DORB, dtype=complex), S)
        S_emb[i0 : i0 + dS, i0 : i0 + dS] = S_layer
    return S_emb


# =============================
# 9) DOS & observables
# =============================

def total_dos(E: float, H: NDArray[complex_t], eta: float) -> float:
    G = greens_function(E, H, eta)
    return float(-(1.0 / math.pi) * np.trace(G).imag)


def layer_dos(E: float, H: NDArray[complex_t], eta: float, L: int, l_idx: int) -> float:
    G = greens_function(E, H, eta)
    P = layer_projector(L, l_idx)
    return float(-(1.0 / math.pi) * np.trace(P @ G).imag)


def spin_resolved_dos(
    E: float, H: NDArray[complex_t], eta: float, mp: ModelParams, up: bool = True
) -> float:
    G = greens_function(E, H, eta)
    Pup, Pdn = spin_projectors(mp.n_dir())
    S = Pup if up else Pdn
    S_emb = spin_op_embedded(len(mp.layers), S)
    return float(-(1.0 / math.pi) * np.trace(S_emb @ G).imag)


# =============================
# 10) Kubo-Greenwood conductivity
# =============================

def kubo_greenwood_sigma(
    E: float, H: NDArray[complex_t], vx: NDArray[complex_t], vy: NDArray[complex_t], eta: float
) -> Tuple[float, float, float]:
    """Return (σxx, σyy, σxy) at energy E using A(E) = -(1/π) Im G and σ = π Tr[v A v A].
    Units: set e^2/ħ=1 and volume=1. For actual units, scale externally.
    """
    G = greens_function(E, H, eta)
    A = -(1.0 / math.pi) * G.imag
    # Hermiticity checks skipped for speed
    sigma_xx = float(math.pi * np.trace(vx @ A @ vx @ A).real)
    sigma_yy = float(math.pi * np.trace(vy @ A @ vy @ A).real)
    sigma_xy = float(math.pi * np.trace(vx @ A @ vy @ A).real)
    return sigma_xx, sigma_yy, sigma_xy


# =============================
# 11) k-space integration helpers
# =============================

def integrate_over_bz(
    mp: ModelParams,
    Nk: int,
    integrand,
) -> Tuple[float, int]:
    """Numerically integrate an integrand(kx,ky) over the rectangular BZ window with uniform weights."""
    kxs, kys = mp.lattice.k_grid(Nk)
    acc = 0.0
    count = 0
    for kx in kxs:
        for ky in kys:
            acc += integrand(kx, ky)
            count += 1
    # Normalize by number of k-points (BZ volume cancels for uniform grids & dimensionless units)
    return acc / count, count


# =============================
# 12) Band structure along path
# =============================

def band_structure_along_path(
    mp: ModelParams, n_per_segment: int = 100
) -> Tuple[NDArray[float_t], NDArray[float_t]]:
    kxs, kys, _, _ = mp.lattice.high_symmetry_path(n_per_segment)
    L = len(mp.layers)
    dim = L * N_DORB * N_SPIN
    bands = np.zeros((len(kxs), dim), dtype=float)
    for i, (kx, ky) in enumerate(zip(kxs, kys)):
        H, _, _ = assemble_stack(kx, ky, mp)
        w, _ = np.linalg.eigh(H)
        bands[i, :] = w.real
    # Return the arclength coordinate and energies
    # For simplicity, use index as x-coordinate; caller can add ticks from high_symmetry_path()
    xs = np.arange(len(kxs), dtype=float)
    return xs, bands


# =============================
# 13) Quick-start builder
# =============================

def default_square_neighbors(t: float = -0.3) -> List[Neighbor]:
    """Nearest neighbors on a square lattice at (±1,0),(0,±1) with simple diagonal orbital hopping t."""
    T = simple_dd_hopping_matrix(t_diag=t, t_off=0.0)
    vecs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    return [Neighbor(R=v, T_orb=T) for v in vecs]


def default_triangular_neighbors(t: float = -0.3) -> List[Neighbor]:
    """Nearest neighbors on a triangular lattice (a0=1) with 6 vectors."""
    T = simple_dd_hopping_matrix(t_diag=t, t_off=0.0)
    vecs = [
        (1.0, 0.0),
        (0.5, math.sqrt(3) / 2.0),
        (-0.5, math.sqrt(3) / 2.0),
        (-1.0, 0.0),
        (-0.5, -math.sqrt(3) / 2.0),
        (0.5, -math.sqrt(3) / 2.0),
    ]
    return [Neighbor(R=v, T_orb=T) for v in vecs]


def build_demo_model(
    lattice_name: str = "square",
    nF: int = 2,
    nN: int = 2,
    t_inplane: float = -0.3,
    t_perp: float = -0.1,
    J_F: float = 0.4,
    eps_F: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    eps_N: Sequence[float] = (0.2, 0.2, 0.2, 0.2, 0.2),
    theta: float = math.radians(25.0),
    phi: float = math.radians(30.0),
    eta: float = 0.02,
    ef: float = 0.0,
) -> ModelParams:
    """Construct a simple F | N | F | N stack demo with diagonal-orbital hoppings.
    You can replace the neighbor lists with full SK-derived (5×5) matrices.
    """
    if lattice_name.lower().startswith("tri"):
        lat = Lattice("triangular")
        nbs = default_triangular_neighbors(t=t_inplane)
    else:
        lat = Lattice("square")
        nbs = default_square_neighbors(t=t_inplane)

    T_perp = simple_dd_hopping_matrix(t_diag=t_perp, t_off=0.0)

    layers: List[LayerParams] = []

    # Ferromagnet prototype layer
    F_layer = LayerParams(
        kind="F",
        onsite=np.array(eps_F, dtype=float),
        exchange=np.array([J_F, J_F, J_F, J_F, J_F], dtype=float),
        inplane_neighbors=nbs,
        hop_up=T_perp,
        hop_dn=T_perp,
    )

    # Normal metal prototype layer
    N_layer = LayerParams(
        kind="N",
        onsite=np.array(eps_N, dtype=float),
        exchange=np.zeros(5, dtype=float),
        inplane_neighbors=nbs,
        hop_up=T_perp,
        hop_dn=T_perp,
    )

    # Build stack: e.g., F F N N
    for _ in range(nF):
        layers.append(F_layer)
    for _ in range(nN):
        layers.append(N_layer)

    mp = ModelParams(lattice=lat, layers=layers, theta=theta, phi=phi, eta=eta, ef=ef)
    return mp


# =============================
# 14) Example: DOS & σ on a grid
# =============================

def example_compute_dos_and_sigma():
    """Small demo computation: total DOS(E) and σxx(E) over a grid (coarse for speed)."""
    mp = build_demo_model(lattice_name="square", nF=2, nN=2)
    Nk = 20  # k-grid per axis (increase for convergence)
    Es = np.linspace(-2.0, 2.0, 121)
    dos = np.zeros_like(Es)
    sxx = np.zeros_like(Es)

    for iE, E in enumerate(Es):
        def integrand_dos(kx, ky):
            H, _, _ = assemble_stack(kx, ky, mp)
            return total_dos(E, H, mp.eta)

        def integrand_sigma(kx, ky):
            H, vx, vy = assemble_stack(kx, ky, mp)
            sx, _, _ = kubo_greenwood_sigma(E, H, vx, vy, mp.eta)
            return sx

        dos[iE], _ = integrate_over_bz(mp, Nk, integrand_dos)
        sxx[iE], _ = integrate_over_bz(mp, Nk, integrand_sigma)

    return Es, dos, sxx


# =============================
# 15) Example: band structure
# =============================

def example_band_structure():
    mp = build_demo_model(lattice_name="square", nF=1, nN=0)
    xs, bands = band_structure_along_path(mp, n_per_segment=60)
    return xs, bands


if __name__ == "__main__":
    # Minimal smoke test (does not plot by default)
    mp = build_demo_model()
    kx, ky = 0.1, -0.2
    H, vx, vy = assemble_stack(kx, ky, mp)
    E0 = 0.0
    G = greens_function(E0, H, mp.eta)
    A = spectral_function(E0, H, mp.eta)
    sxx, syy, sxy = kubo_greenwood_sigma(E0, H, vx, vy, mp.eta)
    print("dim(H)=", H.shape, " Tr A=", np.trace(A).real, " sxx=", sxx)
    # Uncomment to run quick examples (requires matplotlib for plotting)
    # Es, dos, sxx = example_compute_dos_and_sigma()
    # xs, bands = example_band_structure()
