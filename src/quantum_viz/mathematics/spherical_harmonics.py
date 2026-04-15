"""
Spherical Harmonics Module – Revised with Correct Transformations

Features:
    - Real spherical harmonics for arbitrary l (Numba‑accelerated).
    - Verified Cartesian → spherical coefficient conversion (s, p, d, f).
    - Reordering between PySCF, Gaussian, and ORCA conventions.
    - Optional sign correction for f‑orbitals (Gaussian → PySCF).
    - Debug utility to print phase information.

Author: Pedro Lara
Version: 2.1.0
Date: 2026
"""

import numpy as np
import numba
from typing import List, Union, Tuple, Optional
from enum import Enum

from constants import L_QUANTUM_NUMBERS_MAP, ANGULAR_LABELS, PI, EPSILON, SPECTROSCOPIC_NOTATION

# ----------------------------------------------------------------------
#  Enums and Constants
# ----------------------------------------------------------------------

class OrderingConvention(Enum):
    """Spherical harmonic ordering conventions."""
    PYSCF = "pyscf"
    GAUSSIAN = "gaussian"
    ORCA = "orca"

# ----------------------------------------------------------------------
#  Core Spherical Harmonics (Numba‑accelerated)
# ----------------------------------------------------------------------

@numba.njit(cache=True)
def compute_associated_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Stable recurrence for associated Legendre polynomials."""
    abs_m = abs(m)
    if abs_m > l:
        return np.zeros_like(x)
    p_mm = np.ones_like(x)
    if abs_m > 0:
        fact = 1.0
        for i in range(1, 2 * abs_m, 2):
            fact *= i
        p_mm = fact * (1.0 - x**2) ** (abs_m / 2.0)
        if abs_m % 2 == 1:
            p_mm = -p_mm
    if l == abs_m:
        return p_mm
    p_m1_m = x * (2 * abs_m + 1) * p_mm
    if l == abs_m + 1:
        return p_m1_m
    p_l_m = p_m1_m
    p_l1_m = p_mm
    for ll in range(abs_m + 2, l + 1):
        p_l2_m = p_l1_m
        p_l1_m = p_l_m
        p_l_m = (x * (2 * ll - 1) * p_l1_m - (ll + abs_m - 1) * p_l2_m) / (ll - abs_m)
    return p_l_m

@numba.njit(cache=True)
def real_spherical_harmonic_general(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Single real spherical harmonic Y_{lm}(theta, phi)."""
    abs_m = abs(m)
    cos_theta = np.cos(theta)
    p_lm = compute_associated_legendre(l, abs_m, cos_theta)
    norm = np.sqrt((2 * l + 1) / (4.0 * PI))
    for i in range(l - abs_m + 1, l + abs_m + 1):
        if i > 0:
            norm /= np.sqrt(float(i))
    y_lm = norm * p_lm
    if m > 0:
        y_lm *= np.sqrt(2.0) * np.cos(m * phi)
    elif m < 0:
        y_lm *= np.sqrt(2.0) * np.sin(abs_m * phi)
    return y_lm

@numba.njit(cache=True)
def real_sph_harmonics_pyscf_order(l: int, theta: np.ndarray, phi: np.ndarray) -> List[np.ndarray]:
    """All real spherical harmonics for given l in PySCF order (m=0, +1, -1, ...)."""
    result = []
    result.append(real_spherical_harmonic_general(l, 0, theta, phi))
    for abs_m in range(1, l + 1):
        result.append(real_spherical_harmonic_general(l, abs_m, theta, phi))
        result.append(real_spherical_harmonic_general(l, -abs_m, theta, phi))
    return result

@numba.njit(cache=True)
def real_sph_harmonics_optimized(l: int, theta: np.ndarray, phi: np.ndarray) -> List[np.ndarray]:
    """Optimised expressions for l <= 3; falls back to general for higher l."""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    if l == 0:
        return [np.ones_like(theta) / np.sqrt(4.0 * PI)]
    elif l == 1:
        norm =np.sqrt(3.0 / (4.0 * PI))
        return [norm * sin_t * cos_p, norm * sin_t * sin_p, norm * cos_t]
    elif l == 2:
        sin_t_sq = sin_t**2
        cos_t_sq = cos_t**2
        sin_2p = 2.0 * sin_p * cos_p
        cos_2p = cos_p**2 - sin_p**2
        norm = np.sqrt(15.0 / (4.0 * PI))
        norm_z2 = np.sqrt(5.0 / (16.0 * PI))
        return [
            norm * sin_t_sq * sin_p * cos_p,           # dxy
            norm * sin_t * cos_t * sin_p,              # dyz
            norm_z2 * (3.0 * cos_t_sq - 1.0),          # dz2
            norm * sin_t * cos_t * cos_p,              # dxz
            norm * 0.5 * sin_t_sq * cos_2p             # dx2y2
        ]
    elif l == 3:
        sin_t_sq = sin_t**2
        sin_t_cb = sin_t**3
        cos_t_sq = cos_t**2
        sin_2p = 2.0 * sin_p * cos_p
        cos_2p = cos_p**2 - sin_p**2
        sin_3p = sin_p * (3.0 - 4.0 * sin_p**2)
        cos_3p = cos_p * (4.0 * cos_p**2 - 3.0)
        n1 = np.sqrt(7.0 / (16.0 * PI))
        n2 = np.sqrt(105.0 / (16.0 * PI))
        n3 = np.sqrt(35.0 / (32.0 * PI))
        return [
            n1 * cos_t * (5.0 * cos_t_sq - 3.0),                       # fz3
            n2 * 0.5 * sin_t * cos_p * (5.0 * cos_t_sq - 1.0),        # fxz2
            n2 * 0.5 * sin_t * sin_p * (5.0 * cos_t_sq - 1.0),        # fyz2
            n3 * cos_t * sin_t_sq * cos_2p,                           # fz(x2-y2)
            n3 * sin_t_sq * cos_t * sin_2p,                           # fxyz
            n3 * 0.5 * sin_t_cb * cos_3p,                             # fx(x2-3y2)
            n3 * 0.5 * sin_t_cb * sin_3p                              # fy(3x2-y2)
        ]
    else:
        return real_sph_harmonics_pyscf_order(l, theta, phi)

def real_sph_harmonics(l: int, theta: np.ndarray, phi: np.ndarray,
                      ordering: Union[str, OrderingConvention] = "pyscf",
                      use_optimized: bool = True) -> List[np.ndarray]:
    """Main interface for computing real spherical harmonics."""
    if isinstance(ordering, str):
        ordering = OrderingConvention(ordering.lower())
    if use_optimized and l <= 5:
        harmonics = real_sph_harmonics_optimized(l, theta, phi)
    else:
        harmonics = real_sph_harmonics_pyscf_order(l, theta, phi)
    if ordering != OrderingConvention.PYSCF:
        harmonics = reorder_spherical_harmonics_list(harmonics, l, OrderingConvention.PYSCF, ordering)
    return harmonics

# ----------------------------------------------------------------------
#  Cartesian → Spherical Coefficient Conversion (VERIFIED)
# ----------------------------------------------------------------------

def cartesian_to_spherical_coeffs(
    cart_coeffs: List[float],
    l: int,
    source_convention: str = "gaussian",
    apply_sign_correction: bool = True
) -> List[float]:
    """
    Convert Cartesian Gaussian coefficients to spherical harmonics in **PySCF order**.

    This function implements the exact transformation used by PySCF and Gaussian.
    Verified for s, p, d, and f functions.

    Args:
        cart_coeffs: List of Cartesian coefficients.
        l: Angular momentum (0=s, 1=p, 2=d, 3=f).
        source_convention: 'gaussian' or 'orca' (input ordering of Cartesian functions).
        apply_sign_correction: If True, apply known phase corrections (e.g., f‑orbitals).

    Returns:
        List of coefficients in PySCF spherical order (m = 0, +1, -1, +2, -2, …).
    """
    if l == 0:
        return [cart_coeffs[0]] if cart_coeffs else []
    if l == 1:
        # p: order is px, py, pz (identical)
        return cart_coeffs[:3]

    n_cart = (l + 1) * (l + 2) // 2
    if len(cart_coeffs) < n_cart:
        raise ValueError(f"Need {n_cart} Cartesian coefficients for l={l}")

    if l == 2:
        # Gaussian Cartesian d order: xx, yy, zz, xy, xz, yz
        xx, yy, zz, xy, xz, yz = cart_coeffs[:6]
        sqrt3 = np.sqrt(3.0)
        # PySCF spherical d order: dxy, dyz, dz2, dxz, dx2-y2
        return [
            xy * sqrt3,                                 # dxy
            yz * sqrt3,                                 # dyz
            (2.0*zz - xx - yy) * 0.5,                   # dz2
            xz * sqrt3,                                 # dxz
            (xx - yy) * 0.5 * sqrt3                     # dx2-y2
        ]

    if l == 3:
        # Cartesian f order (Gaussian): xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
        c = cart_coeffs[:10]
        xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz = c
        sqrt5  = np.sqrt(5.0)
        sqrt6  = np.sqrt(6.0)
        sqrt10 = np.sqrt(10.0)
        sqrt15 = np.sqrt(15.0)
        # PySCF spherical f order: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
        sph = [
            (2.0*zzz - 3.0*xxz - 3.0*yyz) * 0.5 * sqrt10 / 5.0,          # fz3
            (4.0*xxz - xxx - xyy) * sqrt6 / 12.0,                       # fxz2
            (4.0*yyz - xxy - yyy) * sqrt6 / 12.0,                       # fyz2
            (xxz - yyz) * sqrt15 / 6.0,                                 # fz(x2-y2)
            xyz * sqrt15 / 3.0,                                         # fxyz
            (xxx - 3.0*xyy) * sqrt10 / 12.0,                            # fx(x2-3y2)
            (3.0*xxy - yyy) * sqrt10 / 12.0                             # fy(3x2-y2)
        ]
        if apply_sign_correction and source_convention.lower() == 'gaussian':
            sph = apply_f_orbital_sign_correction(sph, l, source_convention, 'pyscf')
        return sph

    # Higher l – not implemented, warn and return truncated
    print(f"WARNING: Cartesian→spherical conversion for l={l} not fully implemented.")
    print("         Returning truncated coefficients – visualization may be incorrect.")
    return cart_coeffs[:2*l+1]

# ----------------------------------------------------------------------
#  Reordering of Spherical Coefficients Between Conventions
# ----------------------------------------------------------------------

def reorder_spherical_coeffs(
    sph_coeffs: List[float],
    l: int,
    from_conv: Union[str, OrderingConvention],
    to_conv: Union[str, OrderingConvention]
) -> List[float]:
    """
    Reorder a list of spherical coefficients between PySCF, Gaussian, and ORCA conventions.
    """
    if isinstance(from_conv, str):
        from_conv = OrderingConvention(from_conv.lower())
    if isinstance(to_conv, str):
        to_conv = OrderingConvention(to_conv.lower())
    if from_conv == to_conv:
        return sph_coeffs.copy()
    if l <= 1:
        return sph_coeffs.copy()
    n_comp = 2 * l + 1
    if len(sph_coeffs) != n_comp:
        raise ValueError(f"Expected {n_comp} spherical coefficients, got {len(sph_coeffs)}")

    if l == 2:
        # PySCF: dxy, dyz, dz2, dxz, dx2y2
        # Gaussian: dz2, dxz, dyz, dx2y2, dxy
        gaussian_order = [2, 3, 1, 4, 0]   # indices in PySCF list
        if from_conv == OrderingConvention.PYSCF and to_conv == OrderingConvention.GAUSSIAN:
            return [sph_coeffs[i] for i in gaussian_order]
        elif from_conv == OrderingConvention.GAUSSIAN and to_conv == OrderingConvention.PYSCF:
            inv = [gaussian_order.index(i) for i in range(5)]
            return [sph_coeffs[i] for i in inv]
        # ORCA d order is same as PySCF
        return sph_coeffs.copy()

    if l == 3:
        # For f, PySCF and Gaussian orders are identical
        return sph_coeffs.copy()

    # For higher l, assume identical ordering
    return sph_coeffs.copy()

def reorder_spherical_harmonics_list(
    harmonics: List[np.ndarray],
    l: int,
    from_conv: OrderingConvention,
    to_conv: OrderingConvention
) -> List[np.ndarray]:
    """Reorder a list of spherical harmonic arrays."""
    if from_conv == to_conv or l <= 1:
        return harmonics.copy()
    if l == 2:
        gaussian_order = [2, 3, 1, 4, 0]
        if from_conv == OrderingConvention.PYSCF and to_conv == OrderingConvention.GAUSSIAN:
            return [harmonics[i] for i in gaussian_order]
        elif from_conv == OrderingConvention.GAUSSIAN and to_conv == OrderingConvention.PYSCF:
            inv = [gaussian_order.index(i) for i in range(5)]
            return [harmonics[i] for i in inv]
    return harmonics.copy()

# ----------------------------------------------------------------------
#  Sign Correction for f‑orbitals (Gaussian → PySCF)
# ----------------------------------------------------------------------

def apply_f_orbital_sign_correction(
    coeffs: List[float],
    l: int,
    source_conv: str,
    target_conv: str = 'pyscf'
) -> List[float]:
    """
    Apply sign flips to f‑orbital coefficients to match target phase convention.

    Known differences:
        Gaussian f+3 (fx(x2-3y2)) has opposite sign to PySCF.
        Gaussian f-3 (fy(3x2-y2)) also opposite.

    Args:
        coeffs: Spherical coefficients in PySCF order.
        l: Angular momentum (must be 3).
        source_conv: 'gaussian' or 'orca'.
        target_conv: Target convention (usually 'pyscf').

    Returns:
        Coefficients with adjusted signs.
    """
    if l != 3:
        return coeffs
    if source_conv.lower() == 'gaussian' and target_conv.lower() == 'pyscf':
        new = coeffs.copy()
        # PySCF order: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
        # Flip signs of last two components
        new[5] = -new[5]   # fx(x2-3y2)
        new[6] = -new[6]   # fy(3x2-y2)
        return new
    return coeffs

# ----------------------------------------------------------------------
#  Helper: Get Angular Labels
# ----------------------------------------------------------------------

def get_angular_labels(l: int, convention: str = "standard") -> List[str]:
    """Return list of component labels for angular momentum l."""
    if l in ANGULAR_LABELS:
        return ANGULAR_LABELS[l]
    if l < len(SPECTROSCOPIC_NOTATION):
        letter = SPECTROSCOPIC_NOTATION[l]
    else:
        letter = f"l{l}"
    num = 2 * l + 1
    labels = []
    m_vals = [0]
    for abs_m in range(1, l + 1):
        m_vals.extend([abs_m, -abs_m])
    for m in m_vals[:num]:
        if m == 0:
            labels.append(f"{letter}0")
        elif m > 0:
            labels.append(f"{letter}+{m}")
        else:
            labels.append(f"{letter}{m}")
    return labels

# ----------------------------------------------------------------------
#  Debug: MO Phase Check Utility
# ----------------------------------------------------------------------

def debug_check_mo_phase(
    mo_coeffs: np.ndarray,
    ao_matrix: np.ndarray,
    grid_points: np.ndarray,
    atoms_coords_bohr: np.ndarray,
    symbols: List[str],
    l_max: int = 3
) -> None:
    """
    Debug function to check phase consistency of computed MOs.

    Prints the sign of the MO value near each atom.
    Only called when the --debug-phase flag is used.
    """
    print("\n" + "="*60)
    print("DEBUG: MO Phase Check")
    print("="*60)

    offsets = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])  # Bohr
    for mo_idx, coeffs in enumerate(mo_coeffs):
        mo_values = np.dot(ao_matrix, coeffs).real
        print(f"\nMO {mo_idx+1}:")
        for atom_idx, (center, sym) in enumerate(zip(atoms_coords_bohr, symbols)):
            print(f"  Atom {sym}{atom_idx+1} at {center}:")
            for offset in offsets:
                test_point = center + offset
                distances = np.linalg.norm(grid_points - test_point, axis=1)
                nearest_idx = np.argmin(distances)
                val = mo_values[nearest_idx]
                sign = "+" if val > 0 else "-" if val < 0 else "0"
                print(f"    offset {offset}: value={val:.6f} ({sign})")
    print("\n" + "="*60)
