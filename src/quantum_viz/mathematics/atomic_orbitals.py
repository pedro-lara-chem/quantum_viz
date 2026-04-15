"""
Atomic Orbital Computation Module

This module provides high-performance computation of atomic orbitals (AOs)
on 3D grids using Numba-accelerated functions. It supports arbitrary
angular momentum quantum numbers and various basis set conventions.

Features:
    - Efficient AO evaluation on arbitrary 3D grids
    - Support for s, p, d, f, g, h, i orbitals and beyond
    - Automatic handling of general contraction shells (e.g., 'sp')
    - Numba JIT compilation for performance
    - Progress tracking for large computations
    - Memory-efficient batch processing for large grids

Classes:
    AtomicOrbitalComputer: Main class for AO computation
    AOResult: Container for computation results

Functions:
    compute_atomic_orbitals: Convenience function for AO computation
    batch_compute_aos: Compute AOs in batches for memory efficiency

Author: Pedro Lara
Version: 2.0.0
Date: 2024
"""


import numpy as np
import numba
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
from tqdm import tqdm

from quantum_viz.constants import L_QUANTUM_NUMBERS_MAP, EPSILON, PI
from quantum_viz.mathematics.normalization import apply_normalization_factor, detect_normalization_convention
from quantum_viz.mathematics.spherical_harmonics import real_sph_harmonics_optimized, get_angular_labels


# ==============================================================================
# Numba-compatible spherical harmonics
# ==============================================================================

@numba.njit(cache=True)
def compute_associated_legendre_numba(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Numba-compatible associated Legendre polynomials."""
    abs_m = abs(m)
    
    if abs_m > l:
        return np.zeros_like(x)
    
    p_mm = np.ones_like(x)
    if abs_m > 0:
        fact = 1.0
        for i in range(1, 2 * abs_m, 2):
            fact *= i
        p_mm = fact * (1.0 - x**2)**(abs_m / 2.0)
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
def real_spherical_harmonic_numba(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Numba-compatible single real spherical harmonic."""
    abs_m = abs(m)
    
    cos_theta = np.cos(theta)
    p_lm = compute_associated_legendre_numba(l, abs_m, cos_theta)
    
    norm = np.sqrt((2 * l + 1) / (4.0 * PI))
    
    for i in range(l - abs_m + 1, l + abs_m + 1):
        if i > 0:
            norm /= np.sqrt(float(i))
    
    y_lm = norm * p_lm
    
    if m > 0:
        y_lm = y_lm * np.sqrt(2.0) * np.cos(m * phi)
    elif m < 0:
        y_lm = y_lm * np.sqrt(2.0) * np.sin(abs_m * phi)
    
    return y_lm


@numba.njit(cache=True)
def get_spherical_harmonic_component(l: int, m_idx: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Get a specific spherical harmonic component by index (PySCF ordering)."""
    if m_idx == 0:
        return real_spherical_harmonic_numba(l, 0, theta, phi)
    
    abs_m = (m_idx + 1) // 2
    if abs_m > l:
        return np.zeros_like(theta)
    
    if m_idx % 2 == 1:
        return real_spherical_harmonic_numba(l, abs_m, theta, phi)
    else:
        return real_spherical_harmonic_numba(l, -abs_m, theta, phi)


@numba.njit(cache=True)
def real_sph_harmonics_optimized_numba(l: int, m_idx: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Optimized computation of a single spherical harmonic component."""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    if l == 0:
        return np.ones_like(theta) / np.sqrt(4.0 * PI)
    
    elif l == 1:
        norm = np.sqrt(3.0 / (4.0 * PI))
        if m_idx == 0:
            return norm * sin_t * cos_p
        elif m_idx == 1:
            return norm * sin_t * sin_p
        else:
            return norm * cos_t
    
    elif l == 2:
        sin_t_sq = sin_t**2
        cos_t_sq = cos_t**2
        sin_2p = 2.0 * sin_p * cos_p
        cos_2p = cos_p**2 - sin_p**2
        
        norm = np.sqrt(15.0 / (4.0 * PI))
        norm_z2 = np.sqrt(5.0 / (16.0 * PI))
        
        if m_idx == 0:
            return norm * sin_t_sq * sin_p * cos_p
        elif m_idx == 1:
            return norm * sin_t * cos_t * sin_p
        elif m_idx == 2:
            return norm_z2 * (3.0 * cos_t_sq - 1.0)
        elif m_idx == 3:
            return norm * sin_t * cos_t * cos_p
        else:
            return norm * 0.5 * sin_t_sq * cos_2p
    
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
        
        if m_idx == 0:
            return n1 * cos_t * (5.0 * cos_t_sq - 3.0)
        elif m_idx == 1:
            return n2 * 0.5 * sin_t * cos_p * (5.0 * cos_t_sq - 1.0)
        elif m_idx == 2:
            return n2 * 0.5 * sin_t * sin_p * (5.0 * cos_t_sq - 1.0)
        elif m_idx == 3:
            return n3 * cos_t * sin_t_sq * cos_2p
        elif m_idx == 4:
            return n3 * sin_t_sq * cos_t * sin_2p
        elif m_idx == 5:
            return n3 * 0.5 * sin_t_cb * cos_3p
        else:
            return n3 * 0.5 * sin_t_cb * sin_3p
    
    else:
        return get_spherical_harmonic_component(l, m_idx, theta, phi)


# ==============================================================================
# Radial and AO computation functions
# ==============================================================================

@numba.njit(cache=True)
def compute_radial_part(r_sq: np.ndarray, exponents: np.ndarray, 
                       coeffs: np.ndarray, scale_factor_sq: float,
                       l: int, apply_norm: bool) -> np.ndarray:
    """Compute the radial part of atomic orbitals."""
    num_points = r_sq.shape[0]
    result = np.zeros(num_points, dtype=np.float64)
    for i in range(exponents.shape[0]):
        alpha = exponents[i]
        coeff = coeffs[i]
        if abs(coeff) < EPSILON:
            continue
        alpha_scaled = alpha * scale_factor_sq
        if apply_norm:
            coeff = apply_normalization_factor(alpha_scaled, coeff, l, True)
        result += coeff * np.exp(-alpha_scaled * r_sq)
    return result


@numba.njit(cache=True)
def compute_single_ao(r_sq: np.ndarray, r_stable: np.ndarray,
                     theta: np.ndarray, phi: np.ndarray,
                     exponents: np.ndarray, coeffs: np.ndarray,
                     scale_factor_sq: float, l: int, m_idx: int,
                     apply_norm: bool) -> np.ndarray:
    
    radial = compute_radial_part(r_sq, exponents, coeffs, scale_factor_sq, l, apply_norm)
    
    # Get the standard physics Y_lm
    y_lm = real_sph_harmonics_optimized(l, theta, phi)[m_idx]  
    
    # Apply the Quantum Chemistry Solid Harmonic scaling factor
    qc_angular_scale = np.sqrt(4.0 * PI / (2 * l + 1))
    angular = y_lm * qc_angular_scale
    
    r_pow_l = r_stable ** l
    return radial * r_pow_l * angular

# ==============================================================================
# Main AO Computer Class
# ==============================================================================

@dataclass
class AOResult:
    ao_matrix: np.ndarray
    ao_labels: List[str]
    computation_time: float
    num_aos: int
    num_grid_points: int
    normalization_applied: bool
    max_l_computed: int


class AtomicOrbitalComputer:
    """High‑performance AO computer with debug capabilities."""

    def __init__(self, atoms_data: List[Any], gto_data: List[Any],
                 basis_info: Optional[Any] = None,
                 show_progress: bool = True,
                 debug: bool = False):
        """
        Args:
            atoms_data: List of AtomData objects.
            gto_data: List of GTOData objects.
            basis_info: BasisSetInfo from parser.
            show_progress: Show progress bar.
            debug: If True, print detailed AO specifications.
        """
        self.atoms_data = atoms_data
        self.gto_data = gto_data
        self.basis_info = basis_info
        self.show_progress = show_progress
        self.debug = debug

        self.normalization_result = detect_normalization_convention(gto_data, basis_info)
        self.ao_specs = self._build_ao_specifications()

        if self.debug:
            self._print_ao_specs()
    
    def _has_coefficients(self, shell: Any, coeff_idx: int) -> bool:
        """
        Check if a shell has non-zero coefficients for a given component.
        """
        for prim in shell.primitives:
            if coeff_idx < len(prim['coefficients']):
                if abs(prim['coefficients'][coeff_idx]) > EPSILON:
                    return True
        return False
    
    def _build_ao_specifications(self) -> List[Dict]:
        """
        Build AO specifications.
        Handles combined shells like 'sp', 'spd' correctly by iterating over each character.
        """
        specs = []
        for atom_gto in self.gto_data:
            atom_idx = atom_gto.atom_index
            atom = self.atoms_data[atom_idx]
            atom_center = np.array([atom.x, atom.y, atom.z], dtype=np.float64)

            for shell_idx, shell in enumerate(atom_gto.shells):
                shell_type = shell.type.lower()
                # For each angular momentum character in the shell type
                for char_idx, char_l in enumerate(shell_type):
                    if char_l not in L_QUANTUM_NUMBERS_MAP:
                        continue
                    l_val = L_QUANTUM_NUMBERS_MAP[char_l]

                    # Extract exponents and coefficients for this specific l‑component
                    exponents = []
                    coeffs = []
                    for prim in shell.primitives:
                        exponents.append(prim['exponent'])
                        if char_idx < len(prim['coefficients']):
                            coeffs.append(prim['coefficients'][char_idx])
                        else:
                            # This primitive does not have a coefficient for this l (should not happen)
                            coeffs.append(0.0)

                    # If all coefficients are zero, skip this component (saves computation)
                    if all(abs(c) < EPSILON for c in coeffs):
                        continue

                    # Determine principal quantum number for labeling
                    n_quantum = shell.n_quantum_number or (l_val + 1)
                    labels = get_angular_labels(l_val)

                    for m_idx, label in enumerate(labels):
                        full_label = f"{atom.label}{atom_idx+1}_{n_quantum}{label}"
                        specs.append({
                            'atom_idx': atom_idx,
                            'atom_center': atom_center.copy(),
                            'shell_idx': shell_idx,
                            'l': l_val,
                            'm_idx': m_idx,
                            'exponents': np.array(exponents, dtype=np.float64),
                            'coeffs': np.array(coeffs, dtype=np.float64),
                            'scale_factor_sq': shell.scale_factor ** 2,
                            'label': full_label,
                            'char_idx': char_idx,
                            'shell_type': char_l
                        })
        return specs
    
    def _print_ao_specs(self):
        """Debug: print AO specifications."""
        print("\n[DEBUG] AO Specifications:")
        print(f"Total AOs: {len(self.ao_specs)}")
        print("First 10 AOs:")
        for i, spec in enumerate(self.ao_specs[:10]):
            print(f"  {i:3d}: {spec['label']:20s} center=({spec['atom_center'][0]:8.4f}, "
                  f"{spec['atom_center'][1]:8.4f}, {spec['atom_center'][2]:8.4f}) Bohr, "
                  f"l={spec['l']}, m_idx={spec['m_idx']}")
        print("Atom centers (Bohr):")
        for i, atom in enumerate(self.atoms_data):
            print(f"  Atom {i}: {atom.label} at ({atom.x:8.4f}, {atom.y:8.4f}, {atom.z:8.4f})")

    def validate_atom_centers(self, grid_points: np.ndarray, sample_indices: List[int] = None):
        """
        Debug: Evaluate AOs at atom centers to verify they are non‑zero only near correct atoms.
        """
        if sample_indices is None:
            sample_indices = list(range(min(5, len(self.ao_specs))))
        print("\n[DEBUG] AO values at atom centers:")
        for atom_idx, atom in enumerate(self.atoms_data):
            center = np.array([atom.x, atom.y, atom.z])
            # Find grid point closest to atom center
            dist = np.linalg.norm(grid_points - center, axis=1)
            idx = np.argmin(dist)
            print(f"  Atom {atom.label}{atom_idx+1} at {center}:")
            for ao_idx in sample_indices:
                spec = self.ao_specs[ao_idx]
                # Compute AO value at this point (quick)
                R = grid_points[idx] - spec['atom_center']
                r_sq = np.sum(R**2)
                r = np.sqrt(r_sq + EPSILON)
                theta = np.arccos(np.clip(R[2]/r, -1, 1))
                phi = np.arctan2(R[1], R[0])
                val = compute_single_ao(
                    np.array([r_sq]), np.array([r]), np.array([theta]), np.array([phi]),
                    spec['exponents'], spec['coeffs'], spec['scale_factor_sq'],
                    spec['l'], spec['m_idx'], self.normalization_result.should_renormalize
                )[0]
                print(f"    AO {ao_idx:3d} ({spec['label']:20s}): {val:12.6e}")
    
    def compute(self, grid_points: np.ndarray,
                force_renormalize: Optional[bool] = None) -> AOResult:
        start_time = time.time()
        apply_norm = force_renormalize if force_renormalize is not None else self.normalization_result.should_renormalize

        num_points = grid_points.shape[0]
        num_aos = len(self.ao_specs)

        if num_aos == 0:
            return AOResult(
                ao_matrix=np.empty((num_points, 0)),
                ao_labels=[],
                computation_time=time.time()-start_time,
                num_aos=0, num_grid_points=num_points,
                normalization_applied=apply_norm,
                max_l_computed=0
            )

        ao_matrix = np.empty((num_points, num_aos), dtype=np.float64)
        ao_labels = [spec['label'] for spec in self.ao_specs]

        # Compute AOs
        iterator = enumerate(self.ao_specs)
        if self.show_progress:
            iterator = tqdm(list(iterator), total=num_aos, desc="Computing AOs")

        for ao_idx, spec in iterator:
            center = spec['atom_center']
            R = grid_points - center
            x, y, z = R[:, 0], R[:, 1], R[:, 2]
            r_sq = x**2 + y**2 + z**2
            r_stable = np.sqrt(r_sq + EPSILON)
            theta = np.arccos(np.clip(z / r_stable, -1.0, 1.0))
            phi = np.arctan2(y, x)

            ao_values = compute_single_ao(
                r_sq, r_stable, theta, phi,
                spec['exponents'], spec['coeffs'],
                spec['scale_factor_sq'], spec['l'], spec['m_idx'],
                apply_norm
            )
            ao_matrix[:, ao_idx] = ao_values

        max_l = max((spec['l'] for spec in self.ao_specs), default=0)
        comp_time = time.time() - start_time

        return AOResult(
            ao_matrix=ao_matrix,
            ao_labels=ao_labels,
            computation_time=comp_time,
            num_aos=num_aos,
            num_grid_points=num_points,
            normalization_applied=apply_norm,
            max_l_computed=max_l
        )

    
    def _compute_full(self, grid_points: np.ndarray, ao_matrix: np.ndarray,
                     apply_norm: bool) -> None:
        """
        Compute all AOs at once.
        """
        # Compute AOs
        iterator = enumerate(self.ao_specs)
        if self.show_progress:
            iterator = tqdm(list(iterator), total=len(self.ao_specs), desc="Computing AOs")
        
        for ao_idx, spec in iterator:
            # Get atom center
            center = spec['atom_center']
            
            # Compute relative coordinates
            R = grid_points - center
            x, y, z = R[:, 0], R[:, 1], R[:, 2]
            
            r_sq = x**2 + y**2 + z**2
            r_stable = np.sqrt(r_sq + EPSILON)
            
            # Compute angles
            theta = np.arccos(np.clip(z / r_stable, -1.0, 1.0))
            phi = np.arctan2(y, x)
            
            # Compute this AO
            ao_values = compute_single_ao(
                r_sq, r_stable, theta, phi,
                spec['exponents'], spec['coeffs'],
                spec['scale_factor_sq'], spec['l'], spec['m_idx'],
                apply_norm
            )
            
            ao_matrix[:, ao_idx] = ao_values
    
    def _compute_batched(self, grid_points: np.ndarray, ao_matrix: np.ndarray,
                        apply_norm: bool, batch_size: int) -> None:
        """
        Compute AOs in batches to save memory.
        """
        num_points = grid_points.shape[0]
        num_batches = (num_points + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_points)
            batch_points = grid_points[start:end]
            
            # Create temporary matrix for this batch
            batch_ao = np.empty((end - start, len(self.ao_specs)), dtype=np.float64)
            
            # Compute AOs for this batch
            temp_computer = AtomicOrbitalComputer(
                self.atoms_data, self.gto_data, 
                self.basis_info, show_progress=False
            )
            temp_computer.ao_specs = self.ao_specs
            temp_computer._compute_full(batch_points, batch_ao, apply_norm)
            
            # Copy to main matrix
            ao_matrix[start:end, :] = batch_ao
            
            if self.show_progress:
                print(f"  Batch {batch_idx + 1}/{num_batches} complete")


def compute_atomic_orbitals(grid_points: np.ndarray, atoms_data: List[Any],
                           gto_data: List[Any], basis_info: Optional[Any] = None,
                           **kwargs) -> Tuple[np.ndarray, List[str]]:
    computer = AtomicOrbitalComputer(atoms_data, gto_data, basis_info, **kwargs)
    result = computer.compute(grid_points)
    return result.ao_matrix, result.ao_labels
