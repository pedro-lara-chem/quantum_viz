"""
Molecular Geometry Utilities Module

This module provides functions for molecular geometry analysis,
distance calculations, bond detection, and symmetry operations.

Features:
    - Fast distance matrix computation using scipy
    - Bond detection based on covalent radii
    - Principal axes and symmetry detection
    - Grid generation with dynamic buffer for diffuse functions
    - Molecule centering and alignment

Functions:
    - compute_distance_matrix: Fast pairwise distance calculation
    - detect_bonds: Identify chemical bonds
    - compute_principal_axes: Find molecular principal axes
    - generate_grid: Create 3D grid for visualization
    - center_molecule: Center molecule at origin

Author: Pedro Lara
Version: 2.0.0
Date: 2024
"""

"""
Molecular Geometry Utilities Module with Unified Unit Handling

All internal calculations use Bohr (AU) for consistency with quantum chemistry.
Visualization can use either Bohr or Angstroms.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Optional, Dict

import re
from constants import (
    COVALENT_RADII, DEFAULT_RADIUS, BOND_TOLERANCE,
    BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR
)


# ==============================================================================
# Unit Conversion Utilities
# ==============================================================================

class UnitConverter:
    """Utility class for unit conversions."""
    
    @staticmethod
    def to_bohr(coordinates: np.ndarray, from_unit: str) -> np.ndarray:
        """
        Convert coordinates to Bohr (AU).
        
        Args:
            coordinates: Input coordinates
            from_unit: Source unit ("bohr", "au", "angstrom", "angs")
            
        Returns:
            Coordinates in Bohr
        """
        if from_unit.lower() in ["bohr", "au"]:
            return coordinates.copy()
        elif from_unit.lower() in ["angstrom", "angs", "a"]:
            return coordinates * ANGSTROM_TO_BOHR
        else:
            raise ValueError(f"Unknown unit: {from_unit}")
    
    @staticmethod
    def to_angstrom(coordinates: np.ndarray, from_unit: str) -> np.ndarray:
        """
        Convert coordinates to Angstroms.
        
        Args:
            coordinates: Input coordinates
            from_unit: Source unit ("bohr", "au", "angstrom", "angs")
            
        Returns:
            Coordinates in Angstroms
        """
        if from_unit.lower() in ["angstrom", "angs", "a"]:
            return coordinates.copy()
        elif from_unit.lower() in ["bohr", "au"]:
            return coordinates * BOHR_TO_ANGSTROM
        else:
            raise ValueError(f"Unknown unit: {from_unit}")
    
    @staticmethod
    def detect_unit_from_molden_header(header_line: str) -> str:
        """
        Detect coordinate unit from Molden [Atoms] header.
        
        Args:
            header_line: The [Atoms] header line
            
        Returns:
            "AU" or "Angs"
        """
        if "angs" in header_line.lower():
            return "Angs"
        return "AU"


# ==============================================================================
# Distance and Bond Detection (Always uses Bohr internally)
# ==============================================================================

def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix using scipy for speed.
    
    Args:
        coordinates: Array of shape (n_atoms, 3) in BOHR
        
    Returns:
        Distance matrix of shape (n_atoms, n_atoms) in BOHR
    """
    return squareform(pdist(coordinates))


def detect_bonds(coordinates: np.ndarray, symbols: List[str],
                tolerance: float = BOND_TOLERANCE) -> np.ndarray:
    """
    Detect chemical bonds based on atomic distances and covalent radii.
    
    IMPORTANT: This function expects coordinates in BOHR and returns
    distances in BOHR. Covalent radii are converted from Angstroms to Bohr.
    
    Args:
        coordinates: Array of shape (n_atoms, 3) in BOHR
        symbols: List of element symbols
        tolerance: Additional tolerance for bond detection (in Angstroms)
        
    Returns:
        Bond matrix where non-zero entries indicate bonds (distances in BOHR)
    """
    n_atoms = len(symbols)
    dist_mat = compute_distance_matrix(coordinates)
    bond_mat = np.zeros((n_atoms, n_atoms))
    
    # Convert tolerance to Bohr
    tolerance_bohr = tolerance * ANGSTROM_TO_BOHR
    
    for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Clean symbols (e.g., 'H15' -> 'H')
                sym_i = re.sub(r'[^a-zA-Z]', '', symbols[i]).capitalize()
                sym_j = re.sub(r'[^a-zA-Z]', '', symbols[j]).capitalize()
                
                # # Organic chemistry heuristic: Prevent H-H bonds 
                # if sym_i == 'H' and sym_j == 'H':
                #     continue
                    
                # Get covalent radii in Angstroms, convert to Bohr
                r_i_ang = COVALENT_RADII.get(sym_i, DEFAULT_RADIUS)
                r_j_ang = COVALENT_RADII.get(sym_j, DEFAULT_RADIUS)
                
                r_i_bohr = r_i_ang * ANGSTROM_TO_BOHR
                r_j_bohr = r_j_ang * ANGSTROM_TO_BOHR
                
                # Check if distance is within sum of covalent radii + tolerance
                if dist_mat[i, j] <= (r_i_bohr + r_j_bohr + tolerance_bohr):
                    bond_mat[i, j] = dist_mat[i, j]
                    bond_mat[j, i] = dist_mat[i, j]
    
    return bond_mat


# ==============================================================================
# Grid Generation (Always uses Bohr internally)
# ==============================================================================

def generate_grid(coordinates: np.ndarray, resolution: int = 61,
                 dynamic_buffer: bool = True,
                 buffer_ratio: float = 0.4,
                 min_buffer: float = 15.0) -> Tuple[np.ndarray, ...]:
    """
    Generate a 3D grid for molecular orbital visualization.
    
    IMPORTANT: Expects coordinates in BOHR, returns grid in BOHR.
    
    Args:
        coordinates: Array of shape (n_atoms, 3) in BOHR
        resolution: Number of grid points along each dimension
        dynamic_buffer: If True, calculate buffer based on molecule extent
        buffer_ratio: Ratio of max_extent to use as buffer (e.g., 0.4 = 40%)
        min_buffer: Minimum buffer to apply (in BOHR)
        
    Returns:
        Tuple of (grid_x, grid_y, grid_z, points) all in BOHR
    """

    # 1. Find the physical bounds and the true center of the molecule
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    center = (max_coords + min_coords) / 2.0
    
    # 2. Find the largest extent to make a uniform cube
    extents = max_coords - min_coords
    max_extent = np.max(extents)
    
    # 3. Calculate the buffer space
    calculated_buffer = max_extent * buffer_ratio
    effective_buffer = max(min_buffer, calculated_buffer)
    grid_range = max_extent / 2 + effective_buffer
    
    # --- 4. THE FIX: Shift the linspace grid by the molecule's true center ---
    x = np.linspace(center[0] - grid_range, center[0] + grid_range, resolution)
    y = np.linspace(center[1] - grid_range, center[1] + grid_range, resolution)
    z = np.linspace(center[2] - grid_range, center[2] + grid_range, resolution)
    # -------------------------------------------------------------------------
    
    # Create 3D meshgrid
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    
    # Stack into points array - using Fortran order for PyVista consistency
    points = np.vstack((
        grid_x.ravel(order='F'),
        grid_y.ravel(order='F'),
        grid_z.ravel(order='F')
    )).T
    
    return grid_x, grid_y, grid_z, points
