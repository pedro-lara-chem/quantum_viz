"""
Molecular Orbital Visualization Module

This module provides high-quality 3D visualization of molecular orbitals
using isosurface rendering with PyVista.

Features:
    - Positive/negative phase isosurface rendering
    - Customizable isovalues and opacities
    - Combined molecule + orbital visualization
    - Batch orbital export
    - Quality presets for different output needs

Classes:
    OrbitalPlotter: Main class for orbital visualization
    OrbitalStyle: Configuration for orbital appearance

Functions:
    plot_orbital: Convenience function for single orbital plotting
    batch_plot_orbitals: Plot multiple orbitals with progress tracking

Author: Pedro Lara
Version: 2.0.0
Date: 2024
"""

"""
Molecular Orbital Visualization Module - Fixed Grid Orientation
"""

import numpy as np
import pyvista as pv
from typing import List, Optional
from dataclasses import dataclass

import sys
import os
from quantum_viz.constants import DEFAULT_ISOVALUE, DEFAULT_OPACITY, BOHR_TO_ANGSTROM
from quantum_viz.visualization.molecule_plotter import MoleculePlotter


@dataclass
class OrbitalStyle:
    """Configuration for orbital visualization style."""
    isovalue: float = DEFAULT_ISOVALUE
    opacity: float = DEFAULT_OPACITY
    positive_color: str = '#FF3333'  # Bright red
    negative_color: str = '#3333FF'  # Bright blue
    smooth_shading: bool = True
    specular: float = 0.4
    specular_power: float = 15.0


class OrbitalPlotter:
    """
    Molecular orbital visualizer with correct grid orientation.
    """
    
    def __init__(self, mo_coeffs: np.ndarray, ao_matrix: np.ndarray,
                 grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                 style: Optional[OrbitalStyle] = None,
                 plotter: Optional[pv.Plotter] = None):
        """
        Initialize the orbital plotter.
        
        Args:
            mo_coeffs: MO coefficients array (n_mos, n_aos)
            ao_matrix: AO values on grid (n_points, n_aos)
            grid_x, grid_y, grid_z: 3D meshgrid arrays with 'ij' indexing
            style: Orbital visualization style
            plotter: Existing PyVista plotter to use
        """
        self.mo_coeffs = mo_coeffs
        self.ao_matrix = ao_matrix
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.style = style or OrbitalStyle()
        
        # Create or use plotter
        if plotter is None:
            self.plotter = pv.Plotter()
            self._owns_plotter = True
        else:
            self.plotter = plotter
            self._owns_plotter = False
    
    def compute_mo_values(self, mo_index: int) -> np.ndarray:
        """
        Compute molecular orbital values on the grid.
        
        Args:
            mo_index: Index of the MO to compute
            
        Returns:
            Array of MO values for all grid points
        """
        if mo_index >= len(self.mo_coeffs):
            raise ValueError(f"MO index {mo_index} out of range")
        
        if len(self.mo_coeffs[mo_index]) != self.ao_matrix.shape[1]:
            raise ValueError(
                f"Dimension mismatch! MO {mo_index+1} has {len(self.mo_coeffs[mo_index])} coefficients, "
                f"but {self.ao_matrix.shape[1]} AOs were computed on the grid. "
                "Ensure Cartesian/Spherical conventions are properly aligned."
            )
            
        # Compute MO values: Ψ = Σ c_i * φ_i
        mo_values = np.dot(self.ao_matrix, self.mo_coeffs[mo_index]).real
        
        return mo_values
    
    def add_orbital(self, mo_index: int, iso_value: Optional[float] = None) -> None:
        """
        Add molecular orbital isosurfaces to the plotter.
        
        Args:
            mo_index: Index of the MO to plot
            iso_value: Override default isovalue
        """
        iso_val = iso_value or self.style.isovalue
        
        # Compute MO values
        mo_values = self.compute_mo_values(mo_index)
        
        # Reshape to 3D grid (using Fortran order for consistency with 'ij' indexing)
        shape = (self.grid_x.shape[0], self.grid_y.shape[1], self.grid_z.shape[2])
        mo_values_3d = mo_values.reshape(shape, order='F')
        
        # Create structured grid
        # PyVista expects dimensions in (x, y, z) order
        grid = pv.ImageData()
        grid.dimensions = self.grid_x.shape
        
        # Calculate origin and spacing for the uniform grid
        grid.origin = (
            self.grid_x.min() * BOHR_TO_ANGSTROM, 
            self.grid_y.min() * BOHR_TO_ANGSTROM, 
            self.grid_z.min() * BOHR_TO_ANGSTROM
            )
        
        spacing_x = ((self.grid_x.max() - self.grid_x.min()) * BOHR_TO_ANGSTROM) / (grid.dimensions[0] - 1)
        spacing_y = ((self.grid_y.max() - self.grid_y.min()) * BOHR_TO_ANGSTROM) / (grid.dimensions[1] - 1)
        spacing_z = ((self.grid_z.max() - self.grid_z.min()) * BOHR_TO_ANGSTROM) / (grid.dimensions[2] - 1)
        grid.spacing = (spacing_x, spacing_y, spacing_z)
        
        # Flatten the values to match PyVista's expected internal memory order
        # (Make sure the flatten order aligns with your meshgrid 'ij' indexing)
        grid.point_data["values"] = mo_values.flatten(order='F')
        
        grid.point_data['mo_values'] = mo_values_3d.ravel(order='F')
        
        # Create positive isosurface
        try:
            pos_surface = grid.contour(
                [iso_val], 
                scalars='mo_values',
                method='marching_cubes'
            )
            
            if pos_surface.n_points > 0:
                self.plotter.add_mesh(
                    pos_surface,
                    color=self.style.positive_color,
                    opacity=self.style.opacity,
                    smooth_shading=self.style.smooth_shading,
                    specular=self.style.specular,
                    specular_power=self.style.specular_power,
                    name=f"MO_{mo_index}_positive"
                )
        except Exception as e:
            print(f"    Warning: Could not create positive isosurface for MO{mo_index+1}")
        
        # Create negative isosurface
        try:
            neg_surface = grid.contour(
                [-iso_val], 
                scalars='mo_values',
                method='marching_cubes'
            )
            
            if neg_surface.n_points > 0:
                self.plotter.add_mesh(
                    neg_surface,
                    color=self.style.negative_color,
                    opacity=self.style.opacity,
                    smooth_shading=self.style.smooth_shading,
                    specular=self.style.specular,
                    specular_power=self.style.specular_power,
                    name=f"MO_{mo_index}_negative"
                )
        except Exception as e:
            print(f"    Warning: Could not create negative isosurface for MO{mo_index+1}")
    
    def add_molecule(self, coordinates: np.ndarray, symbols: List[str],
                    bonds: Optional[np.ndarray] = None) -> None:
        """
        Add molecule structure to the visualization.
        
        Args:
            coordinates: Atomic coordinates
            symbols: Element symbols
            bonds: Optional bond matrix
        """
        mol_plotter = MoleculePlotter(
            coordinates, symbols, bonds,
            plotter=self.plotter
        )
        mol_plotter.add_molecule()
    
    def export(self, filename: str, coords_bohr: Optional[np.ndarray] = None, 
               atomic_numbers: Optional[List[int]] = None, mo_index: Optional[int] = None) -> None:
        """
        Export the visualization to a file.
        
        Args:
            filename: Output filename
        """
        self.plotter.set_background('white')
        
        if filename.endswith('.gltf') or filename.endswith('.glb'):
            self.plotter.export_gltf(filename)
        elif filename.endswith('.html'):
            self.plotter.export_html(filename)
        elif filename.endswith('.obj'):
            self.plotter.export_obj(filename)
        elif filename.endswith('.png'):
            self.plotter.screenshot(filename, transparent_background=False, return_img=False)
        elif filename.endswith('.cube'):
            if coords_bohr is not None and atomic_numbers is not None and mo_index is not None:
                self._export_cube(filename, mo_index, coords_bohr, atomic_numbers)
            else:
                print("    Warning: Missing atomic data to write .cube file")
        else:
            self.plotter.screenshot(filename, return_img=False)

    def _export_cube(self, filename: str, mo_index: int, coordinates_bohr: np.ndarray, atomic_numbers: List[int]) -> None:
        """Custom writer for Gaussian .cube volumetric files."""
        mo_values = self.compute_mo_values(mo_index)
        nx, ny, nz = self.grid_x.shape
        
        origin = (self.grid_x.min(), self.grid_y.min(), self.grid_z.min())
        
        # Calculate step sizes in Bohr
        dx = (self.grid_x.max() - origin[0]) / (nx - 1) if nx > 1 else 0.0
        dy = (self.grid_y.max() - origin[1]) / (ny - 1) if ny > 1 else 0.0
        dz = (self.grid_z.max() - origin[2]) / (nz - 1) if nz > 1 else 0.0
        
        with open(filename, 'w') as f:
            f.write(f"Quantum Viz Generated Cube\n")
            f.write(f"Molecular Orbital {mo_index+1}\n")
            f.write(f"{len(atomic_numbers):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
            f.write(f"{nx:5d} {dx:12.6f} 0.000000 0.000000\n")
            f.write(f"{ny:5d} 0.000000 {dy:12.6f} 0.000000\n")
            f.write(f"{nz:5d} 0.000000 0.000000 {dz:12.6f}\n")
            
            for i, coord in enumerate(coordinates_bohr):
                Z = atomic_numbers[i]
                f.write(f"{Z:5d} {float(Z):12.6f} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
                
            # Flatten to 3D matrix (Fortran order to match meshgrid, then read in C-order for Cube specs)
            mo_3d = mo_values.reshape((nx, ny, nz), order='F')
            
            count = 0
            for val in np.nditer(mo_3d, order='C'):
                f.write(f"{float(val):13.5E}")
                count += 1
                if count % 6 == 0:
                    f.write("\n")
            if count % 6 != 0:
                f.write("\n")
    
    def close(self) -> None:
        """Close the plotter if we own it."""
        if self._owns_plotter:
            self.plotter.close()
