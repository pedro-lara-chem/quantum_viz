"""
3D Molecule Visualization Module

This module provides high-quality 3D visualization of molecular structures
using PyVista. It includes atom spheres, bond tubes with color gradients,
and customizable styling options.

Features:
    - CPK coloring scheme for atoms
    - Gradient-colored bonds between connected atoms
    - Multiple bond rendering styles (tubes, lines, cylinders)
    - Automatic bond detection
    - Customizable atom radii and colors
    - Export to various 3D formats (GLTF, PLY, STL, etc.)

Classes:
    MoleculePlotter: Main class for molecule visualization
    VisualizationStyle: Configuration for visual appearance

Functions:
    plot_molecule: Convenience function for quick plotting
    export_molecule_gltf: Export molecule to GLTF format

Author: Pedro Lara
Version: 2.0.0
Date: 2024
"""

import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

from constants import (
    ATOMIC_COLORS, COVALENT_RADII, DEFAULT_COLOR, DEFAULT_RADIUS,
    BOND_TOLERANCE, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, ATOM_RADIUS_SCALE, BOND_RADIUS
)
from utils.geometry import detect_bonds


@dataclass
class VisualizationStyle:
    """Configuration for molecule visualization style."""
    # Atom styles
    atom_radius_scale: float = 0.4  # Slightly smaller for better visibility
    atom_smooth_shading: bool = True
    atom_specular: float = 0.5
    atom_specular_power: float = 15.0
    atom_metallic: float = 0.1
    
    # Bond styles
    bond_radius: float = 0.12
    bond_smooth_shading: bool = True
    bond_use_gradient: bool = True
    
    # Background and lighting
    background_color: str = 'white'
    lighting: str = 'three lights'  # 'light kit', 'three lights', 'none'
    

class MoleculePlotter:
    """
    High-quality 3D molecule visualizer with proper unit handling.
    
    Expects coordinates in BOHR internally, but scales for display.
    """
    
    def __init__(self, coordinates: np.ndarray, symbols: List[str],
                 bonds: Optional[np.ndarray] = None,
                 style: Optional[VisualizationStyle] = None,
                 plotter: Optional[pv.Plotter] = None,
                 coords_in_bohr: bool = True):
        """
        Initialize the molecule plotter.
        
        Args:
            coordinates: Atomic coordinates
            symbols: Element symbols
            bonds: Optional bond matrix
            style: Visualization style
            plotter: Existing PyVista plotter
            coords_in_bohr: If True, coordinates are in Bohr (will be scaled for display)
        """
        self.coords_in_bohr = coords_in_bohr
        
        if coords_in_bohr:
            # Store in Bohr internally
            self.coordinates_bohr = np.array(coordinates, dtype=np.float64)
            # Convert to Angstroms for display (PyVista works better with Å-scale numbers)
            self.coordinates = self.coordinates_bohr * BOHR_TO_ANGSTROM
        else:
            self.coordinates = np.array(coordinates, dtype=np.float64)
            self.coordinates_bohr = self.coordinates * ANGSTROM_TO_BOHR
        
        self.symbols = symbols
        self.n_atoms = len(symbols)
        self.style = style or VisualizationStyle()
        
        # Detect bonds if not provided (using Bohr coordinates)
        if bonds is None:
            self.bonds = detect_bonds(self.coordinates_bohr, symbols, tolerance=BOND_TOLERANCE)
        else:
            self.bonds = bonds
        
        # Create or use plotter
        if plotter is None:
            self.plotter = pv.Plotter()
            self._owns_plotter = True
        else:
            self.plotter = plotter
            self._owns_plotter = False
        
        # Set background and lighting
        self.plotter.set_background(self.style.background_color)
        if hasattr(self.style, 'lighting') and self.style.lighting == 'three lights':
            self.plotter.enable_3_lights()
        elif hasattr(self.style, 'lighting') and self.style.lighting == 'light kit':
            self.plotter.enable_lightkit()
    
    def add_molecule(self):
        import re  # Put this at the top of the file or inside the function

        # 1. ADD ATOMS INDIVIDUALLY
        for i, coord in enumerate(self.coordinates):
            # Clean the symbol (e.g., 'C12' -> 'C', 'n3' -> 'N')
            raw_symbol = self.symbols[i]
            symbol = re.sub(r'[^a-zA-Z]', '', raw_symbol).capitalize()
            
            color = ATOMIC_COLORS.get(symbol, DEFAULT_COLOR)
            # Use a slightly smaller radius so it looks like a nice ball-and-stick model
            radius = COVALENT_RADII.get(symbol, DEFAULT_RADIUS) * 0.6
            
            sphere = pv.Sphere(radius=radius, center=coord)
            self.plotter.add_mesh(sphere, color=color, smooth_shading=True)

        # 2. ADD SPLIT BONDS (Using robust Tubes instead of Cylinders)
        n_atoms = len(self.coordinates)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if self.bonds[i, j] > 0:
                    c1 = self.coordinates[i]
                    c2 = self.coordinates[j]
                    
                    # Find the exact geometric center between the two atoms
                    midpoint = (c1 + c2) / 2.0
                    
                    # Clean symbols for colors
                    sym1 = re.sub(r'[^a-zA-Z]', '', self.symbols[i]).capitalize()
                    sym2 = re.sub(r'[^a-zA-Z]', '', self.symbols[j]).capitalize()
                    
                    color1 = ATOMIC_COLORS.get(sym1, DEFAULT_COLOR)
                    color2 = ATOMIC_COLORS.get(sym2, DEFAULT_COLOR)
                    
                    # Draw first half of the bond
                    line1 = pv.Line(c1, midpoint)
                    tube1 = line1.tube(radius=BOND_RADIUS)
                    self.plotter.add_mesh(tube1, color=color1, smooth_shading=True)
                    
                    # Draw second half of the bond
                    line2 = pv.Line(midpoint, c2)
                    tube2 = line2.tube(radius=BOND_RADIUS)
                    self.plotter.add_mesh(tube2, color=color2, smooth_shading=True)
    
    def show(self) -> None:
        """Display the interactive plotter window."""
        self.plotter.show()
    
    def export(self, filename: str) -> None:
        """
        Export the visualization to a file.
        
        Args:
            filename: Output filename
        """
        if filename.endswith('.gltf') or filename.endswith('.glb'):
            self.plotter.export_gltf(filename)
        elif filename.endswith('.html'):
            self.plotter.export_html(filename)
        elif filename.endswith('.ply'):
            self.plotter.export_ply(filename)
        elif filename.endswith('.stl'):
            self.plotter.export_stl(filename)
        else:
            self.plotter.screenshot(filename, return_img=False)
    
    def close(self) -> None:
        """Close the plotter if we own it."""
        if self._owns_plotter:
            self.plotter.close()
