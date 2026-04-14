# quantum_viz/__init__.py
"""
Quantum Chemistry Visualization Package

A comprehensive toolkit for parsing quantum chemistry output files,
computing molecular orbitals, and generating high-quality 3D visualizations.

Modules:
    - parsers: Molden file parsing and basis set detection
    - mathematics: AO computation, spherical harmonics, normalization
    - visualization: 3D molecule and orbital plotting
    - utils: Geometry utilities and file handling

Example:
    >>> from quantum_viz import MoldenParser, AtomicOrbitalComputer
    >>> parser = MoldenParser('molecule.molden')
    >>> atoms, gtos, mos, info = parser.parse()
    >>> computer = AtomicOrbitalComputer(atoms, gtos, info)
    >>> ao_matrix, ao_labels = computer.compute(grid_points)
"""

__version__ = '2.0.0'
__author__ = 'Quantum Chemistry Visualization Team'

from parsers.molden_parser import MoldenParser, parse_molden_file
from mathematics.atomic_orbitals import AtomicOrbitalComputer, compute_atomic_orbitals
from visualization.orbital_plotter import OrbitalPlotter, plot_orbital
