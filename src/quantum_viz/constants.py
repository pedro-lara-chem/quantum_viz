"""
Quantum Chemistry Visualization Package - Physical Constants and Mappings

This module contains all physical constants, element data, and mapping dictionaries
used throughout the quantum chemistry visualization package.

Constants:
    - BOHR_TO_ANGSTROM: Conversion factor from Bohr radii to Angstroms
    - ANGSTROM_TO_BOHR: Conversion factor from Angstroms to Bohr radii
    - EPSILON: Small value for numerical stability

Mappings:
    - L_QUANTUM_NUMBERS_MAP: Spectroscopic notation to angular momentum quantum number
    - ANGULAR_LABELS: Angular momentum to orbital component labels
    - SPECTROSCOPIC_NOTATION: Angular momentum to spectroscopic letter

Element Data:
    - COVALENT_RADII: Covalent radii for elements (in Angstroms)
    - ATOMIC_COLORS: CPK coloring scheme for elements
    - ATOMIC_MASSES: Standard atomic masses (in amu)

Author: Pedro Lara
Version: 2.1.0
Date: 2024
"""

from typing import Dict, List, Tuple
import numpy as np

# ==============================================================================
# Physical Constants
# ==============================================================================

BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM  # 1.8897259886
EPSILON: float = 1e-12
PI: float = np.pi

# ==============================================================================
# Visualization scaling
# ==============================================================================

# Atom radii are stored in Angstroms (standard CPK radii)
# For visualization in Bohr, multiply by ANGSTROM_TO_BOHR
ATOM_RADIUS_SCALE: float = 0.4  # Scale factor for better visualization

# ==============================================================================
# Quantum Number Mappings
# ==============================================================================

L_QUANTUM_NUMBERS_MAP: Dict[str, int] = {
    's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5,
    'i': 6, 'k': 7, 'l': 8, 'm': 9, 'n': 10, 'o': 11, 'q': 12
}

SPECTROSCOPIC_NOTATION: List[str] = [
    's', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'q'
]

ANGULAR_LABELS: Dict[int, List[str]] = {
    0: ['s'],
    1: ['px', 'py', 'pz'],
    2: ['dxy', 'dyz', 'dz2', 'dxz', 'dx2y2'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)'],
    4: ['gz4', 'gxz3', 'gyz3', 'gz2(x2-y2)', 'gxyz2', 'gz2xy', 'gzx3', 'gzy3', 'gx4+y4'],
    5: ['hz5', 'hxz4', 'hyz4', 'hz3(x2-y2)', 'hxyz3', 'hz2(x2-y2)', 'hz2xy', 
        'hzx(x2-3y2)', 'hzy(3x2-y2)', 'hz(x4-6x2y2+y4)', 'hx(x4-10x2y2+5y4)']
}

# ==============================================================================
# Element Data
# ==============================================================================

COVALENT_RADII: Dict[str, float] = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
    'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
    'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
    'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39,
    'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
    'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20,
    'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'I': 1.40
}

ATOMIC_COLORS: Dict[str, str] = {
    'H': '#E8E8E8',  # Light gray
    'He': '#D9FFFF',  # Cyan
    'C': '#505050',   # Dark gray
    'N': '#304FFE',   # Bright blue
    'O': '#FF0D0D',   # Bright red
    'F': '#90E050',   # Light green
    'Ne': '#B3E3F5',  # Light cyan
    'Cl': '#1FF01F',  # Bright green
    'Ar': '#80D1E3',  # Pale cyan
    'S': '#FFC832',   # Golden yellow
    'Br': '#A62929',  # Dark red
    'Kr': '#5CB8D1',  # Cyan-blue
    'B': '#FFB5B5',   # Light pink
    'P': '#FF8000',   # Orange
    'Fe': '#E06633',  # Rust
    'Co': '#F090A0',  # Pink
    'Ni': '#50D050',  # Green
    'Cu': '#C88033',  # Bronze
    'Zn': '#7D80B0',  # Lavender
    'Si': '#F0C8A0',  # Tan
    'Li': '#CC80FF',  # Purple
    'Na': '#AB5CF2',  # Violet
    'Mg': '#8AFF00',  # Lime
    'Al': '#BFA6A6',  # Gray-pink
    'K': '#8F40D4',   # Purple
    'Ca': '#3DFF00',  # Bright green
    'Ti': '#BFC2C7',  # Silver
    'V': '#A6A6AB',   # Gray
    'Mn': '#9C7AC4',  # Lavender
    'I': '#940094'    # Purple
}

ATOMIC_MASSES: Dict[str, float] = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.065, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'I': 126.90
}

# Default values for missing elements
DEFAULT_COLOR: str = '#808080'  # Gray
DEFAULT_RADIUS: float = 0.8     # Angstroms
DEFAULT_MASS: float = 10.0      # amu

# ==============================================================================
# Visualization Constants
# ==============================================================================

# Orbital visualization defaults
DEFAULT_ISOVALUE: float = 0.01
DEFAULT_RESOLUTION: int = 61
DEFAULT_OPACITY: float = 0.7
DEFAULT_SPECULAR: float = 0.5
DEFAULT_SPECULAR_POWER: float = 20.0

# Colors for positive/negative orbital phases
POSITIVE_PHASE_COLOR: str = '#FF4444'  # Soft red
NEGATIVE_PHASE_COLOR: str = '#4444FF'  # Soft blue

# Bond visualization
BOND_RADIUS: float = 0.15
BOND_TOLERANCE: float = 0.20  # Angstroms tolerance for bond detection
