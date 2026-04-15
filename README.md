# Quantum Chemistry Visualization Tool

A high-performance Python toolkit for parsing quantum chemistry output files (Molden format), computing molecular orbitals, and generating high-quality 3D visualizations using PyVista.

## Features

* **High-Performance Computation:** Uses Numba JIT-compilation for efficient evaluation of Atomic Orbitals (AOs) on 3D grids.
* **Broad Compatibility:** Automatically detects and handles basis set normalization conventions from Gaussian, ORCA, PySCF, and Molpro.
* **Advanced Visuals:** Generates 3D isosurfaces of molecular orbitals (positive/negative phases) and accurate CPK-colored molecular geometries.
* **Math-Verified:** Includes debugging tools to numerically integrate MO densities over grids to verify normalization and orthogonality.
* **Smart File Management:** The tool automatically creates your specified output directories if they do not exist. It also intelligently names exported files based on the original filename, orbital index, energy level, and occupancy.
* **Versatile Export Formats:** Export your visualizations to multiple formats depending on your needs: 
  * `gltf` / `glb` (3D web/engine standard)
  * `html` (Interactive 3D webpage)
  * `ply` / `stl` (3D printing and CAD)
  * `png` (Static images for publications)

## Installation

You can install this tool directly from GitHub using pip:
```bash
pip install git+https://github.com/YourUsername/quantum_viz.git
```

Alternatively, if you want to clone the repository for development:

1. Clone the repository:
```bash
   git clone https://github.com/YourUsername/quantum_viz.git
   cd quantum_viz
```

2. Create a virtual environment (recommended):
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
   pip install -e .
```

## Usage

### Command Line Interface

You can run the tool directly from the terminal. Because of how it is packaged, you can use the `quantum-viz` command anywhere on your computer.

#### Interactive Mode (Automatic Scanning)
If you run the command without specifying an input file, the tool will automatically scan your current directory for any `.molden` files and present an interactive menu. You can select a specific file by its number, or type `all` to process every Molden file in the folder sequentially.
```bash
quantum-viz

#### Single File Mode
To bypass the menu and process a specific file directly, use the `--input` argument. The code will automatically create your specified `--output` directory if it does not already exist.
```bash
quantum-viz --input your_molecule.molden --output ./results/ --quality high --format html

#### Available Arguments

| Argument | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | `str` | *None* | Path to the input `.molden` file. If omitted, runs in interactive mode. |
| `--output` | `-o` | `str` | `.` | Directory to save the generated 3D files. Auto-created if missing. |
| `--quality` | `-q` | `str` | `medium` | Grid resolution quality preset (`low`, `medium`, `high`, `ultra`). |
| `--resolution` | `-r` | `int` | *None* | Manually override the grid resolution (e.g., 61). |
| `--isovalue` | `-v` | `float`| *None* | Manually override the isosurface value (e.g., 0.01). |
| `--basis-format` | | `str` | `auto` | Force basis format (`auto`, `spherical`, `cartesian`). |
| `--convention` | | `str` | `auto` | Spherical harmonic phase convention (`auto`, `pyscf`, `gaussian`, `orca`). |
| `--format` | `-f` | `str` | `gltf` | Output file format (`gltf`, `html`, `ply`, `stl`, `png`). |
| `--debug-phase` | | `flag` | `False` | Print phase diagnostic information (debug only). |
| `--debug-ao` | | `flag` | `False` | Print detailed AO specifications and validate atomic centers. |
| `--verify-math`| | `flag` | `False` | Perform numerical grid integration to verify MO normalization. |

### Python API

You can also use the package as a library in your own Python scripts or Jupyter Notebooks:

from quantum_viz.parsers.molden_parser import parse_molden_file
from quantum_viz.mathematics.atomic_orbitals import AtomicOrbitalComputer

# 1. Parse the Molden file
atoms, gtos, mos, basis_info = parse_molden_file('molecule.molden')

# 2. Compute Atomic Orbitals on a grid
computer = AtomicOrbitalComputer(atoms, gtos, basis_info)
ao_matrix, ao_labels = computer.compute(grid_points)


## Core Mathematics & Exotic Orbitals

This package goes far beyond hardcoded orbital formulas. The core of the spherical harmonic generation relies on the `compute_associated_legendre` and `compute_associated_legendre_numba` functions. 

Instead of relying on rigid, pre-defined algebraic equations for specific shells, these functions implement a highly stable mathematical recurrence relation to dynamically build associated Legendre polynomials. 

Because of this generalized programming approach, the tool supports standard **s, p, d, f, g, h,** and **i** orbitals, but is also theoretically capable of rendering extremely exotic high-angular momentum orbitals mapped up to **q** ($l=12$) and beyond without any code modification. The only limitation to this recurrence algorithm is standard 64-bit floating-point overflow at absurdly high quantum numbers.

## License
This project is licensed under the MIT License.
