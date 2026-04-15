# Quantum Chemistry Visualization Tool

A high-performance Python toolkit for parsing quantum chemistry output files (Molden format), computing molecular orbitals, and generating high-quality 3D visualizations using PyVista.

## Features

* **High-Performance Computation:** Uses Numba JIT-compilation for efficient evaluation of Atomic Orbitals (AOs) on 3D grids.
* **Broad Compatibility:** Automatically detects and handles basis set normalization conventions from Gaussian, ORCA, PySCF, and Molpro.
* **Advanced Visuals:** Generates 3D isosurfaces of molecular orbitals (positive/negative phases) and accurate CPK-colored molecular geometries.
* **Math-Verified:** Includes debugging tools to numerically integrate MO densities over grids to verify normalization and orthogonality.
* **Export Options:** Export interactive 3D models to GLTF, HTML, PLY, and STL formats.

## Installation

You can install this tool directly from GitHub using pip:
```bash
pip install git+https://github.com/pedro-lara-chem/quantum_viz.git
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

You can run the tool directly from the terminal to process a Molden file and export 3D visualizations. Because of how it's packaged, you can use the `quantum-viz` command anywhere:
```bash
quantum-viz --input your_molecule.molden --output ./results/ --quality high
```
#### Available Arguments

| Argument | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | `str` | *None* | Path to the input `.molden` file. If omitted, the tool runs in interactive mode. |
| `--output` | `-o` | `str` | `.` | Directory to save the generated 3D files. |
| `--quality` | `-q` | `str` | `medium` | Grid resolution quality preset (`low`, `medium`, `high`, `ultra`). |
| `--resolution` | `-r` | `int` | *None* | Manually override the grid resolution (e.g., 61). |
| `--isovalue` | `-v` | `float`| *None* | Manually override the isosurface value (e.g., 0.01). |
| `--basis-format` | | `str` | `auto` | Force basis format (`auto`, `spherical`, `cartesian`). |
| `--convention` | | `str` | `auto` | Spherical harmonic phase convention (`auto`, `pyscf`, `gaussian`, `orca`). |
| `--debug-phase` | | `flag` | `False` | Print phase diagnostic information (debug only). |
| `--debug-ao` | | `flag` | `False` | Print detailed AO specifications and validate atomic centers. |
| `--verify-math`| | `flag` | `False` | Perform numerical grid integration to verify MO normalization and orthogonality. |

### Python API

You can also use the package as a library in your own Python scripts or Jupyter Notebooks:
```bash
from quantum_viz.parsers.molden_parser import parse_molden_file
from quantum_viz.mathematics.atomic_orbitals import AtomicOrbitalComputer

# 1. Parse the Molden file
atoms, gtos, mos, basis_info = parse_molden_file('molecule.molden')

# 2. Compute Atomic Orbitals on a grid
computer = AtomicOrbitalComputer(atoms, gtos, basis_info)
ao_matrix, ao_labels = computer.compute(grid_points)
```

## Supported Shells
Fully supports **s, p, d, f, g, h,** and **i** orbitals, automatically handling general contraction shells (e.g., 'sp') and Cartesian-to-Spherical harmonic conversions.

## License
This project is licensed under the MIT License.
