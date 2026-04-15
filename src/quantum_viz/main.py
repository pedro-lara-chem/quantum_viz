"""
Quantum Chemistry Molecular Orbital Visualization Tool – Revised

Usage:
    python main.py [--input MOLDEN_FILE] [--output OUTPUT_DIR] [--quality QUALITY]
                  [--basis-format {auto,spherical,cartesian}] [--convention {auto,pyscf,gaussian,orca}]
                  [--debug-phase] [--debug-ao]

Author: Pedro Lara
Version: 2.1.0
Date: 2026
"""

import sys
import os
import argparse
import numpy as np
import pyvista as pv
from typing import List, Optional, Tuple
from tqdm import tqdm

from parsers.molden_parser import MoldenParser, BasisSetInfo
from mathematics.atomic_orbitals import AtomicOrbitalComputer
from constants import BOHR_TO_ANGSTROM, L_QUANTUM_NUMBERS_MAP
from mathematics.spherical_harmonics import (
    cartesian_to_spherical_coeffs,
    reorder_spherical_coeffs,
    OrderingConvention,
    debug_check_mo_phase
)
from utils.geometry import generate_grid, detect_bonds
from visualization.molecule_plotter import MoleculePlotter, VisualizationStyle
from visualization.orbital_plotter import OrbitalPlotter, OrbitalStyle


class OrbitalVisualizationApp:
    """Main application class with improved basis handling and debugging."""

    def __init__(self, quality: str = 'medium', resolution: int = 61,
                 iso_value: float = 0.01, basis_format: str = 'auto',
                 convention: str = 'auto', debug_phase: bool = False,
                 debug_ao: bool = False, verify_math: bool = False):
        self.quality = quality
        self.resolution = resolution
        self.iso_value = iso_value
        self.basis_format = basis_format
        self.convention = convention
        self.debug_phase = debug_phase
        self.debug_ao = debug_ao
        self.verify_math = verify_math
        self._apply_quality_preset()

    def _apply_quality_preset(self) -> None:
        presets = {
            'low': {'resolution': 41, 'iso_value': 0.02},
            'medium': {'resolution': 61, 'iso_value': 0.01},
            'high': {'resolution': 81, 'iso_value': 0.007},
            'ultra': {'resolution': 101, 'iso_value': 0.005}
        }
        if self.quality in presets:
            preset = presets[self.quality]
            self.resolution = preset['resolution']
            self.iso_value = preset['iso_value']

    def find_molden_files(self, directory: str = '.') -> List[str]:
        molden_files = []
        for f in os.listdir(directory):
            if 'molden' in f.lower() and not f.endswith(('.zip', '.tar', '.gz')):
                full = os.path.join(directory, f)
                if os.path.isfile(full):
                    molden_files.append(full)
        return sorted(molden_files)

    def process_molecule(self, molden_file: str) -> Tuple:
        print(f"\n{'='*60}\nProcessing: {os.path.basename(molden_file)}\n{'='*60}")
        parser = MoldenParser(molden_file)
        atoms, gtos, mos, basis_info = parser.parse()
        print(f"  Found {len(atoms)} atoms, {len(mos)} MOs")
        print(f"  Coordinate unit: {parser.coordinate_unit}")
        print(f"  Basis convention: {basis_info.convention.value}")
        print(f"  Program variant: {basis_info.variant.value}")
        coords_bohr = parser.get_coordinates(unit="bohr")
        symbols = parser.get_symbols()
        max_ext = max(np.ptp(coords_bohr[:, i]) for i in range(3))
        print(f"  Molecule extent: {max_ext:.2f} Bohr ({max_ext*BOHR_TO_ANGSTROM:.2f} Å)")
        return atoms, gtos, mos, coords_bohr, symbols, basis_info

    def _count_aos(self, gtos: List) -> int:
        count = 0
        for atom_gto in gtos:
            for shell in atom_gto.shells:
                for char in shell.type.lower():
                    if char in L_QUANTUM_NUMBERS_MAP:
                        l_val = L_QUANTUM_NUMBERS_MAP[char]
                        count += 2 * l_val + 1
        return count

    def compute_aos(self, atoms, gtos, basis_info, coordinates):
        print(f"\n  Generating grid (resolution: {self.resolution})...")
        grid_x, grid_y, grid_z, points = generate_grid(
            coordinates, resolution=self.resolution,
            dynamic_buffer=True, buffer_ratio=0.4, min_buffer=15.0
        )
        print(f"  Grid size: {points.shape[0]:,} points")
        print(f"  Computing atomic orbitals...")
        computer = AtomicOrbitalComputer(
            atoms, gtos, basis_info, show_progress=True, debug=self.debug_ao
        )
        result = computer.compute(points)
        print(f"  Computed {result.num_aos} AOs in {result.computation_time:.2f}s")

        if self.debug_ao:
            # Validate AO centers
            computer.validate_atom_centers(points, sample_indices=list(range(min(20, result.num_aos))))

        return result.ao_matrix, result.ao_labels, grid_x, grid_y, grid_z, points

    def get_mo_range(self, mos: List) -> Tuple[int, int]:
        occupations = np.array([mo.occupancy for mo in mos])
        homo_idx = np.where(occupations >= 1.0)[0]
        if len(homo_idx) > 0:
            homo = homo_idx[-1] + 1
            lumo = homo + 1
            print(f"\n  HOMO: orbital {homo} (energy: {mos[homo-1].energy:.4f} Hartree)")
            if lumo <= len(mos):
                print(f"  LUMO: orbital {lumo} (energy: {mos[lumo-1].energy:.4f} Hartree)")
            print(f"  Available orbitals: 1 to {len(mos)}")
        else:
            homo = 1
            print(f"\n  Available orbitals: 1 to {len(mos)}")
        print(f"\n  Enter MO range to plot:")
        print(f"    - Two integers for range (e.g., '{homo-2} {homo+2}')")
        print(f"    - One integer for single orbital (e.g., '{homo}')")
        print(f"    - 0 for molecule only")
        print(f"    - Press Enter for HOMO only")
        user = input("  > ").strip()
        if not user:
            first = last = homo
        else:
            parts = user.split()
            if len(parts) == 2:
                first, last = int(parts[0]), int(parts[1])
            elif len(parts) == 1:
                first = last = int(parts[0])
            else:
                first = last = homo
        if first == 0:
            return 0, 0
        first = max(1, min(first, len(mos)))
        last = max(first, min(last, len(mos)))
        return first, last

    def export_visualization(self, molden_file: str, output_dir: str = '.') -> None:
        atoms, gtos, mos, coords_bohr, symbols, basis_info = self.process_molecule(molden_file)
        ao_matrix, ao_labels, grid_x, grid_y, grid_z, points = self.compute_aos(
            atoms, gtos, basis_info, coords_bohr
        )
        first_mo, last_mo = self.get_mo_range(mos)

        # Determine basis format (spherical vs Cartesian) from user override or detection
        if self.basis_format == 'auto':
            is_cartesian_input = (basis_info.uses_6d or basis_info.uses_10f)
        else:
            is_cartesian_input = (self.basis_format == 'cartesian')

        # Build shell angular momenta list
        shell_l_seq = []
        for atom_gto in gtos:
            for shell in atom_gto.shells:
                for char in shell.type.lower():
                    if char in L_QUANTUM_NUMBERS_MAP:
                        shell_l_seq.append(L_QUANTUM_NUMBERS_MAP[char])
        spherical_total = sum(2 * l + 1 for l in shell_l_seq)
        cartesian_total = sum((l + 1) * (l + 2) // 2 for l in shell_l_seq)

        # Determine ordering convention for conversion
        if self.convention == 'auto':
            variant = basis_info.variant.value
            if variant in ['gaussian', 'molpro']:
                source_conv = 'gaussian'
            elif variant == 'orca':
                source_conv = 'orca'
            else:
                source_conv = 'pyscf'
        else:
            source_conv = self.convention

        # Align MO coefficients
        aligned_mo_coeffs = []
        for mo in mos:
            coeffs = mo.coefficients
            if len(coeffs) == cartesian_total and ao_matrix.shape[1] == spherical_total:
                if is_cartesian_input:
                    print(f"  Converting Cartesian MO coefficients to spherical (source: {source_conv})...")
                    new_coeffs = []
                    idx = 0
                    for l in shell_l_seq:
                        n_cart = (l + 1) * (l + 2) // 2
                        cart_slice = coeffs[idx:idx + n_cart]
                        sph_slice = cartesian_to_spherical_coeffs(
                            list(cart_slice), l,
                            source_convention=source_conv,
                            apply_sign_correction=True
                        )
                        new_coeffs.extend(sph_slice)
                        idx += n_cart
                    aligned_mo_coeffs.append(new_coeffs)
                else:
                    raise ValueError("MO coefficients appear Cartesian but --basis-format=spherical forced.")
            elif len(coeffs) == spherical_total:
                # Already spherical; optionally reorder if convention differs from PySCF (our AO order)
                if source_conv != 'pyscf':
                    reordered = []
                    idx = 0
                    for l in shell_l_seq:
                        n_sph = 2 * l + 1
                        sph_slice = coeffs[idx:idx + n_sph]
                        reordered_slice = reorder_spherical_coeffs(
                            list(sph_slice), l,
                            from_conv=source_conv, to_conv='pyscf'
                        )
                        reordered.extend(reordered_slice)
                        idx += n_sph
                    aligned_mo_coeffs.append(reordered)
                else:
                    aligned_mo_coeffs.append(coeffs)
            else:
                raise ValueError(
                    f"MO coefficient length mismatch: got {len(coeffs)}, "
                    f"expected {spherical_total} (spherical) or {cartesian_total} (Cartesian)."
                )
        mo_coeffs = np.array(aligned_mo_coeffs)
        mo_energies = np.array([mo.energy for mo in mos])
        mo_occupancies = np.array([mo.occupancy for mo in mos])

        # Debug phase check if requested
        if self.debug_phase:
            debug_check_mo_phase(mo_coeffs, ao_matrix, points, coords_bohr, symbols)

        bonds = detect_bonds(coords_bohr, symbols)
        base_name = os.path.splitext(os.path.basename(molden_file))[0]
        os.makedirs(output_dir, exist_ok=True)

        if first_mo == 0:
            # Molecule only
            output_file = os.path.join(output_dir, f"{base_name}_molecule.gltf")
            plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
            plotter.set_background('white')
            plotter.enable_3_lights()
            mol_plotter = MoleculePlotter(coords_bohr, symbols, bonds, plotter=plotter)
            mol_plotter.add_molecule()
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.5)
            plotter.export_gltf(output_file)
            plotter.close()
            print(f"  ✓ Saved to {output_file}")
        else:
            mo_indices = list(range(first_mo - 1, last_mo))

            # Perform mathematical verification before rendering
            if self.verify_math:
                            self.verify_mo_mathematics(
                                ao_matrix, mo_coeffs, grid_x, grid_y, grid_z, mo_indices
                            )
            print(f"\n  Exporting {len(mo_indices)} orbital(s)...")
            for mo_idx in tqdm(mo_indices, desc="  Rendering orbitals"):
                plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
                plotter.set_background('white')
                plotter.enable_3_lights()
                orbital_style = OrbitalStyle(isovalue=self.iso_value)
                orb_plotter = OrbitalPlotter(
                    mo_coeffs, ao_matrix, grid_x, grid_y, grid_z,
                    style=orbital_style, plotter=plotter
                )
                orb_plotter.add_orbital(mo_idx)
                orb_plotter.add_molecule(coords_bohr, symbols, bonds)
                plotter.camera_position = 'iso'
                plotter.camera.zoom(1.5)
                energy = mo_energies[mo_idx]
                occup = mo_occupancies[mo_idx]
                homo_idx = np.where(mo_occupancies >= 1.0)[0]
                special = ""
                if len(homo_idx) > 0:
                    homo = homo_idx[-1]
                    if mo_idx == homo:
                        special = "_HOMO"
                    elif mo_idx == homo + 1:
                        special = "_LUMO"
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_MO{mo_idx+1}{special}_E{energy:.4f}_Occ{occup:.2f}.gltf"
                )
                orb_plotter.export(output_file)
                orb_plotter.close()
            print(f"\n  ✓ Saved {len(mo_indices)} orbital(s) to {output_dir}/")

    def verify_mo_mathematics(self, ao_matrix: np.ndarray, mo_coeffs: np.ndarray,
                              grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                              mo_indices: List[int]) -> None:
        """
        Numerically integrates MO densities over the 3D grid to verify
        normalization and orthogonality without visual inspection.
        """
        print("\n" + "="*60)
        print("MATHEMATICAL VERIFICATION: Normalization & Orthogonality")
        print("="*60)

        # 1. Calculate the volume element dV (in Bohr^3)
        # Using the dimensions of the generated meshgrid
        dx = (grid_x.max() - grid_x.min()) / (grid_x.shape[0] - 1)
        dy = (grid_y.max() - grid_y.min()) / (grid_y.shape[1] - 1)
        dz = (grid_z.max() - grid_z.min()) / (grid_z.shape[2] - 1)
        dv = dx * dy * dz

        print(f"  Grid spacing (Bohr): dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
        print(f"  Volume element dV:   {dv:.6e} Bohr^3\n")

        # Set a tolerance for the integral. If the grid doesn't capture the far
        # tails of the wavefunction, the norm might be slightly less than 1 (e.g., 0.99)
        tolerance = 0.05 

        for idx, mo_idx in enumerate(mo_indices):
            # Evaluate the MO on the grid: Ψ_i = Σ c_ui * φ_u
            mo_i_values = np.dot(ao_matrix, mo_coeffs[mo_idx]).real
            
            # Calculate Density |Ψ_i|^2
            density_i = mo_i_values ** 2
            
            # Integrate over all space: ∫|Ψ_i|^2 dV
            norm_i = np.sum(density_i) * dv

            status = "✓ (Pass)" if abs(norm_i - 1.0) < tolerance else "✗ (Fail)"
            print(f"  MO {mo_idx+1:3d} | Norm: {norm_i:.6f} {status}")

            # Check orthogonality with the next requested MO in the list
            if idx + 1 < len(mo_indices):
                next_mo_idx = mo_indices[idx + 1]
                mo_j_values = np.dot(ao_matrix, mo_coeffs[next_mo_idx]).real
                
                # Integrate the overlap: ∫ Ψ_i * Ψ_j dV
                overlap = np.sum(mo_i_values * mo_j_values) * dv
                print(f"         | Overlap with MO {next_mo_idx+1}: {overlap:.6e}")

    def run_interactive(self) -> None:
        print("\n" + "="*60)
        print("QUANTUM CHEMISTRY ORBITAL VISUALIZATION TOOL")
        print("="*60)
        molden_files = self.find_molden_files()
        if not molden_files:
            print("\nNo Molden files found.")
            return
        print(f"\nFound {len(molden_files)} Molden file(s):")
        for i, f in enumerate(molden_files):
            print(f"  {i+1}. {os.path.basename(f)}")
        if len(molden_files) > 1:
            choice = input(f"\nSelect file (1-{len(molden_files)}) or 'all': ").strip()
            if choice.lower() == 'all':
                for f in molden_files:
                    try:
                        self.export_visualization(f)
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                try:
                    idx = int(choice) - 1
                    self.export_visualization(molden_files[idx])
                except (ValueError, IndexError):
                    print("Invalid, using first file.")
                    self.export_visualization(molden_files[0])
        else:
            self.export_visualization(molden_files[0])
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quantum Chemistry MO Visualization Tool")
    parser.add_argument('--input', '-i', help='Input Molden file')
    parser.add_argument('--output', '-o', default='.', help='Output directory')
    parser.add_argument('--quality', '-q', default='medium',
                        choices=['low', 'medium', 'high', 'ultra'])
    parser.add_argument('--resolution', '-r', type=int, default=None)
    parser.add_argument('--isovalue', '-v', type=float, default=None)
    parser.add_argument('--basis-format', choices=['auto', 'spherical', 'cartesian'],
                        default='auto', help='Force basis format')
    parser.add_argument('--convention', choices=['auto', 'pyscf', 'gaussian', 'orca'],
                        default='auto', help='Spherical harmonic phase convention')
    parser.add_argument('--debug-phase', action='store_true',
                        help='Print phase diagnostic information (debug only)')
    parser.add_argument('--debug-ao', action='store_true',
                        help='Print detailed AO specifications and validate centers')
    parser.add_argument('--verify-math', action='store_true',
                        help='Perform numerical grid integration to verify MO normalization')
    args = parser.parse_args()

    app = OrbitalVisualizationApp(
        quality=args.quality,
        resolution=args.resolution if args.resolution else 61,
        iso_value=args.isovalue if args.isovalue else 0.01,
        basis_format=args.basis_format,
        convention=args.convention,
        verify_math=args.verify_math,
        debug_phase=args.debug_phase,
        debug_ao=args.debug_ao
    )

    if args.input:
        if os.path.exists(args.input):
            app.export_visualization(args.input, args.output)
        else:
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
    else:
        app.run_interactive()


if __name__ == '__main__':
    main()
