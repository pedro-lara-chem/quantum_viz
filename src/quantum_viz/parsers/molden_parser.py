"""
Molden File Parser Module

This module provides comprehensive parsing capabilities for Molden format files,
which are commonly used in quantum chemistry to store molecular structure,
basis set information, and molecular orbital coefficients.

Features:
    - Parses atomic coordinates and elements
    - Extracts Gaussian-type orbital (GTO) basis set data
    - Reads molecular orbital coefficients, energies, and occupations
    - Detects Cartesian vs. spherical harmonic basis conventions
    - Handles various Molden format variants (Gaussian, ORCA, Molpro, etc.)
    - Provides detailed error reporting with line numbers

Classes:
    MoldenParser: Main parser class for Molden files
    BasisSetInfo: Container for basis set metadata

Functions:
    parse_molden_file: Convenience function for quick parsing
    detect_molden_variant: Identifies the source program of the Molden file

Author: Quantum Chemistry Visualization Team
Version: 2.0.0
Date: 2024
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Import constants
from quantum_viz.constants import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, L_QUANTUM_NUMBERS_MAP


class BasisConvention(Enum):
    """Enumeration of possible basis set conventions."""
    SPHERICAL = "spherical"
    CARTESIAN = "cartesian"
    UNKNOWN = "unknown"


class MoldenVariant(Enum):
    """Enumeration of known Molden file variants."""
    GAUSSIAN = "gaussian"
    ORCA = "orca"
    PYSCF = "pyscf"
    MOLPRO = "molpro"
    GAMESS = "gamess"
    UNKNOWN = "unknown"


@dataclass
class BasisSetInfo:
    """Container for basis set metadata extracted from Molden file."""
    convention: BasisConvention = BasisConvention.UNKNOWN
    uses_5d: bool = False
    uses_6d: bool = False
    uses_7f: bool = False
    uses_10f: bool = False
    uses_9g: bool = False
    uses_15g: bool = False
    variant: MoldenVariant = MoldenVariant.UNKNOWN
    normalized_primitives: Optional[bool] = None


@dataclass
class AtomData:
    """Container for atomic data from Molden file."""
    label: str
    number_in_molden: int
    atomic_number: int
    x: float
    y: float
    z: float
    unit: str = "AU"
    
    def get_coordinates_in_bohr(self) -> np.ndarray:
        """Convert coordinates to Bohr (AU)."""
        coords = np.array([self.x, self.y, self.z])
        if self.unit.lower() == "angs":
            coords = coords * ANGSTROM_TO_BOHR
        return coords
    
    def get_coordinates_in_angstroms(self) -> np.ndarray:
        """Convert coordinates to Angstroms."""
        coords = np.array([self.x, self.y, self.z])
        if self.unit.lower() == "au":
            coords = coords * BOHR_TO_ANGSTROM
        return coords


@dataclass
class GTOShell:
    """Container for GTO shell data."""
    type: str
    scale_factor: float
    primitives: List[Dict[str, Union[float, List[float]]]]
    n_quantum_number: Optional[int] = None


@dataclass
class GTOData:
    """Container for GTO basis data for an atom."""
    atom_index: int
    shells: List[GTOShell]


@dataclass
class MOData:
    """Container for molecular orbital data."""
    symmetry: str = ""
    energy: float = 0.0
    spin: str = ""
    occupancy: float = 0.0
    coefficients: List[float] = field(default_factory=list)


class MoldenParser:
    """
    Comprehensive parser for Molden format files with unit detection.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the Molden parser.
        
        Args:
            filepath: Path to the Molden file to parse
        """
        self.filepath = filepath
        self.atoms_data: List[AtomData] = []
        self.gto_data: List[GTOData] = []
        self.mo_data: List[MOData] = []
        self.basis_info = BasisSetInfo()
        self.coordinate_unit = "AU"  # Default, will be updated during parsing
        
    def parse(self) -> Tuple[List[AtomData], List[GTOData], List[MOData], BasisSetInfo]:
        """
        Parse the Molden file and return all extracted data.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Molden file not found: {self.filepath}")
            
        try:
            with open(self.filepath, 'r') as f:
                self._parse_sections(f)
                self._detect_basis_conventions()
                self._post_process_data()
        except Exception as e:
            raise ValueError(f"Error parsing Molden file: {e}")
            
        return self.atoms_data, self.gto_data, self.mo_data, self.basis_info
    
    def get_coordinates(self, unit: str = "bohr") -> np.ndarray:
        """
        Get atomic coordinates in specified units.
        
        Args:
            unit: Target unit ("bohr" or "angstrom")
            
        Returns:
            Array of shape (n_atoms, 3) with coordinates in specified units
        """
        coords = []
        for atom in self.atoms_data:
            if unit.lower() in ["bohr", "au"]:
                coords.append(atom.get_coordinates_in_bohr())
            elif unit.lower() in ["angstrom", "angs", "a"]:
                coords.append(atom.get_coordinates_in_angstroms())
            else:
                raise ValueError(f"Unknown unit: {unit}")
        return np.array(coords)
    
    def get_symbols(self) -> List[str]:
        """Get list of element symbols."""
        return [atom.label for atom in self.atoms_data]
    
    def _parse_sections(self, file_handle) -> None:
        """
        Parse all sections of the Molden file.
        """
        current_section = None
        current_atom_gto_shells = []
        current_atom_gto_idx = -1
        current_mo_coeffs = []
        current_mo_details = {}
        atom_units = "AU"  # Default unit as string, not list
        
        # Track the line number for error reporting
        for line_number, raw_line in enumerate(file_handle, 1):
            line = raw_line.strip()
            
            if not line:
                continue
                
            # Handle section headers
            if line.startswith('['):
                # Finalize previous sections
                if current_section == "MO" and current_mo_details:
                    self._finalize_mo(current_mo_details, current_mo_coeffs)
                    current_mo_coeffs = []
                    current_mo_details = {}
                
                # --- NEW FIX: Finalize the last atom's GTOs when leaving the [GTO] section ---
                elif current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                    self.gto_data.append(GTOData(
                        atom_index=current_atom_gto_idx,
                        shells=current_atom_gto_shells
                    ))
                    # Reset variables to prevent duplicate appending
                    current_atom_gto_shells = []
                    current_atom_gto_idx = -1
                # -----------------------------------------------------------------------------

                # Parse header and update atom_units
                current_section, atom_units = self._parse_section_header(line, atom_units)
                continue
            
            # Parse data based on current section
            if current_section == "Atoms":
                self._parse_atom_line(line, line_number, atom_units)
            elif current_section == "GTO":
                current_atom_gto_idx, current_atom_gto_shells = self._parse_gto_line(
                    line, line_number, file_handle, 
                    current_atom_gto_idx, current_atom_gto_shells
                )
            elif current_section == "MO":
                current_mo_details, current_mo_coeffs = self._parse_mo_line(
                    line, line_number, current_mo_details, current_mo_coeffs
                )
        
        # Finalize any remaining data
        self._finalize_parsing(current_section, current_atom_gto_idx, 
                              current_atom_gto_shells, current_mo_details, 
                              current_mo_coeffs)
    
    def _parse_section_header(self, line: str, atom_units: str) -> Tuple[str, str]:
        line_lower = line.lower()
        # Check for basis set convention hints
        if '[5d]' in line_lower or '[5d7f]' in line_lower:
            self.basis_info.uses_5d = True
            self.basis_info.convention = BasisConvention.SPHERICAL
        elif '[6d]' in line_lower or '[6d10f]' in line_lower:
            self.basis_info.uses_6d = True
            self.basis_info.convention = BasisConvention.CARTESIAN
        elif '[7f]' in line_lower:
            self.basis_info.uses_7f = True
        elif '[10f]' in line_lower:
            self.basis_info.uses_10f = True
        elif '[9g]' in line_lower:
            self.basis_info.uses_9g = True
        elif '[15g]' in line_lower:
            self.basis_info.uses_15g = True

        if line_lower.startswith("[atoms]"):
            if "angs" in line_lower:
                atom_units = "Angs"
                self.coordinate_unit = "Angs"
            else:
                atom_units = "AU"
                self.coordinate_unit = "AU"
            return "Atoms", atom_units
        elif line_lower.startswith("[gto]"):
            return "GTO", atom_units
        elif line_lower.startswith("[mo]"):
            return "MO", atom_units
        else:
            return line, atom_units
    
    def _parse_atom_line(self, line: str, line_number: int, atom_units: str) -> None:
        """
        Parse an atom specification line.
        """
        parts = line.split()
        if len(parts) >= 5:
            try:
                # Handle different atom line formats
                # Format 1: Label Index AtomicNumber X Y Z (most common)
                # Format 2: Label AtomicNumber X Y Z (no index)
                if len(parts) == 6:
                    label = parts[0]
                    number_in_molden = int(parts[1])
                    atomic_number = int(float(parts[2]))
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                elif len(parts) == 5:
                    label = parts[0]
                    number_in_molden = len(self.atoms_data) + 1  # Auto-assign
                    atomic_number = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                else:
                    return
                
                atom = AtomData(
                    label=label,
                    number_in_molden=number_in_molden,
                    atomic_number=atomic_number,
                    x=x,
                    y=y,
                    z=z,
                    unit=atom_units
                )
                self.atoms_data.append(atom)
            except ValueError:
                print(f"Warning (line {line_number}): Could not parse atom line: {line}")
    
    def _parse_gto_line(self, line: str, line_number: int, file_handle,
                       current_atom_gto_idx: int, 
                       current_atom_gto_shells: List) -> Tuple[int, List]:
        """
        Parse a GTO section line.
        """
        parts = line.split()
        
        # Check for atom index line
        if len(parts) <= 2 and all(p.replace('-', '').isdigit() for p in parts if p):
            # Finalize previous atom
            if current_atom_gto_shells and current_atom_gto_idx != -1:
                self.gto_data.append(GTOData(
                    atom_index=current_atom_gto_idx,
                    shells=current_atom_gto_shells
                ))
            
            atom_seq_num = int(parts[0])
            atom_idx = self._find_atom_index(atom_seq_num)
            return atom_idx, []
        
        # Check for shell definition
        elif parts and parts[0].isalpha() and parts[0].lower() in ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'sp']:
            shell = self._parse_shell_definition(line, line_number, file_handle)
            if shell:
                current_atom_gto_shells.append(shell)
        
        return current_atom_gto_idx, current_atom_gto_shells
    
    def _parse_shell_definition(self, line: str, line_number: int, 
                            file_handle) -> Optional[GTOShell]:
        """
        Parse a GTO shell definition.
        """
        parts = line.split()
        shell_type = parts[0].lower()
        
        # Accept any combination of s, p, d, f, g, h, i, k, l
        # This handles 's', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'sp', 'spd', etc.
        valid_shell_chars = set('spdfghikl')
        if not all(c in valid_shell_chars for c in shell_type):
            print(f"Warning (line {line_number}): Unknown shell type '{shell_type}'")
            return None
        
        try:
            if len(parts) == 3:
                num_primitives = int(parts[1])
                scale_factor = float(parts[2])
            elif len(parts) == 2:
                num_primitives = int(parts[1])
                scale_factor = 1.0
            else:
                return None
        except ValueError:
            print(f"Warning (line {line_number}): Malformed shell definition: {line}")
            return None
        
        # Parse primitives
        primitives = []
        for i in range(num_primitives):
            try:
                primitive_line = next(file_handle).strip()
                prim_parts = primitive_line.split()
                exponent = float(prim_parts[0])
                coefficients = [float(c) for c in prim_parts[1:]]
                primitives.append({
                    'exponent': exponent,
                    'coefficients': coefficients
                })
            except (StopIteration, ValueError, IndexError) as e:
                print(f"Warning (near line {line_number + i + 1}): Error parsing primitive: {e}")
                break
        
        # For each character in shell_type, we'll create a separate shell entry
        # But we store it as a combined shell with all coefficients
        n_quantum = self._estimate_n_quantum_number(shell_type[0])
        
        return GTOShell(
            type=shell_type,
            scale_factor=scale_factor,
            primitives=primitives,
            n_quantum_number=n_quantum
        )
    
    def _parse_mo_line(self, line: str, line_number: int,
                      current_mo_details: Dict, 
                      current_mo_coeffs: List) -> Tuple[Dict, List]:
        """
        Parse a molecular orbital section line.
        """
        line_lower = line.lower()
        
        if line_lower.startswith("sym="):
            if current_mo_details:
                self._finalize_mo(current_mo_details, current_mo_coeffs)
            current_mo_details = {'symmetry': line.split('=')[1].strip()}
            current_mo_coeffs = []
            
        elif line_lower.startswith("ene="):
            try:
                current_mo_details['energy'] = float(line.split('=')[1].strip())
            except ValueError:
                print(f"Warning (line {line_number}): Could not parse MO energy: {line}")
                
        elif line_lower.startswith("spin="):
            current_mo_details['spin'] = line.split('=')[1].strip()
            
        elif line_lower.startswith("occup="):
            try:
                current_mo_details['occupancy'] = float(line.split('=')[1].strip())
            except ValueError:
                print(f"Warning (line {line_number}): Could not parse MO occupancy: {line}")
                
        else:
            parts = line.split()
            if len(parts) == 2:
                try:
                    coeff = float(parts[1])
                    current_mo_coeffs.append(coeff)
                except ValueError:
                    print(f"Warning (line {line_number}): Could not parse MO coefficient: {line}")
                    
        return current_mo_details, current_mo_coeffs
    
    def _find_atom_index(self, molden_number: int) -> int:
        """
        Find the atom index in atoms_data matching the Molden number.
        """
        for idx, atom in enumerate(self.atoms_data):
            if atom.number_in_molden == molden_number:
                return idx
        return molden_number - 1
    
    def _estimate_n_quantum_number(self, shell_type: str) -> int:
        """
        Estimate principal quantum number from shell type.
        """
        if shell_type and shell_type[0] in L_QUANTUM_NUMBERS_MAP:
            l_val = L_QUANTUM_NUMBERS_MAP[shell_type[0]]
            return l_val + 1
        return 1
    
    def _finalize_mo(self, mo_details: Dict, mo_coeffs: List) -> None:
        """
        Create MOData object and add to list.
        """
        if mo_details and mo_coeffs:
            mo = MOData(
                symmetry=mo_details.get('symmetry', ''),
                energy=mo_details.get('energy', 0.0),
                spin=mo_details.get('spin', ''),
                occupancy=mo_details.get('occupancy', 0.0),
                coefficients=mo_coeffs.copy()
            )
            self.mo_data.append(mo)
    
    def _finalize_parsing(self, current_section: str, current_atom_gto_idx: int,
                         current_atom_gto_shells: List, current_mo_details: Dict,
                         current_mo_coeffs: List) -> None:
        """
        Finalize any remaining data after parsing all lines.
        """
        # Finalize GTO data
        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
            self.gto_data.append(GTOData(
                atom_index=current_atom_gto_idx,
                shells=current_atom_gto_shells
            ))
        
        # Finalize MO data
        if current_section == "MO" and current_mo_details and current_mo_coeffs:
            self._finalize_mo(current_mo_details, current_mo_coeffs)
    
    def _detect_basis_conventions(self) -> None:
        """Detect basis set conventions and Molden variant."""
        try:
            with open(self.filepath, 'r') as f:
                content = f.read().lower()
                
                if 'gaussian' in content or 'g16' in content or 'g09' in content:
                    self.basis_info.variant = MoldenVariant.GAUSSIAN
                elif 'orca' in content:
                    self.basis_info.variant = MoldenVariant.ORCA
                elif 'pyscf' in content:
                    self.basis_info.variant = MoldenVariant.PYSCF
                elif 'molpro' in content:
                    self.basis_info.variant = MoldenVariant.MOLPRO
                elif 'gamess' in content or 'firefly' in content:
                    self.basis_info.variant = MoldenVariant.GAMESS
        except:
            pass
        
        # Determine if primitives are normalized based on variant
        if self.basis_info.variant in [MoldenVariant.GAUSSIAN, MoldenVariant.MOLPRO]:
            self.basis_info.normalized_primitives = True
        elif self.basis_info.variant in [MoldenVariant.ORCA]:
            self.basis_info.normalized_primitives = False
    
    def _post_process_data(self) -> None:
        """Post-process parsed data for consistency."""
        # Count expected AOs
        total_expected_aos = 0
        for atom_gto in self.gto_data:
            for shell in atom_gto.shells:
                for char in shell.type.lower():
                    if char in L_QUANTUM_NUMBERS_MAP:
                        l_val = L_QUANTUM_NUMBERS_MAP[char]
                        total_expected_aos += 2 * l_val + 1
                    else:
                        print(f"Warning: Unknown shell character '{char}' in '{shell.type}'")
        
        print(f"  Debug: Expected AOs from GTO section: {total_expected_aos}")
        
        # Refine principal quantum numbers
        for atom_gto in self.gto_data:
            for shell in atom_gto.shells:
                if shell.type and shell.type[0] in L_QUANTUM_NUMBERS_MAP:
                    l_val = L_QUANTUM_NUMBERS_MAP[shell.type[0]]
                    shell.n_quantum_number = l_val + 1


def parse_molden_file(filepath: str) -> Tuple[List, List, List, BasisSetInfo]:
    """
    Convenience function to parse a Molden file.
    """
    parser = MoldenParser(filepath)
    return parser.parse()
