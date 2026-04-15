"""
Basis Set Normalization Detection and Handling Module

This module provides sophisticated detection of basis set normalization
conventions used in quantum chemistry calculations. Different programs
(Gaussian, ORCA, PySCF, Molpro, etc.) have different conventions for
whether primitive Gaussian functions include normalization constants.

Features:
    - Automatic detection of normalization conventions
    - Support for multiple quantum chemistry packages
    - Heuristic analysis of coefficient magnitudes
    - Option to force or skip renormalization
    - Statistical analysis of basis set patterns

Classes:
    NormalizationDetector: Main class for detecting conventions
    NormalizationResult: Container for detection results

Functions:
    detect_normalization_convention: Quick detection function
    apply_normalization: Apply or skip normalization based on detection

Author: Pedro Lara
Version: 2.0.0
Date: 2024
"""

import numpy as np
import numba
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from quantum_viz.constants import EPSILON, PI
from quantum_viz.parsers.molden_parser import BasisSetInfo, MoldenVariant


class NormalizationConvention(Enum):
    """Enumeration of possible normalization conventions."""
    NORMALIZED = "normalized"
    UNNORMALIZED = "unnormalized"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class NormalizationResult:
    """Container for normalization detection results."""
    convention: NormalizationConvention = NormalizationConvention.UNKNOWN
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    recommendation: str = ""
    should_renormalize: bool = True
    program_hint: Optional[MoldenVariant] = None


class NormalizationDetector:
    """
    Detector for basis set normalization conventions.
    
    This class analyzes GTO basis set data to determine whether primitive
    Gaussian functions are already normalized or need normalization.
    
    Attributes:
        gto_data: GTO basis set data
        basis_info: Basis set metadata from Molden parser
        
    Example:
        >>> detector = NormalizationDetector(gto_data, basis_info)
        >>> result = detector.detect()
        >>> if result.should_renormalize:
        ...     print("Renormalization recommended")
    """
    
    def __init__(self, gto_data: List[Any], basis_info: Optional[BasisSetInfo] = None):
        """
        Initialize the normalization detector.
        
        Args:
            gto_data: List of GTOData objects
            basis_info: Optional BasisSetInfo from Molden parser
        """
        self.gto_data = gto_data
        self.basis_info = basis_info
        self.result = NormalizationResult()
        
        # Store program hint if available
        if basis_info and basis_info.variant:
            self.result.program_hint = basis_info.variant
    
    def detect(self) -> NormalizationResult:
        """
        Detect the normalization convention used in the basis set.
        
        Returns:
            NormalizationResult object with detection details
            
        The detection uses multiple heuristics:
            1. Program-specific conventions (if known)
            2. Magnitude of contraction coefficients
            3. Relationship between exponents and coefficients
            4. Statistical analysis of coefficient patterns
        """
        # First, check program-specific conventions
        if self._check_program_convention():
            return self.result
        
        # Sample primitives for analysis
        samples = self._sample_primitives(max_samples=50)
        
        if not samples:
            self.result.convention = NormalizationConvention.UNKNOWN
            self.result.confidence = 0.0
            self.result.reasons.append("No primitives found for analysis")
            self.result.should_renormalize = True
            return self.result
        
        # Analyze the samples
        analysis = self._analyze_samples(samples)
        
        # Determine convention based on analysis
        self._determine_convention(analysis)
        
        # Make recommendation
        self._make_recommendation()
        
        return self.result
    
    def _check_program_convention(self) -> bool:
        """
        Check if program-specific convention is known.
        
        Returns:
            True if convention was determined from program hint
        """
        if not self.basis_info or not self.basis_info.variant:
            return False
        
        variant = self.basis_info.variant
        
        # Known conventions
        if variant in [MoldenVariant.GAUSSIAN, MoldenVariant.MOLPRO]:
            self.result.convention = NormalizationConvention.NORMALIZED
            self.result.confidence = 0.9
            self.result.reasons.append(f"{variant.value} typically uses normalized primitives")
            self.result.should_renormalize = False
            return True
        elif variant == MoldenVariant.ORCA:
            self.result.convention = NormalizationConvention.UNNORMALIZED
            self.result.confidence = 0.85
            self.result.reasons.append(f"{variant.value} typically uses unnormalized primitives")
            self.result.should_renormalize = True
            return True
        elif variant == MoldenVariant.PYSCF:
            # PySCF can use either, need to analyze
            return False
        
        return False
    
    def _sample_primitives(self, max_samples: int = 50) -> List[Dict]:
        """
        Sample primitives from the basis set for analysis.
        
        Args:
            max_samples: Maximum number of samples to collect
            
        Returns:
            List of primitive dictionaries with metadata
        """
        samples = []
        
        for atom_gto in self.gto_data[:min(3, len(self.gto_data))]:
            for shell in atom_gto.shells[:min(3, len(atom_gto.shells))]:
                for prim in shell.primitives[:min(5, len(shell.primitives))]:
                    if len(samples) >= max_samples:
                        break
                    
                    # Determine angular momentum
                    l_val = 0
                    if shell.type and shell.type[0].lower() in 'spdfghi':
                        l_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
                        l_val = l_map.get(shell.type[0].lower(), 0)
                    
                    samples.append({
                        'exponent': prim['exponent'],
                        'coefficients': prim['coefficients'],
                        'l': l_val,
                        'scale_factor': shell.scale_factor
                    })
                    
                    if len(samples) >= max_samples:
                        break
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        
        return samples
    
    def _analyze_samples(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Perform statistical analysis on sampled primitives.
        
        Args:
            samples: List of primitive samples
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {
            'large_coeff_count': 0,
            'near_unity_count': 0,
            'norm_ratio_stats': [],
            'coeff_magnitudes': [],
            'exponents': [],
            'normalized_likelihood': 0.0
        }
        
        for sample in samples:
            alpha = sample['exponent']
            l_val = sample['l']
            coeffs = sample['coefficients']
            scale = sample['scale_factor']
            
            if not coeffs:
                continue
            
            # Calculate expected normalization constant
            expected_norm = self._calculate_expected_norm(alpha * scale**2, l_val)
            
            for coeff in coeffs:
                if abs(coeff) < EPSILON:
                    continue
                
                analysis['coeff_magnitudes'].append(abs(coeff))
                analysis['exponents'].append(alpha)
                
                # Check for large coefficients (suggests unnormalized)
                if abs(coeff) > 100:
                    analysis['large_coeff_count'] += 1
                
                # Check for coefficients near 1.0 (often unnormalized)
                if 0.9 < abs(coeff) < 1.1 and alpha > 0.5:
                    analysis['near_unity_count'] += 1
                
                # Compare with expected normalized value
                if expected_norm > EPSILON:
                    ratio = abs(coeff) / expected_norm
                    analysis['norm_ratio_stats'].append(ratio)
        
        # Calculate likelihood of being normalized
        if analysis['norm_ratio_stats']:
            ratios = np.array(analysis['norm_ratio_stats'])
            # If ratios are close to 1, likely normalized
            # If ratios are far from 1, likely unnormalized
            median_ratio = np.median(ratios)
            if 0.5 < median_ratio < 2.0:
                analysis['normalized_likelihood'] = 1.0 - abs(np.log10(median_ratio))
            else:
                analysis['normalized_likelihood'] = 0.0
        
        return analysis
    
    @staticmethod
    @numba.njit(cache=True)
    def _calculate_expected_norm(alpha: float, l: int) -> float:
        """
        Calculate expected normalization constant for a primitive GTO.
        
        Args:
            alpha: Scaled exponent
            l: Angular momentum quantum number
            
        Returns:
            Expected normalization constant
        """
        if alpha < EPSILON:
            return 0.0
        
        # Normalization for Cartesian GTO:
        # N = (2α/π)^(3/4) * (4α)^(l/2) / √((2l-1)!!)
        
        # Calculate (2l-1)!!
        double_fact = 1.0
        for i in range(1, 2 * l, 2):
            double_fact *= i
        
        if double_fact < EPSILON:
            return 0.0
        
        term1 = (2.0 * alpha / PI) ** 0.75
        term2 = (2.0 * np.sqrt(alpha)) ** l
        
        return term1 * term2 / np.sqrt(double_fact)
    
    def _determine_convention(self, analysis: Dict[str, Any]) -> None:
        """
        Determine normalization convention from analysis results.
        
        Args:
            analysis: Dictionary of analysis results
        """
        reasons = []
        normalized_score = 0
        unnormalized_score = 0
        
        # Check large coefficients
        if analysis['large_coeff_count'] > 0:
            unnormalized_score += 2
            reasons.append(f"Found {analysis['large_coeff_count']} unusually large coefficients")
        
        # Check coefficients near unity
        if analysis['near_unity_count'] > len(analysis['coeff_magnitudes']) * 0.3:
            unnormalized_score += 1
            reasons.append("Many coefficients are near 1.0 (typical for unnormalized)")
        
        # Check normalization ratios
        if analysis['normalized_likelihood'] > 0.7:
            normalized_score += 3
            reasons.append("Coefficient magnitudes match normalized pattern")
        elif analysis['normalized_likelihood'] > 0.3:
            normalized_score += 1
            reasons.append("Coefficient magnitudes partially match normalized pattern")
        else:
            unnormalized_score += 1
            reasons.append("Coefficient magnitudes do not match normalized pattern")
        
        # Check coefficient magnitude statistics
        if analysis['coeff_magnitudes']:
            coeffs = np.array(analysis['coeff_magnitudes'])
            median_coeff = np.median(coeffs)
            
            if median_coeff > 10:
                unnormalized_score += 1
                reasons.append(f"Median coefficient magnitude is large ({median_coeff:.2f})")
            elif median_coeff < 1.0:
                normalized_score += 1
                reasons.append(f"Median coefficient magnitude is small ({median_coeff:.2f})")
        
        # Determine final convention
        self.result.reasons = reasons[:5]  # Keep top 5 reasons
        
        if normalized_score > unnormalized_score:
            self.result.convention = NormalizationConvention.NORMALIZED
            self.result.confidence = normalized_score / (normalized_score + unnormalized_score)
        elif unnormalized_score > normalized_score:
            self.result.convention = NormalizationConvention.UNNORMALIZED
            self.result.confidence = unnormalized_score / (normalized_score + unnormalized_score)
        else:
            self.result.convention = NormalizationConvention.UNKNOWN
            self.result.confidence = 0.5
        
        # Adjust confidence based on sample size
        if len(analysis['coeff_magnitudes']) < 10:
            self.result.confidence *= 0.5
    
    def _make_recommendation(self) -> None:
        """Make a recommendation about whether to renormalize."""
        if self.result.convention == NormalizationConvention.NORMALIZED:
            self.result.recommendation = "Use coefficients as-is (already normalized)"
            self.result.should_renormalize = False
        elif self.result.convention == NormalizationConvention.UNNORMALIZED:
            self.result.recommendation = "Renormalize primitives during AO computation"
            self.result.should_renormalize = True
        else:
            self.result.recommendation = "Unable to determine with certainty - renormalizing is safer"
            self.result.should_renormalize = True


def detect_normalization_convention(gto_data: List[Any], 
                                   basis_info: Optional[BasisSetInfo] = None) -> NormalizationResult:
    """
    Quick function to detect normalization convention.
    
    Args:
        gto_data: GTO basis set data
        basis_info: Optional basis set metadata
        
    Returns:
        NormalizationResult object
    """
    detector = NormalizationDetector(gto_data, basis_info)
    return detector.detect()


@numba.njit(cache=True)
def apply_normalization_factor(alpha: float, coeff: float, l: int, 
                              should_normalize: bool) -> float:
    """
    Apply or skip normalization factor for a primitive.
    
    Args:
        alpha: Exponent value
        coeff: Contraction coefficient
        l: Angular momentum
        should_normalize: Whether to apply normalization
        
    Returns:
        Normalized or unnormalized coefficient
    """
    if not should_normalize or alpha < EPSILON:
        return coeff
    
    # Calculate normalization factor
    double_fact = 1.0
    for i in range(1, 2 * l, 2):
        double_fact *= i
    
    if double_fact < EPSILON:
        return coeff
    
    norm = (2.0 * alpha / PI) ** 0.75 * (2.0 * np.sqrt(alpha)) ** l / np.sqrt(double_fact)
    
    return coeff * norm
