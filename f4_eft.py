"""
F4 Exceptional Fourier Transform (F4-EFT)

The F4-EFT restricts the E8 Exceptional Fourier Transform to the
F4 sublattice, extracting the Jordan-algebraic core of the prime signal.

E8-EFT: E(λ) = Σₙ S(tₙ) · exp(2πi <αₙ, λ>)  (240 components)
F4-EFT: E_F4(λ) = Σₙ S(tₙ) · χ_F4(proj_f4(αₙ))  (48 components)

Where:
- S(tₙ) = g̃ₙ - 1 is the gap fluctuation
- αₙ is the E8 root assigned to gap n
- χ_F4 is the F4 character
- proj_f4 projects E8 roots to the F4 subspace
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from f4_lattice import F4Lattice
from salem_jordan import SalemJordanKernel
from jordan_algebra import JordanTrace


@dataclass
class F4EFTResult:
    """Results from F4 Exceptional Fourier Transform"""
    spectrum: np.ndarray              # 48-component F4 spectrum
    power_spectrum: np.ndarray        # |spectrum|²
    e8_spectrum: Optional[np.ndarray] # Original 240-component E8 spectrum
    f4_fraction: float                # Fraction of gaps mapping to F4
    dominant_roots: List[int]         # Indices of dominant F4 roots
    jordan_decomposition: np.ndarray  # Spectrum decomposed by Jordan trace


class F4ExceptionalFourierTransform:
    """
    Exceptional Fourier Transform restricted to F4 sublattice.

    This implements the "Phase-Locking to 52 Roots" step from
    the F4 tuning protocol.
    """

    def __init__(self, e8_lattice=None):
        """
        Initialize F4-EFT.

        Parameters
        ----------
        e8_lattice : E8Lattice, optional
            Parent E8 lattice for root assignments
        """
        self.e8 = e8_lattice
        self.f4 = F4Lattice(e8_lattice)
        self.jordan_trace = JordanTrace()
        self.salem_kernel = SalemJordanKernel(tau=0.5, f4_lattice=self.f4)

    def compute(self, normalized_gaps: np.ndarray,
                e8_root_assignments: np.ndarray,
                apply_salem_filter: bool = True) -> F4EFTResult:
        """
        Compute the F4 Exceptional Fourier Transform.

        Parameters
        ----------
        normalized_gaps : np.ndarray
            Normalized prime gaps g̃ₙ = gₙ / log(pₙ)
        e8_root_assignments : np.ndarray
            E8 root index for each gap (from E8 analysis)
        apply_salem_filter : bool
            Whether to apply Salem-Jordan filtering

        Returns
        -------
        F4EFTResult
            F4-EFT spectrum and analysis
        """
        n_gaps = len(normalized_gaps)

        # Initialize F4 spectrum (48 components)
        spectrum = np.zeros(48, dtype=complex)

        # Track which gaps map to F4
        f4_count = 0

        # Fluctuations from mean
        fluctuations = normalized_gaps - 1.0

        # Compute F4-EFT
        for n, (fluct, e8_idx) in enumerate(zip(fluctuations, e8_root_assignments)):
            # Project E8 root to F4
            f4_idx = self.f4.project_e8_to_f4(int(e8_idx))

            if f4_idx is None:
                continue  # Not an F4 root, skip

            f4_count += 1

            # Get F4 character weight
            chi = self.f4.get_character(f4_idx)

            # Phase from root norm (normalized by √2)
            root_norm = self.f4.root_norm(f4_idx)
            phase = 2 * np.pi * root_norm / np.sqrt(2)

            # Time-dependent phase
            time_phase = phase * n / n_gaps

            # Add to spectrum with character weighting
            spectrum[f4_idx] += fluct * chi * np.exp(1j * time_phase)

        # Apply Salem-Jordan filter if requested
        if apply_salem_filter:
            # Filter in magnitude space
            magnitudes = np.abs(spectrum)
            filter_result = self.salem_kernel.apply(magnitudes)
            spectrum = spectrum * filter_result.kernel_response

        # Compute power spectrum
        power_spectrum = np.abs(spectrum) ** 2

        # Find dominant roots
        sorted_indices = np.argsort(power_spectrum)[::-1]
        dominant_roots = sorted_indices[:10].tolist()

        # Jordan decomposition
        jordan_decomp = self._jordan_decomposition(spectrum)

        # Compute E8 spectrum for comparison (if E8 lattice available)
        e8_spectrum = None
        if self.e8 is not None:
            e8_spectrum = self._compute_e8_spectrum(fluctuations, e8_root_assignments)

        return F4EFTResult(
            spectrum=spectrum,
            power_spectrum=power_spectrum,
            e8_spectrum=e8_spectrum,
            f4_fraction=f4_count / n_gaps if n_gaps > 0 else 0,
            dominant_roots=dominant_roots,
            jordan_decomposition=jordan_decomp
        )

    def _compute_e8_spectrum(self, fluctuations: np.ndarray,
                             e8_root_assignments: np.ndarray) -> np.ndarray:
        """Compute full E8-EFT for comparison."""
        spectrum = np.zeros(240, dtype=complex)

        for n, (fluct, e8_idx) in enumerate(zip(fluctuations, e8_root_assignments)):
            e8_idx = int(e8_idx)
            root = self.e8.roots[e8_idx]
            root_norm = np.linalg.norm(root)

            phase = 2 * np.pi * root_norm / np.sqrt(2)
            time_phase = phase * n / len(fluctuations)

            spectrum[e8_idx] += fluct * np.exp(1j * time_phase)

        return spectrum

    def _jordan_decomposition(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Decompose F4 spectrum by Jordan trace value.

        Groups roots into bins based on their Jordan trace,
        revealing the idempotent structure.
        """
        # Compute Jordan trace for each F4 root
        traces = np.array([
            self.jordan_trace(self.f4.get_f4_root(i))
            for i in range(48)
        ])

        # Bin by trace value
        trace_bins = np.linspace(-3, 3, 13)  # 12 bins
        decomposition = np.zeros(12, dtype=complex)

        for i, (spec_val, trace) in enumerate(zip(spectrum, traces)):
            bin_idx = np.digitize(trace, trace_bins) - 1
            bin_idx = np.clip(bin_idx, 0, 11)
            decomposition[bin_idx] += spec_val

        return decomposition

    def phase_lock_analysis(self, normalized_gaps: np.ndarray,
                           e8_root_assignments: np.ndarray) -> Dict:
        """
        Analyze phase-locking between E8 and F4.

        This reveals whether the prime gaps exhibit the
        secondary coherence of F4 sub-harmonics.
        """
        # Compute F4-EFT (without Salem filter for analysis)
        result = self.compute(normalized_gaps, e8_root_assignments,
                             apply_salem_filter=False)

        # Analyze phase distribution
        phases = np.angle(result.spectrum)
        phase_coherence = np.abs(np.mean(np.exp(1j * phases)))

        # Analyze power distribution (normalized entropy)
        power_normalized = result.power_spectrum / (np.sum(result.power_spectrum) + 1e-10)
        power_entropy = -np.sum(
            power_normalized * np.log(power_normalized + 1e-10)
        ) / np.log(48)  # Normalize by max entropy

        # Long vs short root power
        long_indices = np.array(self.f4.long_root_indices)
        short_indices = np.array(self.f4.short_root_indices)
        long_power = np.sum(result.power_spectrum[long_indices])
        short_power = np.sum(result.power_spectrum[short_indices])

        # Jordan trace correlation
        jordan_power = np.abs(result.jordan_decomposition) ** 2
        jordan_peak_idx = np.argmax(jordan_power)
        jordan_peak_trace = -3 + 0.5 * jordan_peak_idx

        return {
            'f4_fraction': result.f4_fraction,
            'phase_coherence': phase_coherence,
            'power_entropy': power_entropy,
            'long_short_ratio': long_power / (short_power + 1e-10),
            'jordan_peak_trace': jordan_peak_trace,
            'dominant_roots': result.dominant_roots,
            'is_phase_locked': phase_coherence > 0.3,
            'has_jordan_structure': np.max(jordan_power) > 2 * np.mean(jordan_power)
        }

    def extract_crystalline_pattern(self, normalized_gaps: np.ndarray,
                                   e8_root_assignments: np.ndarray,
                                   n_vertices: int = 100) -> np.ndarray:
        """
        Extract discrete vertices from continuous ring pattern.

        These vertices are the F4 idempotents - the fixed points
        of the Albert algebra that anchor the prime standing wave.

        Parameters
        ----------
        normalized_gaps : np.ndarray
            Normalized prime gaps
        e8_root_assignments : np.ndarray
            E8 root assignments
        n_vertices : int
            Number of vertices to extract

        Returns
        -------
        np.ndarray
            Indices of gaps that correspond to F4 vertices
        """
        result = self.compute(normalized_gaps, e8_root_assignments)

        # Score each gap by its F4 resonance
        scores = np.zeros(len(normalized_gaps))

        for n, e8_idx in enumerate(e8_root_assignments):
            f4_idx = self.f4.project_e8_to_f4(int(e8_idx))

            if f4_idx is None:
                continue

            # Score based on power spectrum at this root
            scores[n] = result.power_spectrum[f4_idx]

            # Boost score for idempotent-type roots
            jordan = self.jordan_trace(self.f4.get_f4_root(f4_idx))
            if abs(abs(jordan) - 1.0) < 0.2:
                scores[n] *= 2.0

        # Select top-scoring gaps as vertices
        vertex_indices = np.argsort(scores)[::-1][:n_vertices]

        return vertex_indices


def f4_filter_primes(primes: np.ndarray, e8_root_assignments: np.ndarray,
                     e8_lattice) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter primes to keep only those with F4 root assignments.

    Parameters
    ----------
    primes : np.ndarray
        Array of prime numbers
    e8_root_assignments : np.ndarray
        E8 root index for each prime gap
    e8_lattice : E8Lattice
        E8 lattice instance

    Returns
    -------
    f4_primes : np.ndarray
        Primes whose gaps map to F4 roots
    f4_mask : np.ndarray
        Boolean mask for F4 primes
    """
    f4 = F4Lattice(e8_lattice)

    # Build mask
    f4_mask = np.zeros(len(primes), dtype=bool)

    # First prime always included
    f4_mask[0] = True

    # Check each gap's root assignment
    for i, e8_idx in enumerate(e8_root_assignments):
        if f4.is_f4_root(int(e8_idx)):
            f4_mask[i + 1] = True

    return primes[f4_mask], f4_mask


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    # Test with synthetic data
    print("=== F4-EFT Test ===")

    # Create F4-EFT without E8 (standalone mode)
    f4_eft = F4ExceptionalFourierTransform()

    # Synthetic normalized gaps
    np.random.seed(42)
    n_gaps = 10000
    normalized_gaps = 1.0 + 0.5 * np.random.randn(n_gaps)
    normalized_gaps = np.clip(normalized_gaps, 0.1, 5.0)

    # Synthetic E8 assignments (random for now)
    e8_assignments = np.random.randint(0, 240, n_gaps)

    # Compute F4-EFT
    result = f4_eft.compute(normalized_gaps, e8_assignments)

    print(f"F4 fraction: {result.f4_fraction:.3f}")
    print(f"Dominant F4 roots: {result.dominant_roots[:5]}")
    print(f"Power spectrum max: {np.max(result.power_spectrum):.2f}")

    # Phase lock analysis
    analysis = f4_eft.phase_lock_analysis(normalized_gaps, e8_assignments)
    print(f"\nPhase coherence: {analysis['phase_coherence']:.3f}")
    print(f"Power entropy: {analysis['power_entropy']:.3f}")
    print(f"Long/short ratio: {analysis['long_short_ratio']:.3f}")
    print(f"Is phase-locked: {analysis['is_phase_locked']}")
    print(f"Has Jordan structure: {analysis['has_jordan_structure']}")
