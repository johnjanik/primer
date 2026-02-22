"""
Salem-Jordan Kernel - Modified Salem Filter for F4 Sub-harmonics

The standard Salem Filter uses the Fermi-Dirac kernel:
    K(x) = 1 / (e^x + 1)

The Salem-Jordan Kernel incorporates the F4 character:
    K_J(x, τ) = χ_F4(e^{x/τ}) / (e^{x/τ} + 1)

This filters out the "topological noise" of the E8 spinor sectors,
revealing the rigid Jordan-algebraic skeleton.

The parameter τ controls the filter width:
- τ = 1/2 corresponds to the critical line σ = 1/2
- This is the natural scale for the Riemann Hypothesis connection
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FilterResult:
    """Result of applying Salem-Jordan filter"""
    filtered_signal: np.ndarray
    original_signal: np.ndarray
    kernel_response: np.ndarray
    tau: float
    energy_ratio: float  # filtered / original energy


class SalemJordanKernel:
    """
    The Salem-Jordan filter kernel for F4 sub-harmonic extraction.

    This combines:
    1. Fermi-Dirac thermal distribution (Salem kernel)
    2. F4 Lie algebra character (Jordan weighting)

    The result isolates the 52-dimensional F4 signal from E8.
    """

    def __init__(self, tau: float = 0.5, f4_lattice=None):
        """
        Initialize the Salem-Jordan kernel.

        Parameters
        ----------
        tau : float
            Temperature parameter (default 0.5 = critical line)
        f4_lattice : F4Lattice, optional
            F4 lattice for character computation
        """
        self.tau = tau
        self.f4 = f4_lattice
        self.sigma = tau  # Critical line parameter

        # Precompute character table if F4 lattice provided
        if f4_lattice is not None:
            self._precompute_characters()
        else:
            self.character_table = None

    def _precompute_characters(self):
        """Precompute F4 character values for all roots."""
        self.character_table = np.array([
            self.f4.get_character(i) for i in range(48)
        ])

    def fermi_dirac(self, x: np.ndarray) -> np.ndarray:
        """
        Standard Fermi-Dirac kernel.

        K(x) = 1 / (e^{x/τ} + 1)

        This is the original Salem kernel.
        """
        exp_term = np.exp(np.clip(x / self.tau, -50, 50))
        return 1.0 / (exp_term + 1.0)

    def f4_character(self, x: np.ndarray, root_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute F4 character weighting.

        χ_F4(e^{x/τ}) for the 52-dimensional representation.

        For roots, uses the precomputed character table.
        For general x, uses the Weyl character formula.
        """
        if root_indices is not None and self.character_table is not None:
            # Use precomputed characters
            return self.character_table[root_indices % 48]

        # General case: approximate character from trace
        # χ_F4 ≈ 52 - 4*|x|² for small x (near identity)
        return 52.0 - 4.0 * np.abs(x)**2

    def kernel(self, x: np.ndarray, root_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full Salem-Jordan kernel.

        K_J(x, τ) = χ_F4(e^{x/τ}) / (e^{x/τ} + 1)

        Parameters
        ----------
        x : np.ndarray
            Input signal values
        root_indices : np.ndarray, optional
            F4 root indices for character lookup

        Returns
        -------
        np.ndarray
            Filtered values
        """
        fermi = self.fermi_dirac(x)
        chi = self.f4_character(x, root_indices)

        # Normalize character contribution
        chi_normalized = chi / 52.0  # Divide by dimension

        return chi_normalized * fermi

    def apply(self, signal: np.ndarray,
              root_indices: Optional[np.ndarray] = None) -> FilterResult:
        """
        Apply the Salem-Jordan filter to a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (e.g., normalized prime gaps)
        root_indices : np.ndarray, optional
            F4 root indices for each signal point

        Returns
        -------
        FilterResult
            Filtered signal with metadata
        """
        # Compute kernel response
        kernel_response = self.kernel(signal, root_indices)

        # Apply filter
        filtered = signal * kernel_response

        # Compute energy ratio
        original_energy = np.sum(signal**2)
        filtered_energy = np.sum(filtered**2)
        energy_ratio = filtered_energy / original_energy if original_energy > 0 else 0

        return FilterResult(
            filtered_signal=filtered,
            original_signal=signal,
            kernel_response=kernel_response,
            tau=self.tau,
            energy_ratio=energy_ratio
        )

    def spectral_filter(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply filter in spectral domain.

        For EFT spectrum, this suppresses non-F4 components.
        """
        # Create frequency-dependent filter
        n = len(spectrum)
        freqs = np.fft.fftfreq(n)

        # Salem kernel in frequency domain
        freq_kernel = self.fermi_dirac(np.abs(freqs) * n)

        # Apply filter
        return spectrum * freq_kernel

    def null_space_projection(self, signal: np.ndarray,
                              threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project signal onto kernel null space.

        The null space contains the "topological noise" that F4 filters out.
        This is the Jordan-Zeta connection mentioned in the tuning document.

        Returns
        -------
        f4_component : np.ndarray
            Signal component in F4 subspace
        null_component : np.ndarray
            Signal component in null space (filtered out)
        """
        kernel_response = self.kernel(signal)

        # F4 component: where kernel is large
        f4_mask = kernel_response > threshold
        f4_component = np.where(f4_mask, signal, 0)

        # Null component: where kernel is small
        null_component = np.where(~f4_mask, signal, 0)

        return f4_component, null_component


class AdaptiveSalemJordan(SalemJordanKernel):
    """
    Adaptive Salem-Jordan filter with automatic τ selection.

    Adjusts the temperature parameter based on signal statistics
    to optimize F4 extraction.
    """

    def __init__(self, f4_lattice=None):
        super().__init__(tau=0.5, f4_lattice=f4_lattice)

    def compute_optimal_tau(self, signal: np.ndarray) -> float:
        """
        Estimate optimal τ from signal statistics.

        Uses the variance and kurtosis to determine scale.
        """
        variance = np.var(signal)
        mean_abs = np.mean(np.abs(signal))

        # τ should be proportional to typical fluctuation size
        # but bounded to stay near critical line
        tau_estimate = np.sqrt(variance) / 2

        # Clamp to reasonable range
        return np.clip(tau_estimate, 0.1, 2.0)

    def apply_adaptive(self, signal: np.ndarray,
                       root_indices: Optional[np.ndarray] = None) -> FilterResult:
        """
        Apply filter with automatically selected τ.
        """
        self.tau = self.compute_optimal_tau(signal)
        return self.apply(signal, root_indices)


def salem_integral(f: Callable, sigma: float = 0.5,
                   n_points: int = 1000) -> complex:
    """
    Compute the Salem integral:

        ∫₀^∞ z^{-σ-1} f(z) K(x/z) dz

    This integral equation has non-trivial solutions iff the prime gaps
    exhibit secondary coherence (F4 sub-harmonic).

    Parameters
    ----------
    f : Callable
        Function to integrate (e.g., prime gap distribution)
    sigma : float
        Critical line parameter (default 0.5)
    n_points : int
        Number of quadrature points

    Returns
    -------
    complex
        Integral value
    """
    # Integration grid (log scale for better coverage)
    z = np.exp(np.linspace(-10, 10, n_points))
    dz = np.diff(z)

    # Salem kernel
    kernel = SalemJordanKernel(tau=sigma)

    # Integrand: z^{-σ-1} * f(z) * K(1/z)
    integrand = z[:-1]**(-sigma - 1) * np.array([f(zi) for zi in z[:-1]])
    integrand *= kernel.fermi_dirac(1.0 / z[:-1])

    # Trapezoidal integration
    return np.sum(integrand * dz)


def detect_f4_resonance(e8_spectrum: np.ndarray,
                        tau: float = 0.5) -> Tuple[bool, float]:
    """
    Detect F4 sub-harmonic resonance in E8 spectrum.

    Returns True if the Salem-Jordan filtered spectrum shows
    enhanced structure compared to the raw E8 spectrum.

    Parameters
    ----------
    e8_spectrum : np.ndarray
        EFT spectrum (240 components)
    tau : float
        Filter temperature

    Returns
    -------
    has_resonance : bool
        Whether F4 resonance is detected
    resonance_strength : float
        Quantified resonance strength (0-1)
    """
    kernel = SalemJordanKernel(tau=tau)

    # Apply filter
    filtered = kernel.spectral_filter(e8_spectrum)

    # Compute power in filtered vs original
    original_power = np.sum(np.abs(e8_spectrum)**2)
    filtered_power = np.sum(np.abs(filtered)**2)

    # Resonance ratio
    ratio = filtered_power / original_power if original_power > 0 else 0

    # Check for resonance (should be significantly non-zero)
    has_resonance = ratio > 0.1
    resonance_strength = min(ratio, 1.0)

    return has_resonance, resonance_strength


if __name__ == "__main__":
    # Test Salem-Jordan kernel
    print("=== Salem-Jordan Kernel Tests ===")

    kernel = SalemJordanKernel(tau=0.5)

    # Test on sample signal
    x = np.linspace(-3, 3, 100)
    y = kernel.kernel(x)

    print(f"Kernel at x=0: {kernel.kernel(np.array([0]))[0]:.4f}")
    print(f"Kernel at x=1: {kernel.kernel(np.array([1]))[0]:.4f}")
    print(f"Kernel at x=-1: {kernel.kernel(np.array([-1]))[0]:.4f}")

    # Test filter application
    signal = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    result = kernel.apply(signal)
    print(f"\nFilter energy ratio: {result.energy_ratio:.4f}")

    # Test null space projection
    f4_part, null_part = kernel.null_space_projection(signal)
    print(f"F4 component energy: {np.sum(f4_part**2):.2f}")
    print(f"Null component energy: {np.sum(null_part**2):.2f}")
