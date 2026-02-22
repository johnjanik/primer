"""
E8-F4 Prime Analysis - Complete Pipeline

Combines E8 root analysis with F4 sub-harmonic extraction
to reveal the crystalline structure of the prime distribution.

Expected visualization result:
- E8 shows concentric RINGS (energy levels)
- F4 shows discrete VERTICES (idempotents)

If you see the vertices emerge from the rings, you have
successfully decoded the Jordan-algebraic skeleton.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import time
import re
from typing import Dict, Tuple, Optional

# Import F4 tuning modules
from f4_lattice import F4Lattice
from salem_jordan import SalemJordanKernel
from jordan_algebra import JordanTrace, jordan_trace_coloring
from f4_eft import F4ExceptionalFourierTransform


# =============================================================================
# E8 Lattice (copied from existing code for self-containment)
# =============================================================================

class E8Lattice:
    """The E8 root lattice with 240 roots in R^8"""

    def __init__(self):
        self.roots = self._generate_roots()
        self.min_norm = np.sqrt(2)
        self.projected_slopes = self._compute_projection_slopes()

    def _generate_roots(self) -> np.ndarray:
        roots = []
        # Type I: 112 roots
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = s1
                        root[j] = s2
                        roots.append(root)
        # Type II: 128 roots
        for mask in range(256):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(8)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                root = np.array([s * 0.5 for s in signs])
                roots.append(root)
        return np.array(roots)

    def _compute_projection_slopes(self) -> np.ndarray:
        slopes = []
        for root in self.roots:
            x = np.sum(root[:4])
            y = np.sum(root[4:])
            if abs(x) > 0.01:
                slopes.append(y / x)
            else:
                slopes.append(np.sign(y) * 10)
        return np.array(slopes)

    def assign_root(self, normalized_gap: float) -> int:
        target_norm = np.sqrt(max(normalized_gap, 0.01))
        phase = (target_norm / self.min_norm) % 1.0
        return int(phase * len(self.roots)) % len(self.roots)


# =============================================================================
# Ulam Spiral Coordinates
# =============================================================================

def ulam_coords(primes: np.ndarray) -> np.ndarray:
    """Compute Ulam spiral coordinates for primes."""
    coords = np.zeros((len(primes), 2), dtype=np.int32)

    for i, p in enumerate(primes):
        if p <= 0:
            continue
        k = int(np.ceil((np.sqrt(p) - 1) / 2))
        t = 2 * k + 1
        m = t * t
        t -= 1

        if p >= m - t:
            coords[i] = [k - (m - p), -k]
        elif p >= m - 2*t:
            coords[i] = [-k, -k + (m - t - p)]
        elif p >= m - 3*t:
            coords[i] = [-k + (m - 2*t - p), k]
        else:
            coords[i] = [k, k - (m - 3*t - p)]

    return coords


# =============================================================================
# Prime Loading
# =============================================================================

def load_primes(base_dir: str = "/home/john/mynotes/HodgedeRham", max_primes: int = 1000000) -> np.ndarray:
    """Load primes from text files."""
    all_primes = []
    for i in range(1, 51):
        filename = Path(base_dir) / f"primes{i}.txt"
        if not filename.exists():
            break
        with open(filename, 'r') as f:
            for line in f:
                numbers = re.findall(r'\b\d+\b', line.strip())
                all_primes.extend([int(n) for n in numbers])
        if len(all_primes) >= max_primes:
            break
    primes = np.array(all_primes, dtype=np.int64)
    primes = np.unique(primes[primes > 1])
    return primes[:max_primes]


# =============================================================================
# E8-F4 Analyzer
# =============================================================================

class E8F4PrimeAnalyzer:
    """
    Combined E8-F4 analysis pipeline.

    Performs:
    1. E8 root assignment (240 roots)
    2. F4 filtering (48 roots)
    3. Salem-Jordan kernel application
    4. Jordan trace computation
    """

    def __init__(self):
        self.e8 = E8Lattice()
        self.f4 = F4Lattice(self.e8)
        self.f4_eft = F4ExceptionalFourierTransform(self.e8)
        self.salem_kernel = SalemJordanKernel(tau=0.5, f4_lattice=self.f4)
        self.jordan_trace = JordanTrace()

    def analyze(self, primes: np.ndarray) -> Dict:
        """
        Full E8-F4 analysis of prime sequence.

        Returns dict with all analysis results.
        """
        print("  Computing gaps...")
        gaps = np.diff(primes.astype(np.float64))
        log_primes = np.log(primes[:-1].astype(np.float64))
        log_primes[log_primes < 1] = 1
        normalized_gaps = gaps / log_primes

        print("  Assigning E8 roots...")
        e8_assignments = np.array([
            self.e8.assign_root(g) for g in normalized_gaps
        ])

        print("  Computing E8 projection slopes...")
        e8_slopes = self.e8.projected_slopes[e8_assignments]

        print("  Filtering to F4...")
        f4_mask = np.array([
            self.f4.is_f4_root(int(idx)) for idx in e8_assignments
        ])
        f4_assignments = np.array([
            self.f4.project_e8_to_f4(int(idx)) if f4_mask[i] else -1
            for i, idx in enumerate(e8_assignments)
        ])

        print("  Computing Jordan traces...")
        jordan_traces = np.zeros(len(e8_assignments))
        for i, (is_f4, f4_idx) in enumerate(zip(f4_mask, f4_assignments)):
            if is_f4 and f4_idx >= 0:
                root = self.f4.get_f4_root(f4_idx)
                jordan_traces[i] = self.jordan_trace(root)
            else:
                jordan_traces[i] = np.nan

        print("  Computing F4-EFT spectrum...")
        f4_result = self.f4_eft.compute(normalized_gaps, e8_assignments,
                                        apply_salem_filter=False)  # Disable filter to see raw signal

        print("  Phase-lock analysis...")
        phase_analysis = self.f4_eft.phase_lock_analysis(
            normalized_gaps, e8_assignments
        )

        print("  Extracting crystalline vertices...")
        vertex_indices = self.f4_eft.extract_crystalline_pattern(
            normalized_gaps, e8_assignments, n_vertices=500
        )

        return {
            'primes': primes,
            'gaps': gaps,
            'normalized_gaps': normalized_gaps,
            'e8_assignments': e8_assignments,
            'e8_slopes': e8_slopes,
            'f4_mask': f4_mask,
            'f4_assignments': f4_assignments,
            'jordan_traces': jordan_traces,
            'f4_spectrum': f4_result.spectrum,
            'f4_power': f4_result.power_spectrum,
            'f4_fraction': f4_result.f4_fraction,
            'phase_analysis': phase_analysis,
            'vertex_indices': vertex_indices,
        }


# =============================================================================
# Visualization
# =============================================================================

class E8F4Visualizer:
    """Visualize E8 vs F4 structure in prime spirals."""

    def __init__(self, output_dir: str = "../spiral_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_e8_vs_f4_comparison(self, data: Dict,
                                  figsize: Tuple[int, int] = (20, 10),
                                  dpi: int = 300,
                                  save_name: str = "e8_f4_comparison.png"):
        """
        Side-by-side comparison: E8 rings vs F4 vertices.
        """
        primes = data['primes']
        coords = ulam_coords(primes)

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                                  facecolor='black')
        fig.suptitle('E8 Energy Levels (Rings) vs F4 Idempotents (Vertices)',
                    fontsize=16, fontweight='bold', color='white')

        # === Left: E8 Projection Slope (shows rings) ===
        ax1 = axes[0]
        ax1.set_facecolor('black')

        e8_slopes = data['e8_slopes']
        clipped_slopes = np.clip(e8_slopes, -3, 3)

        scatter1 = ax1.scatter(coords[1:, 0], coords[1:, 1],
                               c=clipped_slopes, cmap='coolwarm',
                               s=0.5, alpha=0.7, vmin=-3, vmax=3)

        ax1.set_aspect('equal')
        ax1.set_title('E8: Projection Slope\n(Concentric Rings = Energy Levels)',
                     color='white', fontsize=12)
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('white')

        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('E8 Slope (y/x)', color='white')
        cbar1.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')

        # === Right: F4 Jordan Trace (shows vertices) ===
        ax2 = axes[1]
        ax2.set_facecolor('black')

        f4_mask = data['f4_mask']
        jordan_traces = data['jordan_traces']

        # Plot non-F4 primes in dark gray
        non_f4_mask = ~np.array([False] + list(f4_mask))  # Shift by 1 for coords
        ax2.scatter(coords[non_f4_mask, 0], coords[non_f4_mask, 1],
                   c='#333333', s=0.3, alpha=0.3)

        # Plot F4 primes colored by Jordan trace
        f4_coords_mask = np.array([False] + list(f4_mask))
        valid_jordan = ~np.isnan(jordan_traces)
        full_valid = np.array([False] + list(valid_jordan))
        full_valid = full_valid & f4_coords_mask

        jordan_colors = np.zeros(len(primes))
        jordan_colors[1:] = jordan_traces
        jordan_valid = jordan_colors[full_valid]
        clipped_jordan = np.clip(jordan_valid, -2, 2)

        scatter2 = ax2.scatter(coords[full_valid, 0], coords[full_valid, 1],
                               c=clipped_jordan, cmap='viridis',
                               s=1.5, alpha=0.9, vmin=-2, vmax=2)

        ax2.set_aspect('equal')
        ax2.set_title('F4: Jordan Trace\n(Discrete Vertices = Idempotents)',
                     color='white', fontsize=12)
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')

        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('Jordan Trace', color='white')
        cbar2.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='black')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_f4_crystalline_grid(self, data: Dict,
                                  figsize: Tuple[int, int] = (16, 16),
                                  dpi: int = 300,
                                  save_name: str = "f4_crystalline_grid.png"):
        """
        F4 crystalline grid visualization.

        Shows only F4-filtered primes, highlighting the
        discrete vertex structure.
        """
        primes = data['primes']
        coords = ulam_coords(primes)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
        ax.set_facecolor('black')

        f4_mask = data['f4_mask']
        jordan_traces = data['jordan_traces']
        vertex_indices = data['vertex_indices']

        # Prepare F4 coordinates
        f4_coords_mask = np.array([False] + list(f4_mask))
        valid_jordan = ~np.isnan(jordan_traces)
        full_valid = np.array([False] + list(valid_jordan))
        full_valid = full_valid & f4_coords_mask

        jordan_colors = np.zeros(len(primes))
        jordan_colors[1:] = jordan_traces

        # Plot all F4 primes
        jordan_valid = jordan_colors[full_valid]
        clipped_jordan = np.clip(jordan_valid, -2, 2)

        scatter = ax.scatter(coords[full_valid, 0], coords[full_valid, 1],
                            c=clipped_jordan, cmap='plasma',
                            s=1, alpha=0.7, vmin=-2, vmax=2)

        # Highlight crystalline vertices
        vertex_mask = np.zeros(len(primes), dtype=bool)
        # vertex_indices are gap indices, shift by 1 for prime/coord indices
        valid_vertices = vertex_indices[vertex_indices + 1 < len(primes)]
        vertex_mask[valid_vertices + 1] = True
        vertex_mask = vertex_mask & full_valid

        ax.scatter(coords[vertex_mask, 0], coords[vertex_mask, 1],
                  c='white', s=10, alpha=1.0, marker='o',
                  edgecolors='yellow', linewidths=0.5,
                  label=f'Crystalline Vertices ({np.sum(vertex_mask)})')

        ax.set_aspect('equal')
        ax.set_title(f'F4 Crystalline Grid\n{np.sum(f4_mask):,} F4 primes '
                    f'({100*data["f4_fraction"]:.1f}% of gaps)',
                    fontsize=16, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        ax.legend(loc='upper right', facecolor='black', labelcolor='white')

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Jordan Trace', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='black')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_f4_spectrum(self, data: Dict,
                         save_name: str = "f4_spectrum.png"):
        """Plot F4-EFT spectrum analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('F4 Exceptional Fourier Transform Analysis',
                    fontsize=14, fontweight='bold')

        # Power spectrum
        ax = axes[0, 0]
        power = data['f4_power']
        colors = plt.cm.viridis(np.linspace(0, 1, 48))
        ax.bar(range(48), power, color=colors)
        ax.set_xlabel('F4 Root Index')
        ax.set_ylabel('Power')
        ax.set_title('F4-EFT Power Spectrum')

        # Top components
        ax = axes[0, 1]
        sorted_idx = np.argsort(power)[::-1]
        top_n = 15
        ax.barh(range(top_n), power[sorted_idx[:top_n]],
               color=[colors[i] for i in sorted_idx[:top_n]])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f'Root {i}' for i in sorted_idx[:top_n]])
        ax.set_xlabel('Power')
        ax.set_title('Top 15 F4 Components')
        ax.invert_yaxis()

        # Phase distribution
        ax = axes[1, 0]
        spectrum = data['f4_spectrum']
        phases = np.angle(spectrum)
        ax.hist(phases, bins=30, color='steelblue', edgecolor='white')
        ax.set_xlabel('Phase (radians)')
        ax.set_ylabel('Count')
        ax.set_title(f'Phase Distribution\nCoherence: '
                    f'{data["phase_analysis"]["phase_coherence"]:.3f}')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)

        # Long vs short roots
        ax = axes[1, 1]
        f4 = F4Lattice()
        long_power = np.sum(power[f4.long_root_indices])
        short_power = np.sum(power[f4.short_root_indices])
        ax.pie([long_power, short_power],
               labels=[f'Long (√2)\n{long_power:.1f}',
                      f'Short (1)\n{short_power:.1f}'],
               colors=['#4ECDC4', '#FF6B6B'],
               autopct='%1.1f%%')
        ax.set_title('Long vs Short Root Power')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_phase_analysis_summary(self, data: Dict,
                                    save_name: str = "f4_phase_analysis.png"):
        """Summary of F4 phase-locking analysis."""
        pa = data['phase_analysis']

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = [
            ('F4 Fraction', pa['f4_fraction'], 0, 1),
            ('Phase Coherence', pa['phase_coherence'], 0, 1),
            ('Power Entropy', pa['power_entropy'], 0, 1),
            ('Long/Short Ratio', min(pa['long_short_ratio'], 5) / 5, 0, 1),
        ]

        names = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        colors = ['#2ecc71' if v > 0.3 else '#e74c3c' for v in values]

        bars = ax.barh(names, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Normalized Value')
        ax.set_title('F4 Phase-Lock Analysis Summary')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center')

        # Add status indicators
        status = []
        status.append(f"Phase-Locked: {'✓' if pa['is_phase_locked'] else '✗'}")
        status.append(f"Jordan Structure: {'✓' if pa['has_jordan_structure'] else '✗'}")
        status.append(f"Jordan Peak Trace: {pa['jordan_peak_trace']:.2f}")

        ax.text(0.98, 0.02, '\n'.join(status),
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("E8-F4 PRIME ANALYSIS")
    print("Revealing the Jordan-Algebraic Skeleton")
    print("=" * 70)

    start_time = time.time()

    # Configuration
    MAX_PRIMES = 2000000

    # Load primes
    print(f"\nLoading up to {MAX_PRIMES:,} primes...")
    primes = load_primes(max_primes=MAX_PRIMES)
    print(f"Loaded {len(primes):,} primes")
    print(f"Range: {primes[0]:,} to {primes[-1]:,}")

    # Analyze
    print("\nPerforming E8-F4 analysis...")
    analyzer = E8F4PrimeAnalyzer()
    data = analyzer.analyze(primes)

    # Print statistics
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"  Total primes:        {len(primes):,}")
    print(f"  Total gaps:          {len(data['normalized_gaps']):,}")
    print(f"  F4 fraction:         {data['f4_fraction']*100:.2f}%")
    print(f"  F4-mapped gaps:      {np.sum(data['f4_mask']):,}")

    pa = data['phase_analysis']
    print(f"\n  Phase coherence:     {pa['phase_coherence']:.4f}")
    print(f"  Power entropy:       {pa['power_entropy']:.4f}")
    print(f"  Long/short ratio:    {pa['long_short_ratio']:.4f}")
    print(f"  Is phase-locked:     {pa['is_phase_locked']}")
    print(f"  Has Jordan struct:   {pa['has_jordan_structure']}")

    # Visualize
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    visualizer = E8F4Visualizer("/home/john/mynotes/HodgedeRham/spiral_outputs")

    print("\n1. E8 vs F4 comparison...")
    visualizer.plot_e8_vs_f4_comparison(data, dpi=1200)

    print("\n2. F4 crystalline grid...")
    visualizer.plot_f4_crystalline_grid(data, dpi=1200)

    print("\n3. F4 spectrum analysis...")
    visualizer.plot_f4_spectrum(data)

    print("\n4. Phase analysis summary...")
    visualizer.plot_phase_analysis_summary(data)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TOTAL TIME: {total_time:.2f}s")
    print(f"Output saved to: {visualizer.output_dir.resolve()}")
    print("=" * 70)

    # Final message
    if pa['is_phase_locked'] and pa['has_jordan_structure']:
        print("\n" + "=" * 70)
        print("✓ F4 SUB-HARMONIC DETECTED!")
        print("  The crystalline grid should show discrete vertices.")
        print("  These are the F4 idempotents - the fixed points of")
        print("  the Albert algebra that anchor the prime standing wave.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ F4 signal weak or not phase-locked.")
        print("  Try increasing the number of primes or adjusting τ.")
        print("=" * 70)


if __name__ == "__main__":
    main()
