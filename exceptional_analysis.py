#!/usr/bin/env python3
"""
Exceptional Chain Analysis: G2 < F4 < E6 < E7 < E8 + S(16)

Multi-lattice orchestrator for prime gap analysis through the complete
exceptional Lie group chain plus the S(16) half-spinor sublattice.
Produces a 3x2 panel visualization showing E7, E6, F4, G2, and S16
filtered Ulam spirals with selectable color scales.

Color scales:
  jordan  — sum of F4 4D coords (range [-2,+2])
  trace8  — sum of 8D coords (range [-4,+4]), natural for E7/E6/S16
  trace2  — sum of 2D coords (range [-2.5,+2.5]), natural for G2
  norm    — root norm (range [1, sqrt(3)])
  auto    — natural trace for each lattice (default)

Usage:
  python3 exceptional_analysis.py --max-primes 2000000 [--color-scale auto]
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Matplotlib in non-interactive mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))

from e8_f4_prime_analysis import E8Lattice, load_primes, ulam_coords
from f4_lattice import F4Lattice
from e7_lattice import E7Lattice
from e6_lattice import E6Lattice
from g2_lattice import G2Lattice
from s16_lattice import S16Lattice


# ============================================================================
# Color scale definitions
# ============================================================================

COLOR_SCALES = {
    'jordan': {'label': 'Jordan trace (F4)', 'range': (-2.0, 2.0)},
    'trace8': {'label': 'Trace-8 (sum of 8D coords)', 'range': (-4.0, 4.0)},
    'trace2': {'label': 'Trace-2 (sum of 2D coords)', 'range': (-2.5, 2.5)},
    'norm':   {'label': 'Root norm', 'range': (1.0, 1.8)},
    'auto':   {'label': 'Auto (per-lattice)', 'range': None},
}

# Natural color scale for each lattice
AUTO_SCALES = {
    'E7': 'trace8',
    'E6': 'trace8',
    'F4': 'jordan',
    'G2': 'trace2',
    'S16': 'trace8',
}


def get_trace_value(lattice, lattice_name: str, root_idx: int,
                    color_scale: str) -> float:
    """Get the trace/color value for a root given the color scale."""
    if color_scale == 'auto':
        color_scale = AUTO_SCALES.get(lattice_name, 'jordan')

    if color_scale == 'jordan':
        if lattice_name == 'F4':
            root = lattice.get_f4_root(root_idx)
            return np.sum(root)
        elif lattice_name in ('E7', 'E6'):
            root = lattice.roots_8d[root_idx]
            return np.sum(root[:4])  # First 4 coords for Jordan
        elif lattice_name == 'S16':
            root = lattice.get_s16_root(root_idx)
            return np.sum(root[:4])
        else:  # G2
            root = lattice.get_g2_root(root_idx)
            return np.sum(root)
    elif color_scale == 'trace8':
        if lattice_name == 'F4':
            root = lattice.get_f4_root(root_idx)
            return np.sum(root)
        elif lattice_name in ('E7', 'E6'):
            return lattice.jordan_trace(root_idx)
        elif lattice_name == 'S16':
            return lattice.jordan_trace(root_idx)
        else:  # G2
            return lattice.jordan_trace(root_idx)
    elif color_scale == 'trace2':
        if lattice_name == 'G2':
            return lattice.jordan_trace(root_idx)
        elif lattice_name == 'F4':
            root = lattice.get_f4_root(root_idx)
            return root[0] + root[1]
        elif lattice_name == 'S16':
            root = lattice.get_s16_root(root_idx)
            return root[0] + root[1]
        else:  # E7, E6
            root = lattice.roots_8d[root_idx]
            return root[0] + root[1]
    elif color_scale == 'norm':
        return lattice.root_norm(root_idx)
    else:
        raise ValueError(f"Unknown color scale: {color_scale}")


def get_color_range(lattice_name: str, color_scale: str) -> Tuple[float, float]:
    """Get the [min, max] range for a color scale."""
    if color_scale == 'auto':
        color_scale = AUTO_SCALES.get(lattice_name, 'jordan')
    return COLOR_SCALES[color_scale]['range']


# ============================================================================
# Multi-lattice analyzer
# ============================================================================

class ExceptionalPrimeAnalyzer:
    """
    Analyze prime gaps through the full exceptional chain G2 < F4 < E6 < E7 < E8
    plus the S(16) half-spinor sublattice.
    """

    def __init__(self):
        print("Initializing E8 lattice...")
        self.e8 = E8Lattice()

        print("Initializing sublattices...")
        self.lattices = {
            'E7': E7Lattice(self.e8),
            'E6': E6Lattice(self.e8),
            'F4': F4Lattice(self.e8),
            'G2': G2Lattice(self.e8),
            'S16': S16Lattice(self.e8),
        }

        # Print summaries
        for name, lat in self.lattices.items():
            print(f"  {name}: {self._root_count(name)} roots")

    def _root_count(self, name: str) -> int:
        lat = self.lattices[name]
        if name == 'E7':
            return len(lat.roots_8d)
        elif name == 'E6':
            return len(lat.roots_8d)
        elif name == 'F4':
            return len(lat.roots_4d)
        elif name == 'G2':
            return len(lat.roots_2d)
        elif name == 'S16':
            return len(lat.roots_8d)
        return 0

    def _is_lattice_root(self, name: str, e8_idx: int) -> bool:
        lat = self.lattices[name]
        if name == 'E7':
            return lat.is_e7_root(e8_idx)
        elif name == 'E6':
            return lat.is_e6_root(e8_idx)
        elif name == 'F4':
            return lat.is_f4_root(e8_idx)
        elif name == 'G2':
            return lat.is_g2_root(e8_idx)
        elif name == 'S16':
            return lat.is_s16_root(e8_idx)
        return False

    def _project(self, name: str, e8_idx: int) -> Optional[int]:
        lat = self.lattices[name]
        if name == 'E7':
            return lat.project_e8_to_e7(e8_idx)
        elif name == 'E6':
            return lat.project_e8_to_e6(e8_idx)
        elif name == 'F4':
            return lat.project_e8_to_f4(e8_idx)
        elif name == 'G2':
            return lat.project_e8_to_g2(e8_idx)
        elif name == 'S16':
            return lat.project_e8_to_s16(e8_idx)
        return None

    def analyze(self, primes: np.ndarray, color_scale: str = 'auto') -> Dict:
        """
        Analyze prime gaps through all exceptional sublattices.

        Returns dict with per-lattice results.
        """
        n_primes = len(primes)
        gaps = np.diff(primes.astype(np.float64))
        log_p = np.log(primes[:-1].astype(np.float64))
        log_p[log_p < 1] = 1
        norm_gaps = gaps / log_p
        n_gaps = len(gaps)

        # E8 assignments
        print(f"Computing E8 assignments for {n_gaps:,} gaps...")
        e8_assignments = np.array([self.e8.assign_root(g) for g in norm_gaps])

        # Per-lattice analysis
        results = {}
        for name in ['E7', 'E6', 'F4', 'G2', 'S16']:
            print(f"  Filtering {name}...")
            lat = self.lattices[name]

            # Filter gaps by lattice membership
            is_member = np.zeros(n_gaps, dtype=bool)
            root_indices = np.full(n_gaps, -1, dtype=int)
            trace_values = np.full(n_gaps, np.nan)

            for i in range(n_gaps):
                e8_idx = e8_assignments[i]
                if self._is_lattice_root(name, e8_idx):
                    is_member[i] = True
                    sub_idx = self._project(name, e8_idx)
                    if sub_idx is not None:
                        root_indices[i] = sub_idx
                        trace_values[i] = get_trace_value(
                            lat, name, sub_idx, color_scale)

            n_mapped = np.sum(is_member)
            fraction = n_mapped / n_gaps

            results[name] = {
                'is_member': is_member,
                'root_indices': root_indices,
                'trace_values': trace_values,
                'n_mapped': int(n_mapped),
                'fraction': fraction,
                'n_roots': self._root_count(name),
            }
            print(f"    {name}: {n_mapped:,} / {n_gaps:,} = {100*fraction:.1f}%")

        # Verify containment chain
        print("\nContainment chain verification:")
        for sub, sup in [('G2', 'F4'), ('F4', 'E6'), ('E6', 'E7')]:
            # Not strict set containment for projected roots, but check overlap
            sub_set = set(np.where(results[sub]['is_member'])[0])
            sup_set = set(np.where(results[sup]['is_member'])[0])
            overlap = len(sub_set & sup_set)
            if len(sub_set) > 0:
                print(f"  {sub} gaps in {sup}: {overlap}/{len(sub_set)} "
                      f"({100*overlap/len(sub_set):.1f}%)")

        return {
            'primes': primes,
            'gaps': gaps,
            'norm_gaps': norm_gaps,
            'e8_assignments': e8_assignments,
            'lattice_results': results,
        }

    def visualize(self, analysis: Dict, color_scale: str = 'auto',
                  output_path: str = None):
        """
        Produce 3x2 panel visualization: E7, E6, F4, G2, S16 on Ulam spiral.
        Layout:  E7  | E6
                 F4  | G2
                 S16 | (summary)
        """
        primes = analysis['primes']
        results = analysis['lattice_results']

        print("\nComputing Ulam spiral coordinates...")
        coords = ulam_coords(primes)

        panel_order = ['E7', 'E6', 'F4', 'G2', 'S16']

        fig, axes = plt.subplots(3, 2, figsize=(20, 30), facecolor='black')
        fig.suptitle('Exceptional Chain: G2 < F4 < E6 < E7 < E8  +  S(16)',
                     color='white', fontsize=18, y=0.98)

        for ax_idx, name in enumerate(panel_order):
            row, col = divmod(ax_idx, 2)
            ax = axes[row][col]
            ax.set_facecolor('black')
            ax.set_aspect('equal')

            res = results[name]
            is_member = res['is_member']
            trace_vals = res['trace_values']

            # Get indices of member gaps (gap i → prime i+1)
            member_indices = np.where(is_member)[0]
            prime_indices = member_indices + 1

            if len(prime_indices) == 0:
                ax.set_title(f'{name} ({res["n_roots"]} roots) — no gaps mapped',
                             color='white', fontsize=14)
                continue

            x = coords[prime_indices, 0].astype(float)
            y = coords[prime_indices, 1].astype(float)
            c = trace_vals[member_indices]

            # Color range
            crange = get_color_range(name, color_scale)
            if crange is None:
                scale = AUTO_SCALES[name]
                crange = COLOR_SCALES[scale]['range']

            valid = ~np.isnan(c)
            if np.sum(valid) > 0:
                ax.scatter(
                    x[valid], y[valid], c=c[valid],
                    cmap='plasma', s=0.1, alpha=0.7,
                    vmin=crange[0], vmax=crange[1],
                    rasterized=True
                )

            ax.set_title(
                f'{name} ({res["n_roots"]} roots) — '
                f'{res["n_mapped"]:,} gaps ({100*res["fraction"]:.1f}%)',
                color='white', fontsize=14
            )
            ax.tick_params(colors='gray', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('gray')

        # Bottom-right panel: summary info
        ax_info = axes[2][1]
        ax_info.set_facecolor('black')
        ax_info.axis('off')
        summary_lines = ['Lattice Summary', '']
        for name in panel_order:
            res = results[name]
            summary_lines.append(
                f"{name:>4}: {res['n_roots']:>4} roots, "
                f"{res['n_mapped']:>8,} gaps ({100*res['fraction']:.1f}%)")
        summary_lines.append('')
        summary_lines.append(f"Total primes: {len(primes):,}")
        actual_scale = color_scale if color_scale != 'auto' else 'auto'
        summary_lines.append(f"Color scale: {actual_scale}")
        ax_info.text(0.05, 0.95, '\n'.join(summary_lines),
                     color='white', fontsize=13, family='monospace',
                     verticalalignment='top', transform=ax_info.transAxes)

        plt.tight_layout(rect=[0, 0.01, 1, 0.96])

        # Save
        output_dir = Path("/home/john/mynotes/HodgedeRham/spiral_outputs")
        output_dir.mkdir(exist_ok=True)
        if output_path is None:
            output_path = str(output_dir / "exceptional_chain.png")

        plt.savefig(output_path, dpi=300, facecolor='black',
                    bbox_inches='tight')
        plt.close()
        print(f"\nSaved visualization to {output_path}")

        return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Exceptional Chain Analysis: G2 < F4 < E6 < E7 < E8")
    parser.add_argument("--max-primes", type=int, default=2000000,
                        help="Maximum number of primes (default 2000000)")
    parser.add_argument("--color-scale", type=str, default="auto",
                        choices=list(COLOR_SCALES.keys()),
                        help="Color scale for visualization (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    args = parser.parse_args()

    print("=" * 60)
    print("EXCEPTIONAL CHAIN ANALYSIS + S(16)")
    print("G2 (12) < F4 (48) < E6 (72) < E7 (126) < E8 (240)")
    print("S(16) = 128 half-spinor roots (Type II E8)")
    print("=" * 60)

    start_time = time.time()

    # Load primes
    print(f"\nLoading up to {args.max_primes:,} primes...")
    primes = load_primes(max_primes=args.max_primes)
    print(f"Loaded {len(primes):,} primes (range {primes[0]} to {primes[-1]})")

    # Initialize analyzer
    analyzer = ExceptionalPrimeAnalyzer()

    # Run analysis
    print("\n--- Analysis ---")
    analysis = analyzer.analyze(primes, color_scale=args.color_scale)

    # Produce visualization
    print("\n--- Visualization ---")
    out_path = analyzer.visualize(analysis, color_scale=args.color_scale,
                                  output_path=args.output)

    # Summary table
    print("\n" + "=" * 60)
    print("EXCEPTIONAL CHAIN SUMMARY")
    print("=" * 60)
    print(f"{'Lattice':<8} {'Roots':>6} {'Mapped':>10} {'Fraction':>10}")
    print("-" * 40)
    for name in ['E7', 'E6', 'F4', 'G2', 'S16']:
        res = analysis['lattice_results'][name]
        print(f"{name:<8} {res['n_roots']:>6} {res['n_mapped']:>10,} "
              f"{100*res['fraction']:>9.1f}%")
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
