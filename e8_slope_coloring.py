"""
E8 Projection Slope Coloring - Standalone High-Resolution Image
Primes in the Ulam spiral colored by their E8 root projection slope
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for large images

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import re
import time

# =============================================================================
# E8 Lattice
# =============================================================================

class E8Lattice:
    """The E8 root lattice with 240 roots in R^8"""

    def __init__(self):
        self.roots = self._generate_roots()
        self.min_norm = np.sqrt(2)
        # Precompute projection slopes
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
# Ulam Spiral Coordinates (Vectorized for speed)
# =============================================================================

def ulam_coords_vectorized(primes: np.ndarray) -> np.ndarray:
    """Compute Ulam coordinates for array of primes (vectorized)"""
    n = primes.astype(np.float64)
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
# Load Primes
# =============================================================================

def load_primes(base_dir: str = "/home/john/mynotes/HodgedeRham", max_primes: int = 1000000) -> np.ndarray:
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
# Main Visualization Function
# =============================================================================

def generate_slope_visualization(max_primes: int, dpi: int = 300,
                                  figsize: tuple = (20, 20)):
    """Generate E8 projection slope visualization"""
    print(f"\n{'='*60}")
    print(f"E8 Projection Slope: {max_primes:,} primes at {dpi} DPI")
    print(f"{'='*60}")

    start_time = time.time()

    # Load primes
    print(f"Loading primes...")
    load_start = time.time()
    primes = load_primes(max_primes=max_primes)
    print(f"  Loaded {len(primes):,} primes in {time.time()-load_start:.2f}s")
    print(f"  Range: 2 to {primes[-1]:,}")

    # E8 analysis
    print("Computing E8 root assignments...")
    e8_start = time.time()
    e8 = E8Lattice()

    # Compute gaps and normalized gaps
    gaps = np.diff(primes.astype(np.float64))
    log_primes = np.log(primes[:-1].astype(np.float64))
    log_primes[log_primes < 1] = 1
    normalized_gaps = gaps / log_primes

    # Assign roots (vectorized)
    root_assignments = np.array([e8.assign_root(g) for g in normalized_gaps])

    # Get slope for each prime's root
    root_slopes = e8.projected_slopes[root_assignments]
    print(f"  E8 analysis completed in {time.time()-e8_start:.2f}s")

    # Compute Ulam coordinates
    print("Computing Ulam spiral coordinates...")
    coord_start = time.time()
    coords = ulam_coords_vectorized(primes)
    print(f"  Coordinates computed in {time.time()-coord_start:.2f}s")

    # Create colors based on slope
    clipped_slopes = np.clip(root_slopes, -3, 3)

    # Generate dark background version
    print(f"Rendering at {dpi} DPI ({figsize[0]}x{figsize[1]} inches)...")
    render_start = time.time()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    ax.set_facecolor('black')

    # Adjust point size based on number of primes
    if max_primes >= 10000000:
        point_size = 0.1
        alpha = 0.6
    elif max_primes >= 1000000:
        point_size = 0.3
        alpha = 0.7
    else:
        point_size = 0.5
        alpha = 0.8

    scatter = ax.scatter(coords[1:, 0], coords[1:, 1],
                        c=clipped_slopes, cmap='coolwarm',
                        s=point_size, alpha=alpha, vmin=-3, vmax=3)

    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=14, color='white')
    ax.set_ylabel('y', fontsize=14, color='white')

    title = f'Primes Colored by E8 Projection Slope\n{len(primes):,} primes in Ulam Spiral'
    ax.set_title(title, fontsize=18, fontweight='bold', color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('E8 Projection Slope (y/x)', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()

    # Save
    suffix = f"{max_primes // 1000000}M" if max_primes >= 1000000 else f"{max_primes // 1000}k"
    output_path = Path(f"/home/john/mynotes/HodgedeRham/spiral_outputs/e8_projection_slope_{suffix}_{dpi}dpi_dark.png")
    output_path.parent.mkdir(exist_ok=True)

    print(f"Saving to {output_path.name}...")
    save_start = time.time()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='black')
    print(f"  Saved in {time.time()-save_start:.2f}s")

    plt.close()

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Output: {output_path.resolve()}")

    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    print("E8 Projection Slope Visualization - High Resolution")
    print("=" * 60)

    # Generate 1 million primes at 1200 DPI
    generate_slope_visualization(max_primes=10000, dpi=1200, figsize=(24, 24))

    # Generate 10 million primes at 1200 DPI (reduced size for memory)
    #generate_slope_visualization(max_primes=10000000, dpi=1200, figsize=(24, 24))

    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
