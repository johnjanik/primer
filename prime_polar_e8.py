#!/usr/bin/env python3
"""
Prime Polar Self-Map with E₈ Slope Coloring

For each prime p, plot (r, θ) = (f(p), p) in polar coordinates.
Points are colored according to the slope of the E₈ root assigned to the
normalized prime gap (p_{i+1} - p_i) / log(p_i).

Usage:
    python prime_polar_e8.py [N] [outfile]
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ----------------------------------------------------------------------
# E₈ lattice – exactly as in polar_slope_e8.py
# ----------------------------------------------------------------------
class E8Lattice:
    def __init__(self):
        self.roots = self._generate_roots()
        self.min_norm = np.sqrt(2)
        self.projected_slopes = self._compute_projection_slopes()

    def _generate_roots(self):
        roots = []
        # type ±e_i ± e_j (i<j)
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        r = np.zeros(8)
                        r[i] = s1
                        r[j] = s2
                        roots.append(r)
        # type (±1/2, ..., ±1/2) with even number of minus signs
        for mask in range(256):
            signs = [1 if (mask >> i) & 1 else -1 for i in range(8)]
            if sum(1 for s in signs if s == -1) % 2 == 0:
                roots.append(np.array([s * 0.5 for s in signs]))
        return np.array(roots)

    def _compute_projection_slopes(self):
        slopes = []
        for root in self.roots:
            x = np.sum(root[:4])   # projection onto first 4 coordinates
            y = np.sum(root[4:])   # projection onto last 4 coordinates
            slopes.append(y / x if abs(x) > 0.01 else np.sign(y) * 10)
        return np.array(slopes)

    def assign_root(self, ng):
        """Return index of the root that corresponds to normalized gap ng."""
        phase = (np.sqrt(max(ng, 0.01)) / self.min_norm) % 1.0
        return int(phase * len(self.roots)) % len(self.roots)


# ----------------------------------------------------------------------
# Prime sieve (first N primes)
# ----------------------------------------------------------------------
def sieve_first_n_primes(n):
    if n <= 0:
        return []
    # Estimate upper bound using Chebyshev's approximation
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 100
    while True:
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i, v in enumerate(is_prime) if v]
        if len(primes) >= n:
            return primes[:n]
        limit = int(limit * 1.5) + 100


# ----------------------------------------------------------------------
# Compute E₈ slope for each prime
# ----------------------------------------------------------------------
def compute_e8_slopes(primes):
    """Return array of slopes (length = len(primes)), first element = 0."""
    if len(primes) < 2:
        return np.zeros(len(primes))

    e8 = E8Lattice()
    primes_float = np.array(primes, dtype=float)          # ensure float
    gaps = np.diff(primes_float)                          # now works
    logp = np.log(primes_float[:-1])
    logp[logp < 1] = 1
    ng = gaps / logp

    slopes = np.zeros(len(primes))
    indices = np.array([e8.assign_root(g) for g in ng])
    rs = e8.projected_slopes[indices]
    slopes[1:] = np.clip(rs, -3, 3)
    return slopes

# ----------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------
def add_colorbar(fig, ax, cmap, vmin, vmax, label, pos=[0.92, 0.15, 0.015, 0.7]):
    """Add a vertical colorbar to the figure."""
    cax = fig.add_axes(pos)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=cmap),
                      cax=cax, orientation='vertical')
    cb.set_label(label, fontsize=11, color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    return cb


def make_polar_subplot(ax, primes, slopes, r_func, r_label, title_suffix, cmap='coolwarm', vmin=-3, vmax=3):
    """Plot primes on given axes with E₈ coloring."""
    p = np.array(primes, dtype=float)
    r = r_func(p)
    theta = p
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Slopes array length must match primes
    col = slopes[:len(primes)]
    sc = ax.scatter(x, y, s=0.5, c=col, cmap=cmap, vmin=vmin, vmax=vmax,
                    marker='.', linewidths=0, alpha=0.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'(r, θ) = ({r_label}, p)  — {title_suffix}',
                 color='white', fontsize=11, pad=10)
    return sc


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    n_primes = 10000
    outfile = None

    if len(sys.argv) >= 2:
        try:
            n_primes = int(sys.argv[1])
        except ValueError:
            print("Error: first argument must be an integer.")
            return
    if len(sys.argv) >= 3:
        outfile = sys.argv[2]

    print(f"Generating first {n_primes} primes...")
    primes = sieve_first_n_primes(n_primes)
    print(f"  -> largest prime = {primes[-1]}")

    print("Computing E₈ slopes from normalized gaps...")
    slopes = compute_e8_slopes(primes)

    # ---- Four-panel figure (like prime_polar_enhanced.py) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
    for ax in axes.flat:
        ax.set_facecolor('black')

    # Panel 1: r = p, all primes
    make_polar_subplot(axes[0, 0], primes, slopes, lambda p: p, 'p',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Panel 2: r = sqrt(p), all primes
    make_polar_subplot(axes[0, 1], primes, slopes, lambda p: np.sqrt(p), '√p',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Panel 3: r = sqrt(p), first 2000 primes (detail)
    n_inner = min(2000, n_primes)
    make_polar_subplot(axes[1, 0], primes[:n_inner], slopes, lambda p: np.sqrt(p), '√p',
                       f'first {n_inner} (detail)', 'coolwarm', -3, 3)

    # Panel 4: r = log(p), all primes
    make_polar_subplot(axes[1, 1], primes, slopes, lambda p: np.log(p), 'log(p)',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Single colorbar for the whole figure
    add_colorbar(fig, axes[0,0], 'coolwarm', -3, 3,
                 'E₈ projection slope (y/x)',
                 pos=[0.92, 0.15, 0.015, 0.7])

    fig.suptitle(f'Prime Polar Self-Map with E₈ Slope Coloring — {n_primes} primes',
                 color='white', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    save_path = outfile or 'prime_polar_e8_panels.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved four-panel plot to {save_path}")
    plt.close()

    # ---- Standalone sqrt version (large) ----
    fig2, ax2 = plt.subplots(figsize=(14, 14), facecolor='black')
    ax2.set_facecolor('black')
    p = np.array(primes, dtype=float)
    r = np.sqrt(p)
    theta = p
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    sc2 = ax2.scatter(x, y, s=20, c=slopes, cmap='coolwarm',
                      vmin=-3, vmax=3, marker='.', linewidths=0, alpha=0.9)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Prime Polar Self-Map — (r, θ) = (√p, p)\n'
                  f'E₈ Slope Coloring — {n_primes} primes',
                  color='white', fontsize=14, pad=20)

    # Colorbar
    cax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    cb2 = plt.colorbar(sc2, cax=cax2)
    cb2.set_label('E₈ projection slope', fontsize=11, color='white')
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')

    # Annotation
    ax2.text(0.5, -0.02,
             'Color = slope of E₈ root assigned to (p_{i+1}-p_i) / log(p_i).\n'
             'Red = steep positive, Blue = steep negative, White = near zero.',
             transform=ax2.transAxes, fontsize=9, ha='center', va='top',
             color='#aaaaaa')

    save2 = save_path.replace('_panels.png', '_sqrt.png')
    if save2 == save_path:
        save2 = 'prime_polar_e8_sqrt.png'
    plt.savefig(save2, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved standalone sqrt plot to {save2}")
    plt.close()

    # ---- (Optional) r = p standalone ----
    fig3, ax3 = plt.subplots(figsize=(14, 14), facecolor='black')
    ax3.set_facecolor('black')
    r_lin = p
    x_lin = r_lin * np.cos(p)
    y_lin = r_lin * np.sin(p)
    sc3 = ax3.scatter(x_lin, y_lin, s=0.4, c=slopes, cmap='coolwarm',
                      vmin=-3, vmax=3, marker='.', linewidths=0, alpha=0.85)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title(f'Prime Polar Self-Map — (r, θ) = (p, p)\n'
                  f'E₈ Slope Coloring — {n_primes} primes',
                  color='white', fontsize=14, pad=20)
    cax3 = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    cb3 = plt.colorbar(sc3, cax=cax3)
    cb3.set_label('E₈ projection slope', fontsize=11, color='white')
    cb3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
    save3 = save_path.replace('_panels.png', '_linear.png')
    if save3 == save_path:
        save3 = 'prime_polar_e8_linear.png'
    plt.savefig(save3, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved standalone linear plot to {save3}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
