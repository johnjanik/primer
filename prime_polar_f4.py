#!/usr/bin/env python3
"""
Prime Polar Self-Map with F4 Slope Coloring

For each prime p, plot (r, θ) = (f(p), p) in polar coordinates.
Points are colored according to a slope derived from the F4 root assigned to
the normalized prime gap (p_{i+1} - p_i) / log(p_i).

Usage:
    python prime_polar_f4.py [N] [outfile]
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from f4_lattice import F4Lattice   # assume f4_lattice.py is in same directory


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
# Compute F4 coloring value for each prime
# ----------------------------------------------------------------------
def compute_f4_colors(primes):
    """
    Return array of color values (length = len(primes), first element = 0)
    Each value is the slope y/x from the first two coordinates of the F4 root
    assigned to the normalized gap.
    """
    if len(primes) < 2:
        return np.zeros(len(primes))

    f4 = F4Lattice()
    # Use min_norm = sqrt(2) (long root norm) for consistency with E8
    min_norm = np.sqrt(2)

    primes_float = np.array(primes, dtype=float)
    gaps = np.diff(primes_float)
    logp = np.log(primes_float[:-1])
    logp[logp < 1] = 1
    ng = gaps / logp

    # Assign each ng to an F4 root index (0..47)
    indices = []
    for g in ng:
        phase = (np.sqrt(max(g, 0.01)) / min_norm) % 1.0
        idx = int(phase * len(f4.roots_4d)) % len(f4.roots_4d)
        indices.append(idx)

    # Compute color value: slope from first two coordinates of the F4 root
    # slope = root[1] / root[0] (with handling for small denominator)
    slopes = []
    for idx in indices:
        root = f4.roots_4d[idx]
        x, y = root[0], root[1]
        if abs(x) < 1e-6:
            slope = np.sign(y) * 10.0
        else:
            slope = y / x
        slopes.append(np.clip(slope, -3, 3))

    # Build full array (first prime gets 0)
    colors = np.zeros(len(primes))
    colors[1:] = slopes
    return colors


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


def make_polar_subplot(ax, primes, color_vals, r_func, r_label, title_suffix,
                       cmap='coolwarm', vmin=-3, vmax=3):
    """Plot primes on given axes with F4 coloring."""
    p = np.array(primes, dtype=float)
    r = r_func(p)
    theta = p
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    col = color_vals[:len(primes)]
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

    print("Computing F4 colors from normalized gaps...")
    color_vals = compute_f4_colors(primes)

    # ---- Four-panel figure (like prime_polar_enhanced.py) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
    for ax in axes.flat:
        ax.set_facecolor('black')

    # Panel 1: r = p, all primes
    make_polar_subplot(axes[0, 0], primes, color_vals, lambda p: p, 'p',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Panel 2: r = sqrt(p), all primes
    make_polar_subplot(axes[0, 1], primes, color_vals, lambda p: np.sqrt(p), '√p',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Panel 3: r = sqrt(p), first 2000 primes (detail)
    n_inner = min(2000, n_primes)
    make_polar_subplot(axes[1, 0], primes[:n_inner], color_vals, lambda p: np.sqrt(p), '√p',
                       f'first {n_inner} (detail)', 'coolwarm', -3, 3)

    # Panel 4: r = log(p), all primes
    make_polar_subplot(axes[1, 1], primes, color_vals, lambda p: np.log(p), 'log(p)',
                       f'first {n_primes} primes', 'coolwarm', -3, 3)

    # Single colorbar for the whole figure
    add_colorbar(fig, axes[0,0], 'coolwarm', -3, 3,
                 'F4 root slope (y/x from first two coordinates)',
                 pos=[0.92, 0.15, 0.015, 0.7])

    fig.suptitle(f'Prime Polar Self-Map with F4 Slope Coloring — {n_primes} primes',
                 color='white', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    save_path = outfile or 'prime_polar_f4_panels.png'
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
    sc2 = ax2.scatter(x, y, s=20, c=color_vals, cmap='coolwarm',
                      vmin=-3, vmax=3, marker='.', linewidths=0, alpha=0.9)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Prime Polar Self-Map — (r, θ) = (√p, p)\n'
                  f'F4 Slope Coloring — {n_primes} primes',
                  color='white', fontsize=14, pad=20)

    # Colorbar
    cax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    cb2 = plt.colorbar(sc2, cax=cax2)
    cb2.set_label('F4 root slope', fontsize=11, color='white')
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')

    # Annotation
    ax2.text(0.5, -0.02,
             'Color = slope y/x from first two coordinates of F4 root assigned to\n'
             'normalized prime gap (p_{i+1}-p_i)/log(p_i).\n'
             'Red = steep positive, Blue = steep negative, White = near zero.',
             transform=ax2.transAxes, fontsize=9, ha='center', va='top',
             color='#aaaaaa')

    save2 = save_path.replace('_panels.png', '_sqrt.png')
    if save2 == save_path:
        save2 = 'prime_polar_f4_sqrt.png'
    plt.savefig(save2, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved standalone sqrt plot to {save2}")
    plt.close()

    # ---- (Optional) r = p standalone ----
    fig3, ax3 = plt.subplots(figsize=(14, 14), facecolor='black')
    ax3.set_facecolor('black')
    r_lin = p
    x_lin = r_lin * np.cos(p)
    y_lin = r_lin * np.sin(p)
    sc3 = ax3.scatter(x_lin, y_lin, s=0.4, c=color_vals, cmap='coolwarm',
                      vmin=-3, vmax=3, marker='.', linewidths=0, alpha=0.85)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title(f'Prime Polar Self-Map — (r, θ) = (p, p)\n'
                  f'F4 Slope Coloring — {n_primes} primes',
                  color='white', fontsize=14, pad=20)
    cax3 = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    cb3 = plt.colorbar(sc3, cax=cax3)
    cb3.set_label('F4 root slope', fontsize=11, color='white')
    cb3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='white')
    save3 = save_path.replace('_panels.png', '_linear.png')
    if save3 == save_path:
        save3 = 'prime_polar_f4_linear.png'
    plt.savefig(save3, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Saved standalone linear plot to {save3}")
    plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
