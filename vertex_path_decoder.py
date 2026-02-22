#!/usr/bin/env python3
"""
vertex_path_decoder.py — Geodesic Decoding of the Crystalline Path

Implements the HPD (Hamiltonian Path Decoding) algorithm from
VERTEX-DECODE-HPD specs v12.0.

For each consecutive pair of crystalline vertices (V_i, V_{i+1}):
  1. Compute the geodesic angle between E8 root vectors
  2. Map angle → Base-18 alphabet character
  3. Apply topological tension threshold ("Great Silence" filter)

Reads vertex data from path_vertices.csv (output of path_decoder.c)
or generates its own via built-in sieve + E8 assignment.

Usage:
    python3 vertex_path_decoder.py [--csv PATH] [--vertices N] [--tension T]
                                   [--max-primes N] [--output-dir DIR]

Requirements: numpy (no SageMath needed — E8 roots built directly)
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter
from itertools import product as iproduct
from math import log, sqrt, ceil, cos, sin, pi, atan2

import numpy as np


# ================================================================
# E8 Root System: 240 roots in R^8
# Matches e8_common.h exactly (same ordering)
# ================================================================

def build_e8_roots():
    """Generate 240 E8 roots: 112 Type I (±e_i±e_j) + 128 Type II ((±½)^8 even neg)."""
    roots = []

    # Type I: 112 roots — ±e_i ± e_j for i < j
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = np.zeros(8)
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)

    # Type II: 128 roots — (±½)^8 with even number of minus signs
    for mask in range(256):
        v = np.zeros(8)
        neg_count = 0
        for d in range(8):
            if (mask >> d) & 1:
                v[d] = 0.5
            else:
                v[d] = -0.5
                neg_count += 1
        if neg_count % 2 == 0:
            roots.append(v)

    assert len(roots) == 240, f"BUG: got {len(roots)} E8 roots"
    return np.array(roots)


def e8_assign_root(norm_gap, min_norm=sqrt(2.0)):
    """Match e8_common.h: phase = (sqrt(max(g,0.01)) / sqrt(2)) % 1; idx = int(phase*240) % 240"""
    target_norm = sqrt(max(norm_gap, 0.01))
    phase = (target_norm / min_norm) % 1.0
    if phase < 0:
        phase += 1.0
    idx = int(phase * 240) % 240
    if idx < 0:
        idx += 240
    return idx


# ================================================================
# G2 Root System: 12 roots in R^2
# ================================================================

def build_g2_roots():
    """12 G2 roots: 6 short (norm 1) + 6 long (norm √3)."""
    roots = []
    # Short roots: angles 0°, 60°, 120°, 180°, 240°, 300°
    for k in range(6):
        angle = k * pi / 3.0
        roots.append(np.array([cos(angle), sin(angle)]))
    # Long roots: angles 30°, 90°, 150°, 210°, 270°, 330°
    for k in range(6):
        angle = (k * 60.0 + 30.0) * pi / 180.0
        roots.append(np.array([sqrt(3.0) * cos(angle), sqrt(3.0) * sin(angle)]))
    return np.array(roots)


def e8_to_g2_map(e8_roots, g2_roots):
    """Map each E8 root to nearest G2 root via cosine similarity of first 2 coords."""
    mapping = np.full(240, -1, dtype=int)
    is_member = np.zeros(240, dtype=bool)
    g2_norms = np.linalg.norm(g2_roots, axis=1)

    for ei in range(240):
        proj = e8_roots[ei, :2]
        pn = np.linalg.norm(proj)
        if pn < 0.01:
            continue
        proj_n = proj / pn
        best_sim = -1.0
        best_gi = 0
        for gi in range(12):
            gn = g2_norms[gi]
            if gn < 0.01:
                continue
            dot = np.dot(proj_n, g2_roots[gi] / gn)
            if abs(dot) > best_sim:
                best_sim = abs(dot)
                best_gi = gi
        mapping[ei] = best_gi
        is_member[ei] = best_sim >= 0.7

    return mapping, is_member


# ================================================================
# F4 Root System: 48 roots in R^4
# ================================================================

def build_f4_roots():
    """48 F4 roots: 24 long (±e_i±e_j) + 8 short (±e_i) + 16 short ((±½)^4)."""
    roots = []
    # 24 long: ±e_i ± e_j for i < j in 4D
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = np.zeros(4)
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)
    # 8 short: ±e_i
    for i in range(4):
        for s in (-1, 1):
            v = np.zeros(4)
            v[i] = s
            roots.append(v)
    # 8 short: (±½)^4 with even neg
    for mask in range(16):
        v = np.zeros(4)
        neg = 0
        for d in range(4):
            if (mask >> d) & 1:
                v[d] = 0.5
            else:
                v[d] = -0.5
                neg += 1
        if neg % 2 == 0:
            roots.append(v)
    # 8 short: (±½)^4 with odd neg
    for mask in range(16):
        v = np.zeros(4)
        neg = 0
        for d in range(4):
            if (mask >> d) & 1:
                v[d] = 0.5
            else:
                v[d] = -0.5
                neg += 1
        if neg % 2 == 1:
            roots.append(v)

    assert len(roots) == 48, f"BUG: got {len(roots)} F4 roots"
    return np.array(roots)


def e8_to_f4_map(e8_roots, f4_roots):
    """Map E8 → F4 via cosine similarity of first-4-coord projection."""
    f4_norms = np.linalg.norm(f4_roots, axis=1)
    mapping = np.full(240, -1, dtype=int)
    is_member = np.zeros(240, dtype=bool)

    for ei in range(240):
        proj = e8_roots[ei, :4].copy()
        pn = np.linalg.norm(proj)
        if pn < 0.01:
            # Try last 4
            proj = e8_roots[ei, 4:].copy()
            pn = np.linalg.norm(proj)
        if pn < 0.01:
            continue
        proj_n = proj / pn

        best_sim = -1.0
        best_fi = 0
        for fi in range(48):
            fn = f4_norms[fi]
            if fn < 0.01:
                continue
            dot = abs(np.dot(proj_n, f4_roots[fi] / fn))
            if dot > best_sim:
                best_sim = dot
                best_fi = fi
        mapping[ei] = best_fi

        # Quality check uses original first-4 projection (matching e8_common.h)
        orig = e8_roots[ei, :4]
        on = np.linalg.norm(orig)
        fn = f4_norms[best_fi]
        if on >= 0.01 and fn >= 0.01:
            quality = abs(np.dot(orig, f4_roots[best_fi])) / (on * fn)
            is_member[ei] = quality >= 0.7

    return mapping, is_member


# ================================================================
# Simple prime sieve (for standalone mode)
# ================================================================

def sieve_primes(n):
    """Sieve first n primes."""
    if n < 6:
        limit = 15
    else:
        ln_n = log(n)
        limit = int(n * (ln_n + log(ln_n))) + 1000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    return primes[:n]


# ================================================================
# Triplet coherence (matching e8_viz_v2.c)
# ================================================================

def compute_coherence(e8_roots, assignments):
    """κ[i] = |r[i-1] + r[i] + r[i+1]|² / 6"""
    n = len(assignments)
    coh = np.zeros(n)
    for i in range(1, n - 1):
        s = e8_roots[assignments[i-1]] + e8_roots[assignments[i]] + e8_roots[assignments[i+1]]
        coh[i] = np.dot(s, s) / 6.0
    return coh


# ================================================================
# Ulam spiral coordinates
# ================================================================

def ulam_coord(p):
    """Ulam spiral (x, y) for integer p. Matches e8_common.h."""
    if p <= 0:
        return 0, 0
    k = int(ceil((sqrt(p) - 1.0) / 2.0))
    t = 2 * k + 1
    m = t * t
    t -= 1  # t = 2k
    if p >= m - t:
        return k - (m - p), -k
    elif p >= m - 2 * t:
        return -k, -k + (m - t - p)
    elif p >= m - 3 * t:
        return -k + (m - 2 * t - p), k
    else:
        return k, k - (m - 3 * t - p)


# ================================================================
# HPD Algorithm
# ================================================================

# The 18-letter Active Alphabet (from spec)
ALPHABET_18 = "JOIREHQP GUAYVCNFBM"


def decode_vertex_path(vertices, e8_roots, tension_threshold=1.8):
    """
    Decode the Hamiltonian path between crystalline vertices.

    For each edge (V_i → V_{i+1}):
      1. Geodesic angle = arccos(⟨α_i, α_{i+1}⟩ / 2)
      2. Map to base-18 alphabet sector
      3. Apply tension threshold

    Returns: (decoded_string, edge_details)
    """
    decoded = []
    edge_details = []

    for i in range(len(vertices) - 1):
        v_curr = vertices[i]
        v_next = vertices[i + 1]

        r_curr = e8_roots[v_curr['e8_root']]
        r_next = e8_roots[v_next['e8_root']]

        # 1. Geodesic angle
        dot_prod = np.dot(r_curr, r_next)
        cos_theta = np.clip(dot_prod / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_theta)

        # 2. Base-18 mapping: divide [0, π] into 18 sectors
        char_idx = int((angle / np.pi) * 18) % 18
        char = ALPHABET_18[char_idx]

        # 3. Topological tension
        tension = np.linalg.norm(r_next - r_curr)

        # 4. Apply silence threshold
        if tension > tension_threshold:
            symbol = "."
        else:
            symbol = char

        decoded.append(symbol)
        edge_details.append({
            'edge': i,
            'from_prime': v_curr['prime'],
            'to_prime': v_next['prime'],
            'e8_from': v_curr['e8_root'],
            'e8_to': v_next['e8_root'],
            'dot_product': dot_prod,
            'angle_rad': angle,
            'angle_deg': np.degrees(angle),
            'sector': char_idx,
            'raw_char': char,
            'tension': tension,
            'symbol': symbol,
        })

    return ''.join(decoded), edge_details


# ================================================================
# Analysis functions
# ================================================================

def analyze_decoded(decoded_string, edge_details):
    """Print detailed analysis of the decoded path."""
    lines = []

    lines.append("=" * 72)
    lines.append("  HPD Analysis — Decoded Hamiltonian Path")
    lines.append("=" * 72)
    lines.append("")

    # Basic statistics
    n = len(decoded_string)
    n_silence = decoded_string.count('.')
    n_active = n - n_silence
    lines.append(f"Total edges:     {n}")
    lines.append(f"Active symbols:  {n_active} ({100*n_active/max(n,1):.1f}%)")
    lines.append(f"Silence dots:    {n_silence} ({100*n_silence/max(n,1):.1f}%)")
    lines.append("")

    # Full decoded string (chunked for readability)
    lines.append("Decoded string (chunks of 40):")
    for i in range(0, len(decoded_string), 40):
        chunk = decoded_string[i:i+40]
        lines.append(f"  [{i:4d}] {chunk}")
    lines.append("")

    # Character frequency
    active_chars = decoded_string.replace('.', '')
    freq = Counter(active_chars)
    lines.append("Character frequency (active only):")
    lines.append(f"  {'Char':<6} {'Count':<8} {'Freq%':<8} {'Alphabet':<10}")
    for ch, label in sorted([(c, ALPHABET_18[ALPHABET_18.index(c)] if c in ALPHABET_18 else '?')
                              for c in set(active_chars)],
                             key=lambda x: -freq[x[0]]):
        lines.append(f"  {ch:<6} {freq[ch]:<8} {100*freq[ch]/max(len(active_chars),1):<8.1f} sector {ALPHABET_18.index(ch) if ch in ALPHABET_18 else -1}")
    lines.append("")

    # Sector histogram
    sectors = [e['sector'] for e in edge_details]
    sector_counts = Counter(sectors)
    lines.append("Sector histogram (18 sectors of π/18 rad each):")
    lines.append(f"  {'Sector':<8} {'Char':<6} {'Angle range':<20} {'Count':<8}")
    for s in range(18):
        lo = s * 10
        hi = (s + 1) * 10
        ch = ALPHABET_18[s]
        cnt = sector_counts.get(s, 0)
        bar = '█' * cnt
        lines.append(f"  {s:<8} {ch:<6} [{lo:3d}°-{hi:3d}°)         {cnt:<8} {bar}")
    lines.append("")

    # Dot-product spectrum
    dots = [e['dot_product'] for e in edge_details]
    dot_counts = Counter(round(d, 1) for d in dots)
    lines.append("Dot-product spectrum ⟨α_i, α_{i+1}⟩:")
    for d in sorted(dot_counts.keys()):
        cnt = dot_counts[d]
        lines.append(f"  {d:+5.1f} : {cnt:4d} ({100*cnt/len(dots):.1f}%)")
    lines.append(f"  Mean: {np.mean(dots):+.4f}")
    lines.append("")

    # Tension distribution
    tensions = [e['tension'] for e in edge_details]
    lines.append("Tension statistics:")
    lines.append(f"  Mean: {np.mean(tensions):.4f}")
    lines.append(f"  Std:  {np.std(tensions):.4f}")
    lines.append(f"  Min:  {np.min(tensions):.4f}")
    lines.append(f"  Max:  {np.max(tensions):.4f}")
    lines.append("")

    # Tension histogram
    bins = np.linspace(0, 3.0, 13)
    hist, _ = np.histogram(tensions, bins=bins)
    lines.append("Tension histogram:")
    for b in range(len(hist)):
        if hist[b] > 0:
            bar = '█' * hist[b]
            lines.append(f"  [{bins[b]:.2f}-{bins[b+1]:.2f}): {hist[b]:4d} {bar}")
    lines.append("")

    # Look for special patterns from the spec
    lines.append("=" * 72)
    lines.append("  Pattern Search (from HPD spec)")
    lines.append("=" * 72)
    lines.append("")

    # PRIA sequence
    for pattern in ['PRIA', 'PRI', 'RIA', 'PR', 'RI', 'IA']:
        positions = []
        for i in range(len(active_chars) - len(pattern) + 1):
            if active_chars[i:i+len(pattern)] == pattern:
                positions.append(i)
        if positions:
            lines.append(f"  '{pattern}' found at active positions: {positions}")
    lines.append("")

    # J (Triality) at turning points
    j_positions = [i for i, c in enumerate(decoded_string) if c == 'J']
    if j_positions:
        lines.append(f"  'J' (Triality) at edges: {j_positions}")
        for jp in j_positions:
            e = edge_details[jp]
            lines.append(f"    edge {jp}: angle={e['angle_deg']:.1f}°, "
                        f"tension={e['tension']:.3f}, "
                        f"roots {e['e8_from']}→{e['e8_to']}")
    else:
        lines.append("  'J' (Triality) not found in decoded string")
    lines.append("")

    # Silence runs
    runs = []
    run_len = 0
    for c in decoded_string:
        if c == '.':
            run_len += 1
        else:
            if run_len > 0:
                runs.append(run_len)
            run_len = 0
    if run_len > 0:
        runs.append(run_len)
    if runs:
        lines.append(f"  Silence runs: {len(runs)} runs, lengths: {sorted(Counter(runs).items())}")
        lines.append(f"  Max silence run: {max(runs)}")
    else:
        lines.append("  No silence runs found")
    lines.append("")

    # Bigram analysis
    if len(active_chars) >= 2:
        bigrams = Counter(active_chars[i:i+2] for i in range(len(active_chars)-1))
        lines.append("Top 15 bigrams (active characters only):")
        for bg, cnt in bigrams.most_common(15):
            lines.append(f"  '{bg}': {cnt}")
        lines.append("")

    # Entropy
    if active_chars:
        probs = np.array([freq[c] for c in set(active_chars)], dtype=float)
        probs /= probs.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(18)
        lines.append(f"Shannon entropy: {entropy:.4f} bits (max for 18 symbols: {max_entropy:.4f})")
        lines.append(f"  Efficiency: {entropy/max_entropy:.1%}")
        lines.append("")

    return '\n'.join(lines)


# ================================================================
# Null model comparison
# ================================================================

def null_model_test(vertices, e8_roots, decoded_string, edge_details,
                    n_perms=1000, tension_threshold=1.8):
    """Compare decoded path against random vertex permutations."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"  Null Model ({n_perms} random permutations)")
    lines.append("=" * 72)
    lines.append("")

    # True path metrics
    true_active = decoded_string.replace('.', '')
    true_n_active = len(true_active)
    true_entropy = 0
    if true_active:
        freq = Counter(true_active)
        probs = np.array(list(freq.values()), dtype=float)
        probs /= probs.sum()
        true_entropy = -np.sum(probs * np.log2(probs))

    true_dots = [e['dot_product'] for e in edge_details]
    true_mean_dot = np.mean(true_dots)
    true_same_root = sum(1 for e in edge_details if e['e8_from'] == e['e8_to']) / len(edge_details)

    # Run permutations
    null_entropy = []
    null_active_frac = []
    null_mean_dot = []
    null_same_root = []

    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        perm = rng.permutation(len(vertices))
        perm_verts = [vertices[p] for p in perm]
        dec_str, det = decode_vertex_path(perm_verts, e8_roots, tension_threshold)

        active = dec_str.replace('.', '')
        null_active_frac.append(len(active) / max(len(dec_str), 1))

        if active:
            freq = Counter(active)
            probs = np.array(list(freq.values()), dtype=float)
            probs /= probs.sum()
            null_entropy.append(-np.sum(probs * np.log2(probs)))
        else:
            null_entropy.append(0)

        dots_p = [e['dot_product'] for e in det]
        null_mean_dot.append(np.mean(dots_p))
        null_same_root.append(sum(1 for e in det if e['e8_from'] == e['e8_to']) / len(det))

    null_entropy = np.array(null_entropy)
    null_active_frac = np.array(null_active_frac)
    null_mean_dot = np.array(null_mean_dot)
    null_same_root = np.array(null_same_root)

    def zscore(true_val, null_arr):
        m = np.mean(null_arr)
        s = np.std(null_arr)
        return (true_val - m) / s if s > 1e-15 else 0.0

    lines.append(f"{'Metric':<25} {'True':>10} {'Null mean':>10} {'Null std':>10} {'z-score':>10}")
    lines.append("-" * 72)

    z_ent = zscore(true_entropy, null_entropy)
    lines.append(f"{'Entropy':<25} {true_entropy:>10.4f} {np.mean(null_entropy):>10.4f} "
                 f"{np.std(null_entropy):>10.4f} {z_ent:>+10.2f}")

    z_act = zscore(true_n_active / max(len(decoded_string), 1), null_active_frac)
    lines.append(f"{'Active fraction':<25} {true_n_active/max(len(decoded_string),1):>10.4f} "
                 f"{np.mean(null_active_frac):>10.4f} {np.std(null_active_frac):>10.4f} {z_act:>+10.2f}")

    z_dot = zscore(true_mean_dot, null_mean_dot)
    lines.append(f"{'Mean dot product':<25} {true_mean_dot:>+10.4f} {np.mean(null_mean_dot):>+10.4f} "
                 f"{np.std(null_mean_dot):>10.4f} {z_dot:>+10.2f}")

    z_same = zscore(true_same_root, null_same_root)
    lines.append(f"{'Same-root fraction':<25} {true_same_root:>10.4f} {np.mean(null_same_root):>10.4f} "
                 f"{np.std(null_same_root):>10.4f} {z_same:>+10.2f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  |z| > 2.0 : significant at 95% level")
    lines.append("  |z| > 3.0 : highly significant at 99.7% level")
    lines.append("")

    return '\n'.join(lines)


# ================================================================
# Load vertices from CSV (path_decoder.c output)
# ================================================================

def load_vertices_csv(csv_path):
    """Load vertex data from path_vertices.csv."""
    vertices = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vertices.append({
                'seq': int(row['seq']),
                'prime_idx': int(row['prime_idx']),
                'prime': int(row['prime']),
                'coherence': float(row['coherence']),
                'e8_root': int(row['e8_root']),
                'e8_type': row['e8_type'],
                'f4_root': int(row['f4_root']),
                'is_f4': int(row['is_f4']),
                'g2_root': int(row['g2_root']),
                'is_g2': int(row['is_g2']),
                'jordan_trace': float(row['jordan_trace']),
                'ulam_x': int(row['ulam_x']),
                'ulam_y': int(row['ulam_y']),
                'norm_gap': float(row['norm_gap']),
            })
    return vertices


# ================================================================
# Generate vertices from scratch (standalone mode)
# ================================================================

def generate_vertices(max_primes, n_vertices, e8_roots):
    """Full pipeline: sieve → assign → coherence → top-K extraction."""
    print(f"  Sieving {max_primes:,} primes...")
    t0 = time.time()
    primes = sieve_primes(max_primes)
    n_primes = len(primes)
    print(f"  {n_primes:,} primes in {time.time()-t0:.2f}s "
          f"(range: {primes[0]} to {primes[-1]:,})")

    # E8 assignments + normalized gaps
    print("  Computing E8 assignments...")
    n_gaps = n_primes - 1
    assignments = np.zeros(n_gaps, dtype=int)
    norm_gaps = np.zeros(n_gaps)
    for i in range(n_gaps):
        gap = primes[i+1] - primes[i]
        log_p = max(log(primes[i]), 1.0)
        norm_gaps[i] = gap / log_p
        assignments[i] = e8_assign_root(norm_gaps[i])

    # Triplet coherence
    print("  Computing triplet coherence...")
    coherence = compute_coherence(e8_roots, assignments)

    # Top-K extraction (min-heap)
    print(f"  Extracting top-{n_vertices} vertices...")
    import heapq
    heap = []
    for i in range(1, n_gaps - 1):
        c = coherence[i]
        if c <= 0:
            continue
        if len(heap) < n_vertices:
            heapq.heappush(heap, (c, i))
        elif c > heap[0][0]:
            heapq.heapreplace(heap, (c, i))

    # Sort by gap index (ascending) → prime-index order
    selected = sorted(heap, key=lambda x: x[1])

    vertices = []
    for seq, (coh, gi) in enumerate(selected):
        pi = gi + 1
        e8r = assignments[gi]
        ux, uy = ulam_coord(primes[pi])

        # Determine type
        nonzero = sum(1 for d in range(8) if abs(e8_roots[e8r][d]) > 0.01)
        e8_type = 'I' if nonzero == 2 else 'II'

        jt = sum(e8_roots[e8r])

        vertices.append({
            'seq': seq,
            'prime_idx': pi,
            'prime': primes[pi],
            'coherence': coh,
            'e8_root': e8r,
            'e8_type': e8_type,
            'f4_root': -1,  # filled below if needed
            'is_f4': 0,
            'g2_root': -1,
            'is_g2': 0,
            'jordan_trace': jt,
            'ulam_x': ux,
            'ulam_y': uy,
            'norm_gap': norm_gaps[gi],
        })

    return vertices, primes


# ================================================================
# Weyl group analysis (lightweight, no SageMath)
# ================================================================

def weyl_transition_analysis(vertices, e8_roots, edge_details):
    """
    Analyze transitions through the lens of E8 Weyl reflections.
    Since we don't have SageMath, we characterize transitions by their
    geometric invariants: dot product, angle, and root difference vector.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("  Weyl Transition Analysis")
    lines.append("=" * 72)
    lines.append("")

    # Classify transitions by dot product (= cos of Weyl angle * 2)
    # E8 roots have ⟨α,β⟩ ∈ {-2,-1,0,1,2} for integer roots
    # and {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2} including half-integer
    dp_classes = {}
    for e in edge_details:
        dp = round(e['dot_product'], 1)
        if dp not in dp_classes:
            dp_classes[dp] = []
        dp_classes[dp].append(e)

    lines.append("Transition classes by dot product:")
    for dp in sorted(dp_classes.keys()):
        edges = dp_classes[dp]
        # Compute the difference vectors
        diff_norms = []
        for e in edges:
            r1 = e8_roots[e['e8_from']]
            r2 = e8_roots[e['e8_to']]
            diff_norms.append(np.linalg.norm(r2 - r1))

        lines.append(f"\n  ⟨α,β⟩ = {dp:+.1f}: {len(edges)} edges")
        lines.append(f"    |α-β| mean: {np.mean(diff_norms):.4f}, std: {np.std(diff_norms):.4f}")

        # Geometric interpretation
        if abs(dp - 2.0) < 0.01:
            lines.append("    → Identity (same root): Weyl element = 1")
        elif abs(dp - 1.0) < 0.01:
            lines.append("    → Adjacent roots: simple Weyl reflection s_α")
        elif abs(dp) < 0.01:
            lines.append("    → Orthogonal roots: commuting reflections s_α s_β")
        elif abs(dp + 1.0) < 0.01:
            lines.append("    → Opposite-adjacent: s_α s_β s_α (length 3)")
        elif abs(dp + 2.0) < 0.01:
            lines.append("    → Antipodal (α → -α): longest Weyl element w₀")

    lines.append("")

    # G2 subgroup constraint (from the spec: "Janik Selection Rule")
    # Check how many transitions stay within G2 sectors
    g2_consistent = 0
    for e in edge_details:
        v1 = vertices[e['edge']]
        v2 = vertices[e['edge'] + 1]
        if v1.get('is_g2') and v2.get('is_g2'):
            g2_consistent += 1

    lines.append(f"G2 subgroup transitions: {g2_consistent}/{len(edge_details)} "
                 f"({100*g2_consistent/max(len(edge_details),1):.1f}%)")
    lines.append("  (Janik Selection Rule: G2 hexagonal symmetry constrains")
    lines.append("   the Weyl search space from 696M to 12,096 elements)")
    lines.append("")

    return '\n'.join(lines)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="HPD: Geodesic Decoding of Crystalline Path")
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to path_vertices.csv (from path_decoder.c)')
    parser.add_argument('--max-primes', type=int, default=10000000,
                        help='Number of primes to sieve (standalone mode)')
    parser.add_argument('--vertices', type=int, default=500,
                        help='Number of crystalline vertices')
    parser.add_argument('--tension', type=float, default=1.8,
                        help='Tension threshold for Great Silence')
    parser.add_argument('--null-perms', type=int, default=500,
                        help='Number of null permutations')
    parser.add_argument('--output-dir', type=str, default='spiral_outputs',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 72)
    print("  VERTEX-DECODE-HPD — Geodesic Decoding of the Nilpotent Skeleton")
    print("  Version 12.0")
    print("=" * 72)
    print()

    # Build E8 roots
    print("Building E8 root system (240 roots in R^8)...")
    e8_roots = build_e8_roots()
    print(f"  Type I (integer): {sum(1 for r in e8_roots if sum(abs(r[d]) > 0.01 for d in range(8)) == 2)}")
    print(f"  Type II (half-int): {sum(1 for r in e8_roots if sum(abs(r[d]) > 0.01 for d in range(8)) == 8)}")
    print()

    # Load or generate vertices
    if args.csv and os.path.exists(args.csv):
        print(f"Loading vertices from {args.csv}...")
        vertices = load_vertices_csv(args.csv)
        print(f"  Loaded {len(vertices)} vertices")
    else:
        # Try default location
        default_csv = os.path.join(args.output_dir, 'path_vertices.csv')
        if args.csv is None and os.path.exists(default_csv):
            print(f"Loading vertices from {default_csv}...")
            vertices = load_vertices_csv(default_csv)
            print(f"  Loaded {len(vertices)} vertices")
        else:
            print(f"No CSV found. Generating vertices from scratch...")
            vertices, primes = generate_vertices(args.max_primes, args.vertices, e8_roots)
            print(f"  Generated {len(vertices)} vertices")
    print()

    # Limit to requested count
    if len(vertices) > args.vertices:
        vertices = vertices[:args.vertices]
        print(f"  Using first {args.vertices} vertices")
        print()

    # ---- HPD Decoding ----
    print("Decoding Hamiltonian path...")
    t0 = time.time()
    decoded_string, edge_details = decode_vertex_path(vertices, e8_roots, args.tension)
    print(f"  Decoded {len(decoded_string)} edges in {time.time()-t0:.4f}s")
    print()

    # Print the decoded string
    print("=" * 72)
    print("  DECODED HAMILTONIAN PATH")
    print("=" * 72)
    active = decoded_string.replace('.', '')
    print(f"  Full:   [{decoded_string[:80]}{'...' if len(decoded_string)>80 else ''}]")
    print(f"  Active: [{active[:80]}{'...' if len(active)>80 else ''}]")
    print(f"  Length: {len(decoded_string)} edges, {len(active)} active, "
          f"{decoded_string.count('.')} silence")
    print()

    # ---- Full Analysis ----
    analysis = analyze_decoded(decoded_string, edge_details)
    print(analysis)

    # ---- Weyl Transition Analysis ----
    weyl = weyl_transition_analysis(vertices, e8_roots, edge_details)
    print(weyl)

    # ---- Null Model ----
    print(f"Running null model ({args.null_perms} permutations)...")
    t0 = time.time()
    null_report = null_model_test(vertices, e8_roots, decoded_string,
                                  edge_details, args.null_perms, args.tension)
    print(f"  Done in {time.time()-t0:.2f}s")
    print()
    print(null_report)

    # ---- Write output ----
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'hpd_report.txt')
    with open(report_path, 'w') as f:
        f.write("VERTEX-DECODE-HPD — Full Report\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vertices: {len(vertices)}\n")
        f.write(f"Tension threshold: {args.tension}\n\n")
        f.write(f"DECODED STRING:\n{decoded_string}\n\n")
        f.write(f"ACTIVE CHARACTERS:\n{active}\n\n")
        f.write(analysis)
        f.write('\n')
        f.write(weyl)
        f.write('\n')
        f.write(null_report)
    print(f"Full report written to: {report_path}")

    # Edge details CSV
    edge_csv_path = os.path.join(args.output_dir, 'hpd_edges.csv')
    with open(edge_csv_path, 'w') as f:
        f.write("edge,from_prime,to_prime,e8_from,e8_to,dot_product,"
                "angle_deg,sector,raw_char,tension,symbol\n")
        for e in edge_details:
            f.write(f"{e['edge']},{e['from_prime']},{e['to_prime']},"
                    f"{e['e8_from']},{e['e8_to']},{e['dot_product']:.4f},"
                    f"{e['angle_deg']:.2f},{e['sector']},{e['raw_char']},"
                    f"{e['tension']:.4f},{e['symbol']}\n")
    print(f"Edge details written to: {edge_csv_path}")

    print()
    print("=" * 72)
    print("  HPD Complete")
    print("=" * 72)


if __name__ == '__main__':
    main()
