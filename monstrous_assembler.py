#!/usr/bin/env python3
"""
monstrous_assembler.py — Run-Length Assembler for the Crystalline Path

Takes the raw Hamiltonian path (from path_decoder.c / vertex_path_decoder.py)
and compresses it into (Root, Duration, Operator) triplets using the
Phase-Length Modulation scheme:

    Noun     = E8 root index (the logical context)
    Verb     = run length (number of consecutive same-root edges)
    Operator = turning angle to next root (Triality/Duality/Identity)

Reads: path_vertices.csv, path_edges.csv (from path_decoder.c)
Writes: assembled_runs.csv, assembler_report.txt

Usage:
    python3 monstrous_assembler.py [--input-dir DIR] [--output-dir DIR]
                                   [--null-perms N]
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from math import log, sqrt, pi, atan2, acos, degrees

import numpy as np


# ================================================================
# E8 Root System (matching e8_common.h)
# ================================================================

def build_e8_roots():
    """240 E8 roots: 112 Type I + 128 Type II."""
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = np.zeros(8)
                    v[i] = s1
                    v[j] = s2
                    roots.append(v)
    for mask in range(256):
        v = np.zeros(8)
        neg = 0
        for d in range(8):
            if (mask >> d) & 1:
                v[d] = 0.5
            else:
                v[d] = -0.5
                neg += 1
        if neg % 2 == 0:
            roots.append(v)
    assert len(roots) == 240
    return np.array(roots)


def root_type_str(e8_roots, idx):
    """Return 'I' or 'II'."""
    nonzero = sum(1 for d in range(8) if abs(e8_roots[idx][d]) > 0.01)
    return 'I' if nonzero == 2 else 'II'


def root_coord_str(e8_roots, idx):
    """Human-readable coordinate string."""
    parts = []
    for d in range(8):
        v = e8_roots[idx][d]
        if abs(v - 0.5) < 0.01:
            parts.append('+½')
        elif abs(v + 0.5) < 0.01:
            parts.append('-½')
        elif abs(v - 1.0) < 0.01:
            parts.append('+1')
        elif abs(v + 1.0) < 0.01:
            parts.append('-1')
        else:
            parts.append(' 0')
    return '(' + ','.join(parts) + ')'


# ================================================================
# Operator classification
# ================================================================

# Weyl transition operators based on dot product and geodesic angle
OPERATORS = {
    '+2.0': ('1',   'Identity',        '='),   # same root, Weyl = 1
    '+1.0': ('s',   'Simple reflection','→'),   # adjacent, single Weyl reflection
    '+0.5': ('s½',  'Half-step',       '⇀'),   # half-integer adjacency
    ' 0.0': ('ss',  'Orthogonal',      '⊥'),   # commuting reflections
    '-0.5': ('ss½', 'Anti-half-step',  '⇁'),
    '-1.0': ('sss', 'Opposite-adj',    '↻'),   # length-3 Weyl word
    '-1.5': ('w¾',  'Deep rotation',   '⟳'),
    '-2.0': ('w₀',  'Antipodal',       '⊖'),   # longest Weyl element
}


def classify_operator(dot_product):
    """Classify a root transition by its dot product."""
    rounded = round(dot_product * 2) / 2  # round to nearest 0.5
    key = f'{rounded:+.1f}'
    if key in OPERATORS:
        return OPERATORS[key]
    return ('?', f'dp={dot_product:.2f}', '?')


def turning_class(angle_deg):
    """Classify turning angle into geometric operations."""
    a = abs(angle_deg)
    if a < 15:
        return 'straight'
    elif 55 < a < 65:
        return 'hexagonal (60°)'
    elif 85 < a < 95:
        return 'quadrant (90°)'
    elif 115 < a < 125:
        return 'triality (120°)'
    elif 145 < a < 155:
        return 'near-reversal (150°)'
    elif a > 170:
        return 'duality (180°)'
    else:
        return f'oblique ({angle_deg:+.0f}°)'


# ================================================================
# Run-Length Encoding
# ================================================================

def extract_runs(vertices, edges):
    """
    Group consecutive same-root edges into runs.

    Returns list of dicts:
        root:       E8 root index
        root_type:  'I' or 'II'
        length:     number of consecutive edges with this root
        start_seq:  first vertex sequence number in this run
        end_seq:    last vertex sequence number
        start_prime: first prime in the run
        end_prime:   last prime in the run
        exit_dot:   dot product to next run's root (-999 if last run)
        exit_angle: turning angle at exit (-999 if last)
        exit_op:    operator symbol for the exit transition
    """
    if not vertices:
        return []

    runs = []
    cur_root = vertices[0]['e8_root']
    cur_start = 0
    cur_length = 0

    for i, edge in enumerate(edges):
        if edge['e8_from'] == cur_root:
            cur_length += 1
            if edge['e8_to'] != cur_root:
                # Run ends here
                run = {
                    'root': cur_root,
                    'root_type': vertices[cur_start]['e8_type'],
                    'length': cur_length,
                    'start_seq': cur_start,
                    'end_seq': i,
                    'start_prime': vertices[cur_start]['prime'],
                    'end_prime': vertices[i]['prime'],
                    'exit_dot': edge['dot_product'],
                    'exit_turn': edge['turning_angle'],
                    'exit_op': classify_operator(edge['dot_product']),
                }
                runs.append(run)
                cur_root = edge['e8_to']
                cur_start = i + 1
                cur_length = 0
        else:
            # Root changed on the 'from' side — this means a singleton
            # appeared or the tracking fell out of sync
            if cur_length > 0:
                run = {
                    'root': cur_root,
                    'root_type': vertices[cur_start]['e8_type'],
                    'length': cur_length,
                    'start_seq': cur_start,
                    'end_seq': i - 1,
                    'start_prime': vertices[cur_start]['prime'],
                    'end_prime': vertices[max(0, i-1)]['prime'],
                    'exit_dot': edges[i-1]['dot_product'] if i > 0 else 0,
                    'exit_turn': edges[i-1]['turning_angle'] if i > 0 else 0,
                    'exit_op': classify_operator(edges[i-1]['dot_product'] if i > 0 else 0),
                }
                runs.append(run)

            # Start singleton for edge['e8_from']
            cur_root = edge['e8_from']
            cur_start = i
            cur_length = 0

            if edge['e8_to'] == edge['e8_from']:
                cur_length = 1
            else:
                # from != to, both are singletons
                run = {
                    'root': edge['e8_from'],
                    'root_type': vertices[i]['e8_type'],
                    'length': 1,
                    'start_seq': i,
                    'end_seq': i,
                    'start_prime': vertices[i]['prime'],
                    'end_prime': vertices[i]['prime'],
                    'exit_dot': edge['dot_product'],
                    'exit_turn': edge['turning_angle'],
                    'exit_op': classify_operator(edge['dot_product']),
                }
                runs.append(run)
                cur_root = edge['e8_to']
                cur_start = i + 1
                cur_length = 0

    # Final run
    if cur_length > 0 or cur_start <= len(edges):
        last_idx = min(cur_start + max(cur_length, 0), len(vertices) - 1)
        run = {
            'root': cur_root,
            'root_type': vertices[min(cur_start, len(vertices)-1)]['e8_type'],
            'length': max(cur_length, 1),
            'start_seq': cur_start,
            'end_seq': last_idx,
            'start_prime': vertices[min(cur_start, len(vertices)-1)]['prime'],
            'end_prime': vertices[last_idx]['prime'],
            'exit_dot': -999,
            'exit_turn': -999,
            'exit_op': ('END', 'Terminal', '■'),
        }
        runs.append(run)

    return runs


# ================================================================
# Symbolic assembly
# ================================================================

# Root families (grouped by coordinate structure)
ROOT_FAMILIES = {
    'Zeta-axis': range(108, 112),      # ±e₆±e₇ — the Information Axis
    'Deep-half': range(112, 120),       # first half-integer block
    'Mid-half':  range(120, 144),       # second half-integer block
    'Cross-half': range(144, 176),      # third half-integer block
    'Outer-half': range(176, 240),      # remaining half-integer
    'Inner-int':  range(0, 108),        # other integer roots
}


def root_family(root_idx):
    """Return the family name for a root."""
    for name, rng in ROOT_FAMILIES.items():
        if root_idx in rng:
            return name
    return 'Unknown'


def assemble_sentence(runs, e8_roots):
    """
    Convert runs into symbolic sentence using Phase-Length Modulation.

    Encoding scheme:
        Short run (1-2):   lowercase  — "whisper" (transient state)
        Medium run (3-8):  UPPERCASE  — "statement" (stable assertion)
        Long run (9+):     UPPERCASE + repeat marker — "shout" (held tone)

    The exit operator determines the punctuation:
        Identity (dp=+2):    no separator (continuation)
        Simple (dp=+1):      comma (gentle turn)
        Orthogonal (dp=0):   semicolon (right angle)
        Opposite (dp=-1):    period (reversal)
        Antipodal (dp=-2):   exclamation (full flip)
    """
    tokens = []

    for run in runs:
        root = run['root']
        length = run['length']
        family = root_family(root)

        # The "noun": root identity compressed to a letter
        # Use the family initial + root mod 26
        noun_idx = root % 26
        noun_char = chr(ord('A') + noun_idx)

        # The "magnitude": run length encoding
        if length <= 2:
            noun_char = noun_char.lower()
            magnitude = ''
        elif length <= 8:
            magnitude = ''
        else:
            magnitude = f'×{length}'

        # Build token
        token = noun_char + magnitude

        # The "operator": exit transition
        if run['exit_dot'] == -999:
            sep = '■'  # terminal
        else:
            dp = round(run['exit_dot'] * 2) / 2
            if dp >= 2.0:
                sep = ''       # identity — continuation
            elif dp >= 0.5:
                sep = ','      # gentle
            elif dp >= -0.5:
                sep = ';'      # orthogonal
            elif dp >= -1.5:
                sep = '.'      # reversal
            else:
                sep = '!'      # antipodal
        tokens.append(token + sep)

    return ''.join(tokens)


# ================================================================
# Analysis
# ================================================================

def analyze_runs(runs, e8_roots, n_null=500):
    """Produce detailed analysis of the run structure."""
    lines = []

    lines.append('=' * 72)
    lines.append('  Monstrous Assembler — Run-Length Analysis')
    lines.append('=' * 72)
    lines.append('')

    n_runs = len(runs)
    total_edges = sum(r['length'] for r in runs)
    lengths = [r['length'] for r in runs]

    lines.append(f'Total runs:    {n_runs}')
    lines.append(f'Total edges:   {total_edges}')
    lines.append(f'Compression:   {total_edges} edges → {n_runs} runs '
                 f'({n_runs/max(total_edges,1)*100:.1f}%)')
    lines.append('')

    # ---- Run length distribution ----
    lines.append('— Run Length Distribution —')
    len_counts = Counter(lengths)
    lines.append(f'  {"Length":<8} {"Count":<8} {"Frac%":<8} {"Histogram"}')
    for l in sorted(len_counts.keys()):
        c = len_counts[l]
        bar = '█' * c
        lines.append(f'  {l:<8} {c:<8} {100*c/n_runs:<8.1f} {bar}')
    lines.append(f'\n  Mean run length: {np.mean(lengths):.2f}')
    lines.append(f'  Median:          {np.median(lengths):.1f}')
    lines.append(f'  Max:             {max(lengths)}')
    lines.append(f'  Std:             {np.std(lengths):.2f}')
    lines.append('')

    # ---- Root distribution across runs ----
    lines.append('— Root Distribution (by run) —')
    root_counts = Counter(r['root'] for r in runs)
    root_weighted = defaultdict(int)
    for r in runs:
        root_weighted[r['root']] += r['length']

    lines.append(f'  Distinct roots: {len(root_counts)}')
    lines.append(f'\n  {"Root":<6} {"Runs":<6} {"Edges":<7} {"Type":<5} '
                 f'{"Family":<12} {"Coordinates"}')
    for root, cnt in root_counts.most_common(25):
        w = root_weighted[root]
        lines.append(f'  {root:<6} {cnt:<6} {w:<7} '
                     f'{root_type_str(e8_roots, root):<5} '
                     f'{root_family(root):<12} '
                     f'{root_coord_str(e8_roots, root)}')
    lines.append('')

    # ---- Exit operator distribution ----
    lines.append('— Exit Operator Distribution —')
    op_counts = Counter()
    dp_values = []
    turn_values = []
    for r in runs:
        if r['exit_dot'] != -999:
            sym, name, glyph = r['exit_op']
            op_counts[name] += 1
            dp_values.append(r['exit_dot'])
        if r['exit_turn'] != -999:
            turn_values.append(r['exit_turn'])

    lines.append(f'  {"Operator":<20} {"Count":<8} {"Frac%":<8}')
    for name, cnt in op_counts.most_common():
        lines.append(f'  {name:<20} {cnt:<8} {100*cnt/max(sum(op_counts.values()),1):<8.1f}')
    lines.append('')

    # Dot product at transitions
    if dp_values:
        dp_counts = Counter(round(d, 1) for d in dp_values)
        lines.append('  Dot product at run boundaries:')
        for dp in sorted(dp_counts.keys()):
            cnt = dp_counts[dp]
            lines.append(f'    ⟨α,β⟩ = {dp:+.1f}: {cnt}')
        lines.append(f'    Mean: {np.mean(dp_values):+.4f}')
        lines.append('')

    # ---- Turning angle at transitions ----
    if turn_values:
        lines.append('— Turning Angles at Run Boundaries —')
        turn_class_counts = Counter(turning_class(a) for a in turn_values)
        lines.append(f'  {"Class":<25} {"Count":<8}')
        for cls, cnt in turn_class_counts.most_common():
            lines.append(f'  {cls:<25} {cnt:<8}')

        # Histogram
        lines.append(f'\n  Turning angle histogram (30° bins):')
        bins = np.arange(-180, 210, 30)
        hist, _ = np.histogram(turn_values, bins=bins)
        for i in range(len(hist)):
            if hist[i] > 0:
                bar = '█' * hist[i]
                lines.append(f'    [{bins[i]:+4.0f}°,{bins[i+1]:+4.0f}°): '
                             f'{hist[i]:3d} {bar}')
        lines.append('')

    # ---- Run adjacency patterns ----
    lines.append('— Run Adjacency Patterns (root_i → root_{i+1}) —')
    adj_counts = Counter()
    for i in range(len(runs) - 1):
        adj_counts[(runs[i]['root'], runs[i+1]['root'])] += 1

    lines.append(f'  Top 20 run-to-run transitions:')
    lines.append(f'  {"From":<6} {"To":<6} {"Count":<6} {"DotProd":<8} {"Operator"}')
    for (fr, to), cnt in adj_counts.most_common(20):
        dp = float(np.dot(e8_roots[fr], e8_roots[to]))
        _, name, glyph = classify_operator(dp)
        lines.append(f'  {fr:<6} {to:<6} {cnt:<6} {dp:+.1f}     {glyph} {name}')
    lines.append('')

    # ---- "Burst" analysis (clusters of runs) ----
    lines.append('— Arithmetic Bursts (clusters of dense vertices) —')
    # A "burst" is a sequence of runs where prime-index gaps are small
    # We look at the gaps between consecutive run start primes
    if len(runs) >= 2:
        inter_run_gaps = []
        for i in range(len(runs) - 1):
            gap = runs[i+1]['start_prime'] - runs[i]['end_prime']
            inter_run_gaps.append(gap)

        lines.append(f'  Inter-run prime gaps:')
        lines.append(f'    Mean: {np.mean(inter_run_gaps):,.0f}')
        lines.append(f'    Median: {np.median(inter_run_gaps):,.0f}')
        lines.append(f'    Min: {min(inter_run_gaps):,}')
        lines.append(f'    Max: {max(inter_run_gaps):,}')

        # Find bursts: runs that are close together (gap < 10× median)
        median_gap = np.median(inter_run_gaps)
        burst_threshold = max(median_gap * 10, 1000)
        bursts = []
        cur_burst = [0]
        for i in range(len(inter_run_gaps)):
            if inter_run_gaps[i] < burst_threshold:
                cur_burst.append(i + 1)
            else:
                if len(cur_burst) > 1:
                    bursts.append(cur_burst)
                cur_burst = [i + 1]
        if len(cur_burst) > 1:
            bursts.append(cur_burst)

        lines.append(f'\n  Burst threshold: gap < {burst_threshold:,.0f}')
        lines.append(f'  Number of bursts: {len(bursts)}')
        if bursts:
            burst_lens = [len(b) for b in bursts]
            lines.append(f'  Burst sizes: min={min(burst_lens)}, '
                         f'max={max(burst_lens)}, '
                         f'mean={np.mean(burst_lens):.1f}')
            lines.append(f'\n  Top 5 largest bursts:')
            for b in sorted(bursts, key=len, reverse=True)[:5]:
                root_seq = [runs[i]['root'] for i in b]
                p_start = runs[b[0]]['start_prime']
                p_end = runs[b[-1]]['end_prime']
                lines.append(f'    Runs {b[0]}-{b[-1]}: '
                             f'{len(b)} runs, primes {p_start:,}-{p_end:,}, '
                             f'roots: {root_seq}')
    lines.append('')

    # ---- Assembled sentence ----
    sentence = assemble_sentence(runs, e8_roots)
    lines.append('=' * 72)
    lines.append('  Assembled Sentence (Phase-Length Modulation)')
    lines.append('=' * 72)
    lines.append('')
    lines.append('Encoding:')
    lines.append('  UPPER = stable run (3+ edges), lower = transient (1-2)')
    lines.append('  ×N = held tone (9+ edges)')
    lines.append('  , = gentle turn (dp=+1)   ; = orthogonal (dp=0)')
    lines.append('  . = reversal (dp=-1)       ! = antipodal (dp=-2)')
    lines.append('  ■ = terminal')
    lines.append('')

    # Chunk for readability
    lines.append('Sentence (chunks of 60):')
    for i in range(0, len(sentence), 60):
        chunk = sentence[i:i+60]
        lines.append(f'  [{i:4d}] {chunk}')
    lines.append('')
    lines.append(f'Total sentence length: {len(sentence)} characters')
    lines.append(f'Compression ratio: {total_edges} edges → {len(sentence)} chars '
                 f'({len(sentence)/max(total_edges,1)*100:.1f}%)')
    lines.append('')

    # ---- Null model ----
    lines.append('=' * 72)
    lines.append(f'  Null Model ({n_null} random permutations)')
    lines.append('=' * 72)
    lines.append('')

    true_mean_len = np.mean(lengths)
    true_max_len = max(lengths)
    true_n_runs = n_runs
    true_compression = n_runs / max(total_edges, 1)

    null_mean_lens = []
    null_max_lens = []
    null_n_runs_arr = []
    null_compression_arr = []

    rng = np.random.default_rng(42)
    # Collect all edge root-from values for permutation
    all_roots = []
    for r in runs:
        all_roots.extend([r['root']] * r['length'])

    for _ in range(n_null):
        perm_roots = rng.permutation(all_roots)
        # Count runs in permuted sequence
        nr = 1
        rl = [1]
        for j in range(1, len(perm_roots)):
            if perm_roots[j] == perm_roots[j-1]:
                rl[-1] += 1
            else:
                nr += 1
                rl.append(1)
        null_n_runs_arr.append(nr)
        null_mean_lens.append(np.mean(rl))
        null_max_lens.append(max(rl))
        null_compression_arr.append(nr / max(len(perm_roots), 1))

    null_mean_lens = np.array(null_mean_lens)
    null_max_lens = np.array(null_max_lens)
    null_n_runs_arr = np.array(null_n_runs_arr)
    null_compression_arr = np.array(null_compression_arr)

    def zs(true_val, null_arr):
        m, s = np.mean(null_arr), np.std(null_arr)
        return (true_val - m) / s if s > 1e-15 else 0.0

    lines.append(f'{"Metric":<22} {"True":>10} {"Null mean":>10} '
                 f'{"Null std":>10} {"z-score":>10}')
    lines.append('-' * 72)

    lines.append(f'{"Num runs":<22} {true_n_runs:>10} '
                 f'{np.mean(null_n_runs_arr):>10.1f} '
                 f'{np.std(null_n_runs_arr):>10.2f} '
                 f'{zs(true_n_runs, null_n_runs_arr):>+10.2f}')

    lines.append(f'{"Mean run length":<22} {true_mean_len:>10.2f} '
                 f'{np.mean(null_mean_lens):>10.4f} '
                 f'{np.std(null_mean_lens):>10.4f} '
                 f'{zs(true_mean_len, null_mean_lens):>+10.2f}')

    lines.append(f'{"Max run length":<22} {true_max_len:>10} '
                 f'{np.mean(null_max_lens):>10.1f} '
                 f'{np.std(null_max_lens):>10.2f} '
                 f'{zs(true_max_len, null_max_lens):>+10.2f}')

    lines.append(f'{"Compression ratio":<22} {true_compression:>10.4f} '
                 f'{np.mean(null_compression_arr):>10.4f} '
                 f'{np.std(null_compression_arr):>10.4f} '
                 f'{zs(true_compression, null_compression_arr):>+10.2f}')

    lines.append('')
    lines.append('Interpretation:')
    lines.append('  Fewer runs (negative z) = MORE clustering = MORE structured')
    lines.append('  Longer mean run (positive z) = path HOLDS roots longer')
    lines.append('')

    return '\n'.join(lines), sentence


# ================================================================
# Load data from CSVs
# ================================================================

def load_vertices(path):
    verts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            verts.append({
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
    return verts


def load_edges(path):
    edges = []
    with open(path) as f:
        for row in csv.DictReader(f):
            edges.append({
                'edge': int(row['edge']),
                'prime_gap': int(row['prime_gap']),
                'ulam_dx': float(row['ulam_dx']),
                'ulam_dy': float(row['ulam_dy']),
                'ulam_dist': float(row['ulam_dist']),
                'angle': float(row['angle']),
                'turning_angle': float(row['turning_angle']),
                'e8_from': int(row['e8_from']),
                'e8_to': int(row['e8_to']),
                'same_e8': int(row['same_e8']),
                'same_f4': int(row['same_f4']),
                'dot_product': float(row['dot_product']),
            })
    return edges


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Monstrous Assembler: Run-Length compression of crystalline path')
    parser.add_argument('--input-dir', type=str, default='spiral_outputs')
    parser.add_argument('--output-dir', type=str, default='spiral_outputs')
    parser.add_argument('--null-perms', type=int, default=500)
    args = parser.parse_args()

    print('=' * 72)
    print('  Monstrous Assembler v1.0')
    print('  Phase-Length Modulation Decoder')
    print('=' * 72)
    print()

    # Load data
    vert_path = os.path.join(args.input_dir, 'path_vertices.csv')
    edge_path = os.path.join(args.input_dir, 'path_edges.csv')

    if not os.path.exists(vert_path) or not os.path.exists(edge_path):
        print(f'ERROR: Cannot find {vert_path} and/or {edge_path}')
        print('Run path_decoder first:')
        print('  ./path_decoder --max-primes 100000000 --vertices 500')
        sys.exit(1)

    print(f'Loading vertices from {vert_path}...')
    vertices = load_vertices(vert_path)
    print(f'  {len(vertices)} vertices loaded')

    print(f'Loading edges from {edge_path}...')
    edges = load_edges(edge_path)
    print(f'  {len(edges)} edges loaded')
    print()

    # Build E8 roots for coordinate display
    e8_roots = build_e8_roots()

    # Extract runs
    print('Extracting runs (Run-Length Encoding)...')
    t0 = time.time()
    runs = extract_runs(vertices, edges)
    print(f'  {len(runs)} runs extracted from {len(edges)} edges '
          f'in {time.time()-t0:.4f}s')
    print()

    # Quick preview
    print('First 15 runs:')
    print(f'  {"#":<4} {"Root":<6} {"Type":<5} {"Len":<5} '
          f'{"Family":<12} {"Exit Op":<20} {"Primes"}')
    for i, r in enumerate(runs[:15]):
        _, opname, glyph = r['exit_op']
        lines_p = f'{r["start_prime"]:,}-{r["end_prime"]:,}'
        print(f'  {i:<4} {r["root"]:<6} {r["root_type"]:<5} {r["length"]:<5} '
              f'{root_family(r["root"]):<12} {glyph} {opname:<16} {lines_p}')
    print()

    # Full analysis
    print('Running analysis...')
    report, sentence = analyze_runs(runs, e8_roots, n_null=args.null_perms)
    print(report)

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Runs CSV
    csv_path = os.path.join(args.output_dir, 'assembled_runs.csv')
    with open(csv_path, 'w') as f:
        f.write('run,root,root_type,length,start_seq,end_seq,'
                'start_prime,end_prime,exit_dot,exit_turn,'
                'exit_op_sym,exit_op_name,family\n')
        for i, r in enumerate(runs):
            sym, name, glyph = r['exit_op']
            f.write(f'{i},{r["root"]},{r["root_type"]},{r["length"]},'
                    f'{r["start_seq"]},{r["end_seq"]},'
                    f'{r["start_prime"]},{r["end_prime"]},'
                    f'{r["exit_dot"]:.4f},{r["exit_turn"]:.2f},'
                    f'{sym},{name},{root_family(r["root"])}\n')
    print(f'Runs CSV written to: {csv_path}')

    # Report
    report_path = os.path.join(args.output_dir, 'assembler_report.txt')
    with open(report_path, 'w') as f:
        f.write('Monstrous Assembler — Full Report\n')
        f.write(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Vertices: {len(vertices)}, Edges: {len(edges)}, '
                f'Runs: {len(runs)}\n\n')
        f.write(report)
    print(f'Report written to: {report_path}')

    # Sentence file
    sent_path = os.path.join(args.output_dir, 'assembled_sentence.txt')
    with open(sent_path, 'w') as f:
        f.write(sentence)
    print(f'Sentence written to: {sent_path}')

    print()
    print('=' * 72)
    print('  Assembly Complete')
    print('=' * 72)


if __name__ == '__main__':
    main()
