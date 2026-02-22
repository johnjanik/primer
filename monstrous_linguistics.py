#!/usr/bin/env python3
"""
Monstrous Linguistics — N-gram & Cryptographic Profiling of the F4 String

Treats the F4-root-index-mod-26 encoding of prime gaps as a linguistic corpus
and applies classical cryptographic analysis: IC, Kasiski, Shannon entropy,
N-grams, keyword matching, letter frequency profiling.
"""

import argparse
import math
import sys
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from math import gcd, log2
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Import existing infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from e8_f4_prime_analysis import E8Lattice, load_primes
from f4_lattice import F4Lattice


# ---------------------------------------------------------------------------
# F4 string generation (replicates Method 14 pipeline)
# ---------------------------------------------------------------------------

def generate_f4_string(primes: np.ndarray, e8: E8Lattice, f4: F4Lattice) -> str:
    """
    primes → gaps → norm_gaps → E8 assign → F4 filter → f4_idx % 26 + 'A'
    Returns: the F4 letter string (A-Z).

    Vectorized: precompute a 240-entry lookup table (E8 idx → letter or -1),
    then do all gap→E8 assignment in numpy.
    """
    # Precompute: for each E8 root index 0-239, what letter (0-25) does it map to?
    # -1 means not an F4 root.
    e8_to_letter = np.full(240, -1, dtype=np.int8)
    for ei in range(240):
        if f4.is_f4_root(ei):
            f4_idx = f4.project_e8_to_f4(ei)
            if f4_idx is not None:
                e8_to_letter[ei] = f4_idx % 26

    # Vectorized gap computation
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p = np.maximum(log_p, 1.0)
    norm_gaps = gaps / log_p

    # Vectorized E8 assignment: phase = (sqrt(max(g, 0.01)) / sqrt(2)) % 1.0
    target_norm = np.sqrt(np.maximum(norm_gaps, 0.01))
    phase = np.fmod(target_norm / np.sqrt(2.0), 1.0)
    phase[phase < 0] += 1.0
    e8_indices = (phase * 240).astype(np.int32) % 240

    # Lookup: E8 index → letter code (-1 = skip)
    letter_codes = e8_to_letter[e8_indices]

    # Filter to F4-mapped only and convert to string
    mask = letter_codes >= 0
    mapped_codes = letter_codes[mask]
    # Convert to ASCII bytes: 65 = 'A'
    ascii_bytes = (mapped_codes.astype(np.uint8) + 65).tobytes()
    return ascii_bytes.decode('ascii')


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

class MonstrousLinguistics:
    """Cryptographic / linguistic profiler for an alphabetic string."""

    EXCEPTIONAL_KEYWORDS = ['JHI', 'LLL', 'HOD', 'WEY', 'SEL', 'GQQ', 'PRIA']
    EXCEPTIONAL_PERIODS = {
        14: 'G2 dim + 2',
        26: 'F4/2 + 2',
        48: '|F4 roots|',
        52: 'F4 dim',
        126: '|E7 roots|',
        240: '|E8 roots|',
    }

    def __init__(self, text: str):
        self.text = text
        self.N = len(text)
        self._freq_cache: Optional[Dict[str, int]] = None

    # --- letter frequencies ---

    def letter_frequencies(self) -> Dict[str, int]:
        """Return {letter: count} for A-Z."""
        if self._freq_cache is None:
            self._freq_cache = Counter(self.text)
        return dict(self._freq_cache)

    def chi_square_uniformity(self) -> Tuple[float, float]:
        """Chi-square test against uniform distribution. Returns (statistic, p-value)."""
        freq = self.letter_frequencies()
        observed = np.array([freq.get(chr(65 + i), 0) for i in range(26)], dtype=float)
        expected = np.full(26, self.N / 26.0)
        stat, p = stats.chisquare(observed, expected)
        return float(stat), float(p)

    # --- Index of Coincidence ---

    def calculate_ic(self) -> float:
        """Normalized IC (1.0 = random, 1.73 = English)."""
        freq = self.letter_frequencies()
        numerator = sum(n * (n - 1) for n in freq.values())
        denominator = self.N * (self.N - 1) / 26.0
        if denominator == 0:
            return 0.0
        return numerator / denominator

    # --- Shannon entropy ---

    def calculate_entropy(self) -> float:
        """Shannon entropy in bits/char over the 26-letter alphabet."""
        freq = self.letter_frequencies()
        H = 0.0
        for n in freq.values():
            if n > 0:
                p = n / self.N
                H -= p * log2(p)
        return H

    # --- N-gram analysis ---

    def _ngram_counts(self, n: int) -> Counter:
        """Cached n-gram counter."""
        if not hasattr(self, '_ngram_cache'):
            self._ngram_cache = {}
        if n not in self._ngram_cache:
            self._ngram_cache[n] = Counter(
                self.text[i:i+n] for i in range(self.N - n + 1))
        return self._ngram_cache[n]

    def find_ngrams(self, n: int, top_k: int = 20) -> List[Tuple[str, int]]:
        """Return top-k n-grams by frequency."""
        return self._ngram_counts(n).most_common(top_k)

    def ngram_entropy(self, n: int) -> float:
        """Entropy at the n-gram level (bits per n-gram)."""
        counts = self._ngram_counts(n)
        total = sum(counts.values())
        H = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * log2(p)
        return H

    def ngram_distinct_count(self, n: int) -> int:
        """Number of distinct n-grams observed."""
        return len(self._ngram_counts(n))

    # --- Kasiski examination ---

    def kasiski_test(self, seq_len: int = 3, max_periods: int = 10) -> List[Tuple[int, int]]:
        """
        Find candidate periods via Kasiski examination.
        Returns [(period, gcd_count), ...] sorted by count descending.

        For large corpora (>5M chars), samples uniformly to keep runtime bounded.
        """
        # For very large texts, sample positions to keep runtime under ~30s
        sample_size = min(self.N - seq_len + 1, 5_000_000)
        if sample_size < self.N - seq_len + 1:
            rng = np.random.default_rng(42)
            indices = np.sort(rng.choice(self.N - seq_len + 1, size=sample_size, replace=False))
        else:
            indices = range(self.N - seq_len + 1)

        # Find all positions of each trigram
        positions: Dict[str, List[int]] = defaultdict(list)
        for i in indices:
            gram = self.text[i:i+seq_len]
            positions[gram].append(i)

        # Collect distances between repeated trigrams
        distances: List[int] = []
        for gram, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            for i in range(len(pos_list)):
                for j in range(i + 1, min(i + 5, len(pos_list))):
                    distances.append(pos_list[j] - pos_list[i])

        if not distances:
            return []

        # Vectorized factoring: for each candidate factor 2..300, count how many distances are divisible
        dist_arr = np.array(distances, dtype=np.int64)
        factor_counts: Counter = Counter()
        for f in range(2, 301):
            count = int(np.sum(dist_arr % f == 0))
            if count > 0:
                factor_counts[f] = count

        return factor_counts.most_common(max_periods)

    def kasiski_exceptional(self, seq_len: int = 3) -> Dict[int, Tuple[int, str]]:
        """Check Kasiski factors against exceptional Lie-theoretic dimensions."""
        all_factors = dict(self.kasiski_test(seq_len, max_periods=300))
        results = {}
        for period, label in self.EXCEPTIONAL_PERIODS.items():
            count = all_factors.get(period, 0)
            results[period] = (count, label)
        return results

    # --- Keyword matching ---

    def exceptional_match(self) -> Dict[str, Tuple[int, float, float]]:
        """
        Search for exceptional keywords.
        Returns {keyword: (found_count, expected_count, ratio)}.
        """
        results = {}
        for kw in self.EXCEPTIONAL_KEYWORDS:
            k = len(kw)
            found = 0
            start = 0
            while True:
                idx = self.text.find(kw, start)
                if idx == -1:
                    break
                found += 1
                start = idx + 1
            expected = max((self.N - k + 1) / (26 ** k), 1e-6)
            ratio = found / expected if expected > 0 else 0.0
            results[kw] = (found, expected, ratio)
        return results

    # --- Bigram heatmap summary ---

    def bigram_matrix(self) -> np.ndarray:
        """26x26 bigram frequency matrix."""
        mat = np.zeros((26, 26), dtype=int)
        for i in range(self.N - 1):
            r = ord(self.text[i]) - 65
            c = ord(self.text[i + 1]) - 65
            if 0 <= r < 26 and 0 <= c < 26:
                mat[r, c] += 1
        return mat

    # --- Full report ---

    def full_report(self, ngram_max: int = 5) -> str:
        """Generate the complete formatted report."""
        lines = []
        lines.append('=' * 60)
        lines.append('  MONSTROUS LINGUISTICS REPORT')
        lines.append('=' * 60)
        lines.append(f'Corpus: F4 root index mod 26, {self.N:,} characters')
        lines.append('')

        # --- Letter frequencies ---
        lines.append('-' * 60)
        lines.append('  LETTER FREQUENCIES')
        lines.append('-' * 60)
        freq = self.letter_frequencies()
        max_count = max(freq.values()) if freq else 1
        bar_width = 40
        for i in range(26):
            ch = chr(65 + i)
            count = freq.get(ch, 0)
            pct = 100.0 * count / self.N if self.N > 0 else 0
            bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
            bar = '\u2588' * bar_len
            lines.append(f'  {ch}: {bar:<{bar_width}} {count:>8,} ({pct:5.2f}%)')

        chi_stat, chi_p = self.chi_square_uniformity()
        uniform_label = 'UNIFORM' if chi_p > 0.05 else 'NOT uniform'
        lines.append(f'\n  Chi-square statistic: {chi_stat:.2f}')
        lines.append(f'  Chi-square p-value:  {chi_p:.6f} ({uniform_label})')
        lines.append('')

        # --- Global statistics ---
        lines.append('-' * 60)
        lines.append('  GLOBAL STATISTICS')
        lines.append('-' * 60)
        ic = self.calculate_ic()
        entropy = self.calculate_entropy()
        max_ent = log2(26)
        util = 100.0 * entropy / max_ent if max_ent > 0 else 0
        lines.append(f'  Index of Coincidence: {ic:.6f}  (random=1.000, English=1.73)')
        lines.append(f'  Shannon Entropy:      {entropy:.4f} bits/char  (max={max_ent:.3f}, utilization={util:.2f}%)')
        lines.append('')

        # --- Exceptional keyword matches ---
        lines.append('-' * 60)
        lines.append('  EXCEPTIONAL KEYWORD MATCHES')
        lines.append('-' * 60)
        lines.append(f'  {"Keyword":<10} {"Found":>8} {"Expected":>12} {"Ratio":>8}')
        lines.append(f'  {"-"*10} {"-"*8} {"-"*12} {"-"*8}')
        for kw, (found, expected, ratio) in sorted(self.exceptional_match().items()):
            lines.append(f'  {kw:<10} {found:>8,} {expected:>12.1f} {ratio:>7.2f}x')
        lines.append('')

        # --- N-gram analysis ---
        for n in range(2, ngram_max + 1):
            lines.append('-' * 60)
            lines.append(f'  TOP {n}-GRAMS')
            lines.append('-' * 60)
            top = self.find_ngrams(n, top_k=20)
            ng_ent = self.ngram_entropy(n)
            distinct = self.ngram_distinct_count(n)
            possible = 26 ** n
            lines.append(f'  Distinct: {distinct:,} / {possible:,} possible  '
                         f'({100.0 * distinct / possible:.2f}% coverage)')
            lines.append(f'  {n}-gram entropy: {ng_ent:.4f} bits')
            lines.append('')
            for gram, count in top:
                pct = 100.0 * count / (self.N - n + 1)
                lines.append(f'    {gram}: {count:>8,}  ({pct:.3f}%)')
            lines.append('')

        # --- Bigram matrix summary ---
        lines.append('-' * 60)
        lines.append('  BIGRAM FREQUENCY MATRIX (top-left 10x10 excerpt)')
        lines.append('-' * 60)
        mat = self.bigram_matrix()
        header = '     ' + ''.join(f'{chr(65+j):>6}' for j in range(10))
        lines.append(header)
        for i in range(10):
            row = f'  {chr(65+i)}  ' + ''.join(f'{mat[i,j]:>6}' for j in range(10))
            lines.append(row)
        lines.append('  ... (remaining 16 rows/cols omitted)')
        lines.append('')

        # --- Kasiski periodicity ---
        lines.append('-' * 60)
        lines.append('  KASISKI PERIODICITY')
        lines.append('-' * 60)
        kasiski = self.kasiski_test(seq_len=3, max_periods=15)
        lines.append(f'  {"Period":>8} {"Factor count":>14}')
        lines.append(f'  {"-"*8} {"-"*14}')
        for period, count in kasiski:
            lines.append(f'  {period:>8} {count:>14,}')
        lines.append('')

        # Check exceptional dimensions
        lines.append('  Exceptional dimensions in Kasiski factors:')
        exc = self.kasiski_exceptional()
        for period in sorted(exc.keys()):
            count, label = exc[period]
            marker = '  ***' if count > 0 else ''
            lines.append(f'    {period:>4} ({label:<14}): {count:>8,} hits{marker}')
        lines.append('')

        lines.append('=' * 60)
        lines.append('  END OF REPORT')
        lines.append('=' * 60)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Monstrous Linguistics: N-gram & cryptographic profiling of the F4 string')
    parser.add_argument('--max-primes', type=int, default=2_000_000,
                        help='Number of primes to load (default: 2000000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path (default: spiral_outputs/monstrous_linguistics_report.txt)')
    parser.add_argument('--ngram-max', type=int, default=5,
                        help='Maximum N-gram length to analyze (default: 5)')
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        out_dir = Path(__file__).parent / 'spiral_outputs'
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / 'monstrous_linguistics_report.txt')

    # Load primes
    print(f'Loading up to {args.max_primes:,} primes...')
    primes = load_primes(max_primes=args.max_primes)
    print(f'  Loaded {len(primes):,} primes (range [{primes[0]}, {primes[-1]}])')

    # Build lattices
    print('Building E8 (240 roots) and F4 (48 roots) lattices...')
    e8 = E8Lattice()
    f4 = F4Lattice(e8)

    # Generate F4 string
    print('Generating F4 string (Method 14 pipeline)...')
    text = generate_f4_string(primes, e8, f4)
    n_gaps = len(primes) - 1
    f4_pct = 100.0 * len(text) / n_gaps if n_gaps > 0 else 0
    print(f'  {len(text):,} characters from {n_gaps:,} gaps ({f4_pct:.1f}% F4-mapped)')

    # Sanity checks
    assert len(text) > 0, 'F4 string is empty — check prime loading and lattice mapping'
    assert 0.70 < f4_pct / 100.0 < 0.95, (
        f'F4 mapping rate {f4_pct:.1f}% outside expected 70-95% range')

    # Analyze
    print(f'Running analysis (ngram_max={args.ngram_max})...')
    ml = MonstrousLinguistics(text)
    report = ml.full_report(ngram_max=args.ngram_max)

    # Print to stdout
    print()
    print(report)

    # Save to file
    with open(output_path, 'w') as f:
        f.write(report)
    print(f'\nReport saved to {output_path}')


if __name__ == '__main__':
    main()
