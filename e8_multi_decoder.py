#!/usr/bin/env python3
"""
E8 Multi-Method Decoder

Tries ALL extraction methods on the prime gap data simultaneously,
outputting readable text for each so the user can evaluate which
(if any) produces a meaningful message.

Methods:
  Group 1: E8 Lattice Decoding (Hamming 4-bit, sign bits, sublattice, parity)
  Group 2: E8 Root Assignment (root index mod 256, projection slope)
  Group 3: Raw Gap Properties (gap as ASCII, normalized gap quantized)
  Group 4: F4 Sub-harmonic (F4 root index, Jordan trace classification)
  Group 5: Crystalline Vertices (vertex primes, vertex gap spacing)
  Group 6: Filtered subsets (low-error only, first-N blocks, Mersenne windows)

Usage:
  python3 e8_multi_decoder.py [--max-primes 2000000]
"""

import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional
import struct
import math
import time

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))

from e8_prime_decoder import (
    load_all_primes, compute_gap_embedding, e8_decode, extract_bits,
    bits_to_nibble, E8_PACKING_RADIUS, is_e8_lattice_point,
    chi_square_test, runs_test, compute_entropy
)
from e8_f4_prime_analysis import E8Lattice, E8F4PrimeAnalyzer, load_primes, ulam_coords
from f4_lattice import F4Lattice
from e7_lattice import E7Lattice
from e6_lattice import E6Lattice
from g2_lattice import G2Lattice
from jordan_algebra import JordanTrace
from f4_eft import F4ExceptionalFourierTransform

# ============================================================================
# Result container
# ============================================================================

@dataclass
class MethodResult:
    name: str
    raw_bytes: bytes
    description: str


# ============================================================================
# Utility functions
# ============================================================================

def bytes_to_printable(data: bytes) -> str:
    """Filter to printable ASCII only."""
    return ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)


def longest_printable_run(data: bytes) -> Tuple[str, int, int]:
    """Find longest contiguous run of printable ASCII."""
    best_str = ""
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_chars = []

    for i, b in enumerate(data):
        if 32 <= b < 127:
            if not cur_chars:
                cur_start = i
            cur_chars.append(chr(b))
        else:
            if len(cur_chars) > best_len:
                best_str = ''.join(cur_chars)
                best_start = cur_start
                best_len = len(cur_chars)
            cur_chars = []
            cur_start = i + 1

    if len(cur_chars) > best_len:
        best_str = ''.join(cur_chars)
        best_start = cur_start
        best_len = len(cur_chars)

    return best_str, best_len, best_start


def byte_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte."""
    if len(data) == 0:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def chi_square_byte(data: bytes) -> float:
    """Chi-square p-value for byte uniformity."""
    if len(data) < 256:
        return 1.0
    counts = np.zeros(256)
    for b in data:
        counts[b] += 1
    expected = np.full(256, len(data) / 256.0)
    chi2 = np.sum((counts - expected)**2 / expected)
    # Normal approximation for chi-square with df=255
    z = (chi2 - 255) / math.sqrt(2 * 255)
    p = 0.5 * (1 + math.tanh(z * 0.7978845608))
    return 1 - p


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack bit array into bytes (MSB first)."""
    n_bytes = len(bits) // 8
    result = bytearray(n_bytes)
    for i in range(n_bytes):
        val = 0
        for j in range(8):
            val = (val << 1) | int(bits[i*8 + j])
        result[i] = val
    return bytes(result)


def nibbles_to_bytes_high_low(nibbles: np.ndarray) -> bytes:
    """Pair nibbles as high-low into bytes."""
    n_bytes = len(nibbles) // 2
    result = bytearray(n_bytes)
    for i in range(n_bytes):
        result[i] = (int(nibbles[2*i]) << 4) | int(nibbles[2*i+1])
    return bytes(result)


def nibbles_to_bytes_low_high(nibbles: np.ndarray) -> bytes:
    """Pair nibbles as low-high into bytes."""
    n_bytes = len(nibbles) // 2
    result = bytearray(n_bytes)
    for i in range(n_bytes):
        result[i] = (int(nibbles[2*i+1]) << 4) | int(nibbles[2*i])
    return bytes(result)


# ============================================================================
# Method implementations
# ============================================================================

def method_hamming_4bit(primes: np.ndarray) -> List[MethodResult]:
    """Group 1a: E8/2E8 Hamming code extraction (4 bits per 8-prime block)."""
    results = []
    n_blocks = (len(primes) - 1) // 8
    all_nibbles = np.zeros(n_blocks, dtype=np.uint8)
    all_bits = np.zeros(n_blocks * 4, dtype=np.uint8)
    errors = np.zeros(n_blocks)

    for bi in range(n_blocks):
        start = bi * 8
        block = primes[start+1:start+9]
        prev = primes[start]
        if len(block) < 8:
            break
        emb = compute_gap_embedding(block, prev)
        lp, err = e8_decode(emb)
        errors[bi] = err
        bits = extract_bits(lp)
        all_bits[bi*4:bi*4+4] = bits
        all_nibbles[bi] = bits_to_nibble(bits)

    # All blocks, high-low
    data_hl = nibbles_to_bytes_high_low(all_nibbles)
    results.append(MethodResult("Hamming 4-bit (all, high-low)", data_hl,
        "E8/2E8 ≅ H8 ≅ F₂⁴, 4 bits per 8-prime block, nibbles paired high-low"))

    # All blocks, low-high
    data_lh = nibbles_to_bytes_low_high(all_nibbles)
    results.append(MethodResult("Hamming 4-bit (all, low-high)", data_lh,
        "Same extraction, nibbles paired low-high"))

    # Low-error blocks only
    mask = errors < E8_PACKING_RADIUS
    if np.sum(mask) > 10:
        good_nibs = all_nibbles[mask]
        data_good = nibbles_to_bytes_high_low(good_nibs)
        results.append(MethodResult("Hamming 4-bit (low-error only)", data_good,
            f"Only correctable blocks (error < {E8_PACKING_RADIUS:.4f}), "
            f"{np.sum(mask)}/{n_blocks} blocks"))

    # Raw 4-bit stream as bytes
    data_raw = bits_to_bytes(all_bits)
    results.append(MethodResult("Hamming raw bits", data_raw,
        "Raw 4-bit stream packed into bytes (MSB first)"))

    return results


def method_sign_bits(primes: np.ndarray) -> List[MethodResult]:
    """Group 1b: Sign bits of E8 lattice point (8 bits per block)."""
    n_blocks = (len(primes) - 1) // 8
    all_bits = np.zeros(n_blocks * 8, dtype=np.uint8)

    for bi in range(n_blocks):
        start = bi * 8
        block = primes[start+1:start+9]
        prev = primes[start]
        if len(block) < 8:
            break
        emb = compute_gap_embedding(block, prev)
        lp, _ = e8_decode(emb)
        for d in range(8):
            all_bits[bi*8 + d] = 1 if lp[d] >= 0 else 0

    data = bits_to_bytes(all_bits)
    return [MethodResult("E8 sign bits", data,
        "Sign of each coordinate of nearest E8 lattice point (8 bits/block)")]


def method_sublattice_flag(primes: np.ndarray) -> List[MethodResult]:
    """Group 1c: Integer vs half-integer sublattice (1 bit per block)."""
    n_blocks = (len(primes) - 1) // 8
    all_bits = np.zeros(n_blocks, dtype=np.uint8)

    for bi in range(n_blocks):
        start = bi * 8
        block = primes[start+1:start+9]
        prev = primes[start]
        if len(block) < 8:
            break
        emb = compute_gap_embedding(block, prev)
        lp, _ = e8_decode(emb)
        frac = lp[0] - math.floor(lp[0])
        all_bits[bi] = 1 if abs(frac - 0.5) < 0.1 else 0

    data = bits_to_bytes(all_bits)
    return [MethodResult("Sublattice flag", data,
        "1 bit per block: 0=integer sublattice, 1=half-integer sublattice")]


def method_parity_bits(primes: np.ndarray) -> List[MethodResult]:
    """Group 1d: Coordinate parities of E8 lattice point (8 bits per block)."""
    n_blocks = (len(primes) - 1) // 8
    all_bits = np.zeros(n_blocks * 8, dtype=np.uint8)

    for bi in range(n_blocks):
        start = bi * 8
        block = primes[start+1:start+9]
        prev = primes[start]
        if len(block) < 8:
            break
        emb = compute_gap_embedding(block, prev)
        lp, _ = e8_decode(emb)
        for d in range(8):
            coord = int(round(lp[d]))
            all_bits[bi*8 + d] = coord % 2

    data = bits_to_bytes(all_bits)
    return [MethodResult("E8 parity bits", data,
        "Parity (mod 2) of each coordinate of E8 lattice point (8 bits/block)")]


def method_e8_root_index(primes: np.ndarray, e8: E8Lattice) -> List[MethodResult]:
    """Group 2a: E8 root index mod 256 as byte."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    data = bytearray(len(norm_gaps))
    for i, g in enumerate(norm_gaps):
        idx = e8.assign_root(g)
        data[i] = idx % 256

    return [MethodResult("E8 root index mod 256", bytes(data),
        "Each gap's assigned E8 root index (0-239) taken mod 256")]


def method_e8_projection_slope(primes: np.ndarray, e8: E8Lattice) -> List[MethodResult]:
    """Group 2b: E8 projection slope quantized to byte."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    slopes = e8.projected_slopes
    data = bytearray(len(norm_gaps))
    for i, g in enumerate(norm_gaps):
        idx = e8.assign_root(g)
        s = slopes[idx]
        # Map slope [-10, 10] → [0, 255]
        t = (s + 10.0) / 20.0
        t = max(0.0, min(1.0, t))
        data[i] = int(t * 255)

    return [MethodResult("E8 projection slope (quantized)", bytes(data),
        "E8 projection slope (sum(root[4:])/sum(root[:4])) mapped to byte")]


def method_gap_ascii(primes: np.ndarray) -> List[MethodResult]:
    """Group 3a: Gap values directly as ASCII characters."""
    gaps = np.diff(primes.astype(np.int64))
    data = bytearray(len(gaps))
    for i, g in enumerate(gaps):
        data[i] = int(g) % 256

    results = [MethodResult("Gap value mod 256", bytes(data),
        "Raw gap size modulo 256 as byte")]

    # Also try gap/2 (since gaps are even for p>2)
    data2 = bytearray(len(gaps))
    for i, g in enumerate(gaps):
        data2[i] = (int(g) // 2) % 256
    results.append(MethodResult("Gap/2 mod 256", bytes(data2),
        "Gap size divided by 2, modulo 256"))

    return results


def method_norm_gap_quantized(primes: np.ndarray) -> List[MethodResult]:
    """Group 3b: Normalized gap quantized to byte."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    # Map [0, 4] → [0, 255]
    data = bytearray(len(norm_gaps))
    for i, g in enumerate(norm_gaps):
        t = g / 4.0
        t = max(0.0, min(1.0, t))
        data[i] = int(t * 255)

    return [MethodResult("Normalized gap (quantized)", bytes(data),
        "g̃ = gap/log(p), mapped from [0,4] to [0,255]")]


def method_f4_root_index(primes: np.ndarray, e8: E8Lattice,
                         f4: F4Lattice) -> List[MethodResult]:
    """Group 4a: F4 root index for F4-mapped gaps."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    f4_indices = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if f4.is_f4_root(e8_idx):
            f4_idx = f4.project_e8_to_f4(e8_idx)
            if f4_idx is not None:
                f4_indices.append(f4_idx)

    if not f4_indices:
        return []

    # Pack: 6 bits per index (0-47), pack 4 per 3 bytes
    # Simpler: just use index as byte (0-47, repeated mod patterns)
    data = bytearray(len(f4_indices))
    for i, idx in enumerate(f4_indices):
        data[i] = idx % 256

    results = [MethodResult("F4 root index (F4 gaps only)", bytes(data),
        f"F4 root index (0-47) for {len(f4_indices)} F4-mapped gaps")]

    # Also try mod 26 + 'A' for letter encoding
    data_letters = bytearray(len(f4_indices))
    for i, idx in enumerate(f4_indices):
        data_letters[i] = 65 + (idx % 26)  # A-Z
    results.append(MethodResult("F4 root index mod 26 as letter", bytes(data_letters),
        "F4 root index mod 26 mapped to A-Z"))

    return results


def method_jordan_trace_bits(primes: np.ndarray, e8: E8Lattice,
                             f4: F4Lattice) -> List[MethodResult]:
    """Group 4b: Jordan trace classification as bits."""
    jt = JordanTrace()
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    bits = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if not f4.is_f4_root(e8_idx):
            continue
        f4_idx = f4.project_e8_to_f4(e8_idx)
        if f4_idx is None:
            continue
        root = f4.get_f4_root(f4_idx)
        trace = jt(root)

        # Encode: 2 bits per gap
        # bit 0: positive (1) or negative (0)
        # bit 1: magnitude > 1 (1) or <= 1 (0)
        bits.append(1 if trace > 0 else 0)
        bits.append(1 if abs(trace) > 1.0 else 0)

    if not bits:
        return []

    bits_arr = np.array(bits, dtype=np.uint8)
    data = bits_to_bytes(bits_arr)
    return [MethodResult("Jordan trace classification bits", data,
        f"2 bits per F4 gap: sign + magnitude threshold at 1.0 ({len(bits)//2} gaps)")]


def method_e7_root_index(primes: np.ndarray, e8: E8Lattice,
                         e7: E7Lattice) -> List[MethodResult]:
    """Group 6a: E7 root index for E7-mapped gaps."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    e7_indices = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if e7.is_e7_root(e8_idx):
            e7_idx = e7.project_e8_to_e7(e8_idx)
            if e7_idx is not None:
                e7_indices.append(e7_idx)

    if not e7_indices:
        return []

    data = bytearray(len(e7_indices))
    for i, idx in enumerate(e7_indices):
        data[i] = idx % 256

    results = [MethodResult("E7 root index (E7 gaps only)", bytes(data),
        f"E7 root index (0-125) for {len(e7_indices)} E7-mapped gaps")]

    data_letters = bytearray(len(e7_indices))
    for i, idx in enumerate(e7_indices):
        data_letters[i] = 65 + (idx % 26)
    results.append(MethodResult("E7 root index mod 26 as letter", bytes(data_letters),
        "E7 root index mod 26 mapped to A-Z"))

    return results


def method_e7_trace_bits(primes: np.ndarray, e8: E8Lattice,
                         e7: E7Lattice) -> List[MethodResult]:
    """Group 6b: E7 trace-8 classification as bits."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    bits = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if not e7.is_e7_root(e8_idx):
            continue
        e7_idx = e7.project_e8_to_e7(e8_idx)
        if e7_idx is None:
            continue
        trace = e7.jordan_trace(e7_idx)

        # 2 bits: sign + magnitude threshold at 2.0
        bits.append(1 if trace > 0 else 0)
        bits.append(1 if abs(trace) > 2.0 else 0)

    if not bits:
        return []

    bits_arr = np.array(bits, dtype=np.uint8)
    data = bits_to_bytes(bits_arr)
    return [MethodResult("E7 trace-8 classification bits", data,
        f"2 bits per E7 gap: sign + magnitude threshold at 2.0 ({len(bits)//2} gaps)")]


def method_e6_root_index(primes: np.ndarray, e8: E8Lattice,
                         e6: E6Lattice) -> List[MethodResult]:
    """Group 7a: E6 root index for E6-mapped gaps."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    e6_indices = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if e6.is_e6_root(e8_idx):
            e6_idx = e6.project_e8_to_e6(e8_idx)
            if e6_idx is not None:
                e6_indices.append(e6_idx)

    if not e6_indices:
        return []

    data = bytearray(len(e6_indices))
    for i, idx in enumerate(e6_indices):
        data[i] = idx % 256

    results = [MethodResult("E6 root index (E6 gaps only)", bytes(data),
        f"E6 root index (0-71) for {len(e6_indices)} E6-mapped gaps")]

    data_letters = bytearray(len(e6_indices))
    for i, idx in enumerate(e6_indices):
        data_letters[i] = 65 + (idx % 26)
    results.append(MethodResult("E6 root index mod 26 as letter", bytes(data_letters),
        "E6 root index mod 26 mapped to A-Z"))

    return results


def method_e6_trace_bits(primes: np.ndarray, e8: E8Lattice,
                         e6: E6Lattice) -> List[MethodResult]:
    """Group 7b: E6 trace-8 classification as bits."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    bits = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if not e6.is_e6_root(e8_idx):
            continue
        e6_idx = e6.project_e8_to_e6(e8_idx)
        if e6_idx is None:
            continue
        trace = e6.jordan_trace(e6_idx)

        bits.append(1 if trace > 0 else 0)
        bits.append(1 if abs(trace) > 2.0 else 0)

    if not bits:
        return []

    bits_arr = np.array(bits, dtype=np.uint8)
    data = bits_to_bytes(bits_arr)
    return [MethodResult("E6 trace-8 classification bits", data,
        f"2 bits per E6 gap: sign + magnitude threshold at 2.0 ({len(bits)//2} gaps)")]


def method_g2_root_index(primes: np.ndarray, e8: E8Lattice,
                         g2: G2Lattice) -> List[MethodResult]:
    """Group 8a: G2 root index for G2-mapped gaps."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    g2_indices = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if g2.is_g2_root(e8_idx):
            g2_idx = g2.project_e8_to_g2(e8_idx)
            if g2_idx is not None:
                g2_indices.append(g2_idx)

    if not g2_indices:
        return []

    data = bytearray(len(g2_indices))
    for i, idx in enumerate(g2_indices):
        data[i] = idx % 256

    results = [MethodResult("G2 root index (G2 gaps only)", bytes(data),
        f"G2 root index (0-11) for {len(g2_indices)} G2-mapped gaps")]

    data_letters = bytearray(len(g2_indices))
    for i, idx in enumerate(g2_indices):
        data_letters[i] = 65 + (idx % 12)  # Map to A-L (12 roots)
    results.append(MethodResult("G2 root index as letter (A-L)", bytes(data_letters),
        "G2 root index mapped to A-L (12 roots)"))

    return results


def method_g2_trace_bits(primes: np.ndarray, e8: E8Lattice,
                         g2: G2Lattice) -> List[MethodResult]:
    """Group 8b: G2 trace-2 classification as bits."""
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p

    bits = []
    for g in norm_gaps:
        e8_idx = e8.assign_root(g)
        if not g2.is_g2_root(e8_idx):
            continue
        g2_idx = g2.project_e8_to_g2(e8_idx)
        if g2_idx is None:
            continue
        trace = g2.jordan_trace(g2_idx)

        # 2 bits: sign + long/short root
        bits.append(1 if trace > 0 else 0)
        bits.append(1 if g2.is_long_root(g2_idx) else 0)

    if not bits:
        return []

    bits_arr = np.array(bits, dtype=np.uint8)
    data = bits_to_bytes(bits_arr)
    return [MethodResult("G2 trace-2 + long/short bits", data,
        f"2 bits per G2 gap: trace sign + long/short ({len(bits)//2} gaps)")]


def method_crystalline_vertices(primes: np.ndarray, vertex_indices: np.ndarray,
                                e8: E8Lattice) -> List[MethodResult]:
    """Group 5: Crystalline vertex prime properties."""
    results = []
    gaps = np.diff(primes.astype(np.int64))

    # 5a: Gap sizes at vertex positions
    valid = vertex_indices[vertex_indices < len(gaps)]
    vertex_gaps = gaps[valid]

    if len(vertex_gaps) > 0:
        data = bytearray(len(vertex_gaps))
        for i, g in enumerate(vertex_gaps):
            data[i] = int(g) % 256
        results.append(MethodResult("Vertex gap values mod 256", bytes(data),
            f"Gap sizes at {len(vertex_gaps)} crystalline vertex positions"))

        # mod 26 as letters
        data_let = bytearray(len(vertex_gaps))
        for i, g in enumerate(vertex_gaps):
            data_let[i] = 65 + (int(g) % 26)
        results.append(MethodResult("Vertex gaps mod 26 as letters", bytes(data_let),
            "Gap sizes at vertices mod 26 mapped to A-Z"))

    # 5b: Vertex gap spacing (differences between consecutive vertex positions)
    if len(valid) > 1:
        sorted_v = np.sort(valid)
        spacings = np.diff(sorted_v)
        data_sp = bytearray(len(spacings))
        for i, s in enumerate(spacings):
            data_sp[i] = int(s) % 256
        results.append(MethodResult("Vertex spacing mod 256", bytes(data_sp),
            f"Differences between consecutive vertex positions ({len(spacings)} values)"))

    # 5c: Vertex prime values mod 256
    valid_prime_idx = vertex_indices[vertex_indices + 1 < len(primes)] + 1
    if len(valid_prime_idx) > 0:
        vertex_primes = primes[valid_prime_idx]
        data_p = bytearray(len(vertex_primes))
        for i, p in enumerate(vertex_primes):
            data_p[i] = int(p) % 256
        results.append(MethodResult("Vertex prime values mod 256", bytes(data_p),
            f"Prime values at vertex positions, mod 256"))

    return results


# ============================================================================
# First-N filtering
# ============================================================================

def first_n_filter(data: bytes, n: int, name: str) -> Optional[MethodResult]:
    """Take only first n bytes."""
    if len(data) < n:
        return None
    return MethodResult(f"{name} (first {n})", data[:n],
        f"First {n} bytes of '{name}'")


# ============================================================================
# Result printing
# ============================================================================

def print_result(idx: int, result: MethodResult):
    """Print a single method result."""
    data = result.raw_bytes
    if not data:
        print(f"\n=== METHOD {idx}: {result.name} ===")
        print(f"Description: {result.description}")
        print("  (no data)")
        return

    printable = bytes_to_printable(data)
    run_str, run_len, run_start = longest_printable_run(data)
    ent = byte_entropy(data)
    chi_p = chi_square_byte(data)

    print(f"\n{'='*70}")
    print(f"METHOD {idx}: {result.name}")
    print(f"{'='*70}")
    print(f"Description: {result.description}")
    print(f"Total bytes: {len(data)}")
    print(f"Raw hex (first 100 bytes): {data[:100].hex()}")
    print(f"ASCII (first 200 chars): {printable[:200]}")
    print(f"Longest printable run: \"{run_str[:80]}\" (len={run_len}, start=byte {run_start})")
    print(f"Entropy: {ent:.3f} bits/byte (max 8.0)")
    print(f"Chi-square p-value: {chi_p:.4f}")

    # If there's a notable printable run, highlight it
    if run_len >= 8:
        print(f"  ** Notable printable run of {run_len} chars **")


# ============================================================================
# Summary table
# ============================================================================

def print_summary(all_results: List[Tuple[int, MethodResult]]):
    """Print ranked summary table."""
    rows = []
    for idx, result in all_results:
        data = result.raw_bytes
        if not data:
            continue
        _, run_len, _ = longest_printable_run(data)
        ent = byte_entropy(data)
        chi_p = chi_square_byte(data)
        rows.append((idx, result.name, len(data), run_len, ent, chi_p))

    print(f"\n{'='*90}")
    print("SUMMARY — Ranked by Longest Printable Run")
    print(f"{'='*90}")
    print(f"{'#':>3}  {'Method':<45} {'Bytes':>7} {'MaxRun':>6} {'Entropy':>7} {'Chi-p':>7}")
    print(f"{'-'*3}  {'-'*45} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")

    rows.sort(key=lambda r: -r[3])  # Sort by longest run descending
    for idx, name, nbytes, run_len, ent, chi_p in rows:
        name_trunc = name[:45]
        print(f"{idx:3d}  {name_trunc:<45} {nbytes:7d} {run_len:6d} {ent:7.3f} {chi_p:7.4f}")

    print()

    # Also rank by lowest entropy
    print(f"{'='*90}")
    print("SUMMARY — Ranked by Lowest Entropy (most structure)")
    print(f"{'='*90}")
    print(f"{'#':>3}  {'Method':<45} {'Bytes':>7} {'MaxRun':>6} {'Entropy':>7} {'Chi-p':>7}")
    print(f"{'-'*3}  {'-'*45} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")

    rows.sort(key=lambda r: r[4])  # Sort by entropy ascending
    for idx, name, nbytes, run_len, ent, chi_p in rows:
        name_trunc = name[:45]
        print(f"{idx:3d}  {name_trunc:<45} {nbytes:7d} {run_len:6d} {ent:7.3f} {chi_p:7.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="E8 Multi-Method Decoder")
    parser.add_argument("--max-primes", type=int, default=2000000,
                        help="Maximum number of primes to load (default 2000000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to file")
    args = parser.parse_args()

    print("=" * 70)
    print("E8 MULTI-METHOD DECODER")
    print("Trying ALL extraction methods on prime data")
    print("=" * 70)

    start_time = time.time()

    # Load primes
    print(f"\nLoading up to {args.max_primes:,} primes...")
    primes = load_primes(max_primes=args.max_primes)
    print(f"Loaded {len(primes):,} primes (range {primes[0]} to {primes[-1]})")

    # Set up E8/F4 infrastructure
    print("\nInitializing E8/F4 lattices...")
    e8 = E8Lattice()
    f4 = F4Lattice(e8)

    # Compute E8 assignments for F4 methods
    print("Computing E8 assignments...")
    gaps = np.diff(primes.astype(np.float64))
    log_p = np.log(primes[:-1].astype(np.float64))
    log_p[log_p < 1] = 1
    norm_gaps = gaps / log_p
    e8_assignments = np.array([e8.assign_root(g) for g in norm_gaps])

    # Compute crystalline vertices
    print("Extracting crystalline vertices...")
    f4_eft = F4ExceptionalFourierTransform(e8)
    vertex_indices = f4_eft.extract_crystalline_pattern(norm_gaps, e8_assignments, n_vertices=500)

    # Collect all results
    all_results = []
    method_idx = 1

    # === Group 1: E8 Lattice Decoding ===
    print("\n--- Group 1: E8 Lattice Decoding ---")

    print("  1a: Hamming 4-bit extraction...")
    for r in method_hamming_4bit(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  1b: Sign bits...")
    for r in method_sign_bits(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  1c: Sublattice flag...")
    for r in method_sublattice_flag(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  1d: Parity bits...")
    for r in method_parity_bits(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 2: E8 Root Assignment ===
    print("\n--- Group 2: E8 Root Assignment ---")

    print("  2a: Root index mod 256...")
    for r in method_e8_root_index(primes, e8):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  2b: Projection slope...")
    for r in method_e8_projection_slope(primes, e8):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 3: Raw Gap Properties ===
    print("\n--- Group 3: Raw Gap Properties ---")

    print("  3a: Gap as ASCII...")
    for r in method_gap_ascii(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  3b: Normalized gap quantized...")
    for r in method_norm_gap_quantized(primes):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 4: F4 Sub-harmonic ===
    print("\n--- Group 4: F4 Sub-harmonic ---")

    print("  4a: F4 root index...")
    for r in method_f4_root_index(primes, e8, f4):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  4b: Jordan trace bits...")
    for r in method_jordan_trace_bits(primes, e8, f4):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 5: Crystalline Vertices ===
    print("\n--- Group 5: Crystalline Vertices ---")

    print("  5: Vertex properties...")
    for r in method_crystalline_vertices(primes, vertex_indices, e8):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 6: E7 Sub-harmonic ===
    print("\n--- Group 6: E7 Sub-harmonic ---")

    print("  Initializing E7 lattice...")
    e7 = E7Lattice(e8)

    print("  6a: E7 root index...")
    for r in method_e7_root_index(primes, e8, e7):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  6b: E7 trace bits...")
    for r in method_e7_trace_bits(primes, e8, e7):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 7: E6 Sub-harmonic ===
    print("\n--- Group 7: E6 Sub-harmonic ---")

    print("  Initializing E6 lattice...")
    e6 = E6Lattice(e8)

    print("  7a: E6 root index...")
    for r in method_e6_root_index(primes, e8, e6):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  7b: E6 trace bits...")
    for r in method_e6_trace_bits(primes, e8, e6):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 8: G2 Sub-harmonic ===
    print("\n--- Group 8: G2 Sub-harmonic ---")

    print("  Initializing G2 lattice...")
    g2 = G2Lattice(e8)

    print("  8a: G2 root index...")
    for r in method_g2_root_index(primes, e8, g2):
        all_results.append((method_idx, r))
        method_idx += 1

    print("  8b: G2 trace bits...")
    for r in method_g2_trace_bits(primes, e8, g2):
        all_results.append((method_idx, r))
        method_idx += 1

    # === Group 9: First-N filters on best methods ===
    print("\n--- Group 9: First-N Subsets ---")
    # Apply first-100, first-1000 to each prior method
    for orig_idx, orig_result in list(all_results):
        for n in [100, 1000]:
            filtered = first_n_filter(orig_result.raw_bytes, n, orig_result.name)
            if filtered:
                all_results.append((method_idx, filtered))
                method_idx += 1

    elapsed = time.time() - start_time
    print(f"\nAll {len(all_results)} method variants computed in {elapsed:.1f}s")

    # Print all results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    for idx, result in all_results:
        print_result(idx, result)

    # Print summary
    print_summary(all_results)

    # Optionally save to file
    output_dir = Path("/home/john/mynotes/HodgedeRham/spiral_outputs")
    output_dir.mkdir(exist_ok=True)
    out_path = args.output or str(output_dir / "decoder_results.txt")

    with open(out_path, 'w') as f:
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        for idx, result in all_results:
            print_result(idx, result)
        print_summary(all_results)

        content = sys.stdout.getvalue()
        sys.stdout = old_stdout
        f.write(content)

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
