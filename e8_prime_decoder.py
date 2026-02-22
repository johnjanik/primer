#!/usr/bin/env python3
"""
E8 Prime Decoder: Experimental Protocol Implementation

This script implements the decoding procedure from Section 6 of the manuscript:
1. Embed blocks of 8 consecutive primes into R^8 via gap structure
2. Decode via E8 lattice (closest vector problem)
3. Measure decoding errors
4. Extract logical bits via E8/2E8 ≅ Hamming code
5. Test extracted bits for non-random structure

Author: Generated for E8 Modular Spacetime research
Date: 2026-01-31
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator
from dataclasses import dataclass
from collections import Counter
import struct
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

PRIMES_DIR = Path("/home/john/mynotes/HodgedeRham")
NUM_FILES = 50  # primes1.txt through primes50.txt
BLOCK_SIZE = 8  # E8 lattice dimension
E8_PACKING_RADIUS = 1.0 / np.sqrt(2)  # ≈ 0.7071

# Output files
OUTPUT_DIR = Path("/home/john/mynotes/HodgedeRham/spiral_outputs")
BITS_FILE = OUTPUT_DIR / "decoded_bits.bin"
ERRORS_FILE = OUTPUT_DIR / "decoding_errors.npy"
STATS_FILE = OUTPUT_DIR / "statistics.txt"
LATTICE_POINTS_FILE = OUTPUT_DIR / "lattice_points.npy"

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DecodingResult:
    """Result of decoding a single block of 8 primes."""
    block_index: int
    primes: np.ndarray          # The 8 primes
    embedding: np.ndarray       # Point in R^8
    lattice_point: np.ndarray   # Nearest E8 lattice point
    error_vector: np.ndarray    # embedding - lattice_point
    error_norm: float           # ||error_vector||
    logical_bits: np.ndarray    # 4 extracted bits
    is_correctable: bool        # error_norm < packing_radius

@dataclass
class ExperimentStats:
    """Aggregate statistics from the experiment."""
    total_blocks: int
    correctable_blocks: int
    uncorrectable_blocks: int
    mean_error: float
    std_error: float
    max_error: float
    min_error: float
    bit_frequencies: np.ndarray  # Frequency of each 4-bit pattern
    total_ones: int
    total_zeros: int

# =============================================================================
# PRIME FILE PARSING
# =============================================================================

def parse_prime_file(filepath: Path) -> Generator[int, None, None]:
    """
    Parse a prime file from t5k.org format.
    
    Format:
    - First line: header "The First X Primes (from t5k.org)"
    - Subsequent lines: whitespace-separated integers, 10 per line
    """
    with open(filepath, 'r') as f:
        # Skip header line
        header = f.readline()
        if "Primes" not in header and "primes" not in header:
            # If no header detected, reset to start
            f.seek(0)
        
        for line in f:
            # Split on whitespace and parse integers
            parts = line.split()
            for part in parts:
                try:
                    prime = int(part.replace(',', ''))
                    yield prime
                except ValueError:
                    # Skip non-integer tokens (e.g., header remnants)
                    continue

def load_all_primes() -> np.ndarray:
    """Load all primes from the data files into a single array."""
    print("Loading primes from files...")
    all_primes = []
    
    for i in range(1, NUM_FILES + 1):
        filepath = PRIMES_DIR / f"primes{i}.txt"
        if not filepath.exists():
            print(f"  Warning: {filepath} not found, stopping at file {i-1}")
            break
        
        file_primes = list(parse_prime_file(filepath))
        all_primes.extend(file_primes)
        print(f"  Loaded {filepath.name}: {len(file_primes):,} primes (total: {len(all_primes):,})")
    
    primes = np.array(all_primes, dtype=np.int64)
    print(f"Total primes loaded: {len(primes):,}")
    return primes

# =============================================================================
# E8 LATTICE OPERATIONS
# =============================================================================

def compute_gap_embedding(primes_block: np.ndarray, prev_prime: int) -> np.ndarray:
    """
    Compute the normalized gap embedding for a block of 8 primes.
    
    Args:
        primes_block: Array of 8 consecutive primes [p_n, p_{n+1}, ..., p_{n+7}]
        prev_prime: The prime before this block (p_{n-1})
    
    Returns:
        Normalized embedding vector in R^8
    """
    # Compute gaps: g_i = p_{n+i} - p_{n+i-1}
    all_primes = np.concatenate([[prev_prime], primes_block])
    gaps = np.diff(all_primes).astype(np.float64)  # 8 gaps
    
    # Center: subtract mean
    mean_gap = np.mean(gaps)
    centered = gaps - mean_gap
    
    # Normalize: scale so typical vectors have length ≈ sqrt(2)
    std_gap = np.std(gaps)
    if std_gap < 1e-10:
        # Degenerate case: all gaps equal
        return np.zeros(8)
    
    # Target length is sqrt(2) (E8 root length)
    # Current length is sqrt(8) * std after centering
    normalized = centered * np.sqrt(2) / (np.sqrt(8) * std_gap)
    
    return normalized

def round_to_integer_lattice(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Round to nearest point in Z^8 with even coordinate sum."""
    z = np.round(x).astype(np.int64)
    
    # Check parity
    if np.sum(z) % 2 != 0:
        # Find coordinate with largest rounding error and flip it
        errors = np.abs(x - z)
        flip_idx = np.argmax(errors)
        # Flip toward the original value
        if x[flip_idx] > z[flip_idx]:
            z[flip_idx] += 1
        else:
            z[flip_idx] -= 1
    
    dist = np.linalg.norm(x - z)
    return z.astype(np.float64), dist

def round_to_half_integer_lattice(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Round to nearest point in (Z + 1/2)^8 with even coordinate sum."""
    c = np.full(8, 0.5)  # The shift vector
    shifted = x - c
    z = np.round(shifted).astype(np.int64)
    
    # Check parity (sum should be even, which means sum of z should be 0 mod 2
    # since each component is z_i + 0.5 and we have 8 of them: sum = sum(z) + 4)
    if np.sum(z) % 2 != 0:
        errors = np.abs(shifted - z)
        flip_idx = np.argmax(errors)
        if shifted[flip_idx] > z[flip_idx]:
            z[flip_idx] += 1
        else:
            z[flip_idx] -= 1
    
    result = z.astype(np.float64) + c
    dist = np.linalg.norm(x - result)
    return result, dist

def e8_decode(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the closest E8 lattice point to x.
    
    The E8 lattice is: D8 ∪ (D8 + c) where c = (1/2, ..., 1/2)
    D8 is the checkerboard lattice: Z^8 with even coordinate sum.
    
    Returns:
        (lattice_point, distance)
    """
    # Try integer sublattice
    y1, d1 = round_to_integer_lattice(x)
    
    # Try half-integer sublattice
    y2, d2 = round_to_half_integer_lattice(x)
    
    # Return the closer one
    if d1 <= d2:
        return y1, d1
    else:
        return y2, d2

def is_e8_lattice_point(y: np.ndarray, tol: float = 1e-10) -> bool:
    """Verify that y is actually an E8 lattice point."""
    # Check if all integer or all half-integer
    frac = y - np.floor(y)
    all_integer = np.all(np.abs(frac) < tol)
    all_half = np.all(np.abs(frac - 0.5) < tol)
    
    if not (all_integer or all_half):
        return False
    
    # Check even sum
    coord_sum = np.sum(y)
    return np.abs(coord_sum - np.round(coord_sum / 2) * 2) < tol

# =============================================================================
# BIT EXTRACTION
# =============================================================================

def extract_bits(lattice_point: np.ndarray) -> np.ndarray:
    """
    Extract 4 logical bits from an E8 lattice point.
    
    Uses the isomorphism E8/2E8 ≅ extended Hamming code H8 ≅ F_2^4.
    
    For integer lattice points (y ∈ Z^8 ∩ E8):
        m1 = (y1 + y2 + y3 + y4) mod 2
        m2 = (y1 + y2 + y5 + y6) mod 2
        m3 = (y1 + y3 + y5 + y7) mod 2
        m4 = y1 mod 2
    
    For half-integer points, translate by -c first.
    """
    y = lattice_point.copy()
    
    # Check if half-integer
    frac = y - np.floor(y)
    if np.all(np.abs(frac - 0.5) < 0.1):
        # Half-integer: translate to integer
        y = y - 0.5
    
    # Convert to integers
    y_int = np.round(y).astype(np.int64)
    
    # Extract bits using Hamming code generator matrix pattern
    m1 = (y_int[0] + y_int[1] + y_int[2] + y_int[3]) % 2
    m2 = (y_int[0] + y_int[1] + y_int[4] + y_int[5]) % 2
    m3 = (y_int[0] + y_int[2] + y_int[4] + y_int[6]) % 2
    m4 = y_int[0] % 2
    
    return np.array([m1, m2, m3, m4], dtype=np.uint8)

def bits_to_nibble(bits: np.ndarray) -> int:
    """Convert 4 bits to a nibble (0-15)."""
    return bits[0] * 8 + bits[1] * 4 + bits[2] * 2 + bits[3]

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    """
    Compute chi-square statistic and p-value approximation.
    
    Returns (chi2_statistic, p_value_approx)
    """
    # Avoid division by zero
    expected = np.maximum(expected, 1e-10)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    
    # Degrees of freedom
    df = len(observed) - 1
    
    # Approximate p-value using chi-square distribution
    # For large df, chi2 ≈ normal with mean df, variance 2*df
    z = (chi2 - df) / np.sqrt(2 * df)
    # Standard normal CDF approximation
    p_value = 0.5 * (1 + np.tanh(z * 0.7978845608))  # Rough approximation
    p_value = 1 - p_value  # Upper tail
    
    return chi2, p_value

def runs_test(bits: np.ndarray) -> Tuple[int, float, float]:
    """
    Wald-Wolfowitz runs test for randomness.
    
    Returns (num_runs, expected_runs, z_score)
    """
    n = len(bits)
    n1 = np.sum(bits)      # Number of 1s
    n0 = n - n1            # Number of 0s
    
    if n1 == 0 or n0 == 0:
        return 0, 0, 0
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            runs += 1
    
    # Expected number of runs
    expected = 1 + 2 * n0 * n1 / n
    
    # Variance
    var = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n * n * (n - 1))
    
    if var < 1e-10:
        return runs, expected, 0
    
    z = (runs - expected) / np.sqrt(var)
    
    return runs, expected, z

def compute_entropy(bit_sequence: np.ndarray, block_size: int = 8) -> float:
    """
    Compute the entropy per bit of the sequence, using block_size-bit blocks.
    """
    n = len(bit_sequence)
    n_blocks = n // block_size
    
    if n_blocks == 0:
        return 0
    
    # Count block frequencies
    counts = Counter()
    for i in range(n_blocks):
        block = tuple(bit_sequence[i*block_size:(i+1)*block_size])
        counts[block] += 1
    
    # Compute entropy
    total = sum(counts.values())
    entropy = 0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Normalize to bits per bit
    return entropy / block_size

def autocorrelation(bits: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation of bit sequence for lags 1 to max_lag."""
    n = len(bits)
    mean = np.mean(bits)
    var = np.var(bits)
    
    if var < 1e-10:
        return np.zeros(max_lag)
    
    autocorr = np.zeros(max_lag)
    bits_centered = bits - mean
    
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        autocorr[lag-1] = np.mean(bits_centered[:-lag] * bits_centered[lag:]) / var
    
    return autocorr

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Execute the full experimental protocol."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load primes
    primes = load_all_primes()
    n_primes = len(primes)
    
    # Number of complete blocks
    # We need 8 primes per block, plus one previous prime for gap calculation
    n_blocks = (n_primes - 1) // BLOCK_SIZE
    print(f"\nProcessing {n_blocks:,} blocks of {BLOCK_SIZE} primes each...")
    
    # Storage for results
    errors = np.zeros(n_blocks, dtype=np.float64)
    lattice_points = np.zeros((n_blocks, 8), dtype=np.float64)
    all_bits = np.zeros(n_blocks * 4, dtype=np.uint8)
    nibble_counts = np.zeros(16, dtype=np.int64)
    
    correctable_count = 0
    uncorrectable_count = 0
    
    # Process blocks
    print_interval = n_blocks // 100 if n_blocks > 100 else 1
    
    for block_idx in range(n_blocks):
        # Get the 8 primes for this block and the previous prime
        start_idx = block_idx * BLOCK_SIZE
        primes_block = primes[start_idx + 1 : start_idx + 1 + BLOCK_SIZE]
        prev_prime = primes[start_idx]
        
        if len(primes_block) < BLOCK_SIZE:
            break
        
        # Step 1: Embed into R^8
        embedding = compute_gap_embedding(primes_block, prev_prime)
        
        # Step 2: Decode via E8 lattice
        lattice_point, error_norm = e8_decode(embedding)
        
        # Verify it's actually an E8 point
        if not is_e8_lattice_point(lattice_point):
            print(f"Warning: Block {block_idx} decoded to non-E8 point!")
        
        # Step 3: Record error
        errors[block_idx] = error_norm
        lattice_points[block_idx] = lattice_point
        
        if error_norm < E8_PACKING_RADIUS:
            correctable_count += 1
        else:
            uncorrectable_count += 1
        
        # Step 4: Extract bits
        bits = extract_bits(lattice_point)
        all_bits[block_idx * 4 : block_idx * 4 + 4] = bits
        
        # Track nibble distribution
        nibble = bits_to_nibble(bits)
        nibble_counts[nibble] += 1
        
        # Progress update
        if block_idx % print_interval == 0:
            pct = 100 * block_idx / n_blocks
            print(f"  Progress: {pct:.1f}% ({block_idx:,}/{n_blocks:,} blocks)", end='\r')
    
    print(f"\n  Completed processing {n_blocks:,} blocks.")
    
    # ==========================================================================
    # SAVE RAW DATA
    # ==========================================================================
    
    print("\nSaving raw data...")
    
    # Save decoded bits as binary file
    with open(BITS_FILE, 'wb') as f:
        # Pack bits into bytes (8 bits per byte)
        n_bytes = len(all_bits) // 8
        for i in range(n_bytes):
            byte_bits = all_bits[i*8:(i+1)*8]
            byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
            f.write(struct.pack('B', byte_val))
        # Handle remaining bits
        remaining = len(all_bits) % 8
        if remaining > 0:
            byte_bits = np.concatenate([all_bits[n_bytes*8:], np.zeros(8-remaining, dtype=np.uint8)])
            byte_val = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
            f.write(struct.pack('B', byte_val))
    
    print(f"  Saved {len(all_bits):,} bits to {BITS_FILE}")
    
    # Save errors
    np.save(ERRORS_FILE, errors)
    print(f"  Saved errors to {ERRORS_FILE}")
    
    # Save lattice points
    np.save(LATTICE_POINTS_FILE, lattice_points)
    print(f"  Saved lattice points to {LATTICE_POINTS_FILE}")
    
    # ==========================================================================
    # STATISTICAL ANALYSIS
    # ==========================================================================
    
    print("\nComputing statistics...")
    
    # Basic error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    median_error = np.median(errors)
    
    # Error distribution percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    error_percentiles = np.percentile(errors, percentiles)
    
    # Bit balance
    total_ones = np.sum(all_bits)
    total_zeros = len(all_bits) - total_ones
    bit_ratio = total_ones / len(all_bits)
    
    # Chi-square test for nibble uniformity
    expected_nibble = np.full(16, n_blocks / 16)
    chi2, p_value = chi_square_test(nibble_counts, expected_nibble)
    
    # Runs test
    num_runs, expected_runs, runs_z = runs_test(all_bits)
    
    # Entropy
    entropy_1bit = compute_entropy(all_bits, block_size=1)
    entropy_4bit = compute_entropy(all_bits, block_size=4)
    entropy_8bit = compute_entropy(all_bits, block_size=8)
    
    # Autocorrelation
    autocorr = autocorrelation(all_bits, max_lag=100)
    max_autocorr = np.max(np.abs(autocorr))
    significant_lags = np.where(np.abs(autocorr) > 2 / np.sqrt(len(all_bits)))[0] + 1
    
    # ==========================================================================
    # WRITE STATISTICS REPORT
    # ==========================================================================
    
    with open(STATS_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("E8 PRIME DECODING EXPERIMENT - STATISTICAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total primes processed:     {n_primes:,}\n")
        f.write(f"Block size:                 {BLOCK_SIZE}\n")
        f.write(f"Total blocks decoded:       {n_blocks:,}\n")
        f.write(f"Total bits extracted:       {len(all_bits):,}\n")
        f.write(f"Bits per prime (effective): {len(all_bits) / n_primes:.4f}\n\n")
        
        f.write("ERROR STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"E8 packing radius (threshold): {E8_PACKING_RADIUS:.6f}\n")
        f.write(f"Correctable blocks:            {correctable_count:,} ({100*correctable_count/n_blocks:.4f}%)\n")
        f.write(f"Uncorrectable blocks:          {uncorrectable_count:,} ({100*uncorrectable_count/n_blocks:.4f}%)\n")
        f.write(f"Mean error:                    {mean_error:.6f}\n")
        f.write(f"Std error:                     {std_error:.6f}\n")
        f.write(f"Median error:                  {median_error:.6f}\n")
        f.write(f"Min error:                     {min_error:.6f}\n")
        f.write(f"Max error:                     {max_error:.6f}\n\n")
        
        f.write("Error percentiles:\n")
        for p, val in zip(percentiles, error_percentiles):
            f.write(f"  {p:3d}th percentile: {val:.6f}\n")
        f.write("\n")
        
        f.write("BIT BALANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total 0s: {total_zeros:,} ({100*total_zeros/len(all_bits):.4f}%)\n")
        f.write(f"Total 1s: {total_ones:,} ({100*total_ones/len(all_bits):.4f}%)\n")
        f.write(f"Ratio (should be ~0.5 for random): {bit_ratio:.6f}\n")
        f.write(f"Deviation from 0.5: {abs(bit_ratio - 0.5):.6f}\n\n")
        
        f.write("NIBBLE (4-BIT) DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern  Count      Expected   Deviation\n")
        for i in range(16):
            pattern = f"{i:04b}"
            count = nibble_counts[i]
            expected = n_blocks / 16
            deviation = (count - expected) / np.sqrt(expected) if expected > 0 else 0
            f.write(f"  {pattern}   {count:10,}  {expected:10.1f}  {deviation:+.2f}σ\n")
        f.write(f"\nChi-square statistic: {chi2:.4f}\n")
        f.write(f"Approximate p-value:  {p_value:.6f}\n")
        f.write(f"Degrees of freedom:   15\n")
        if p_value < 0.01:
            f.write("*** SIGNIFICANT: Distribution is NON-UNIFORM (p < 0.01) ***\n")
        elif p_value < 0.05:
            f.write("* Marginally significant: Distribution may be non-uniform (p < 0.05) *\n")
        else:
            f.write("Distribution is consistent with uniform random (p >= 0.05)\n")
        f.write("\n")
        
        f.write("RUNS TEST\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of runs:    {num_runs:,}\n")
        f.write(f"Expected runs:     {expected_runs:,.1f}\n")
        f.write(f"Z-score:           {runs_z:+.4f}\n")
        if abs(runs_z) > 2.576:
            f.write("*** SIGNIFICANT: Sequence shows clustering/alternation (|z| > 2.576) ***\n")
        elif abs(runs_z) > 1.96:
            f.write("* Marginally significant: Some clustering/alternation (|z| > 1.96) *\n")
        else:
            f.write("Runs are consistent with random (|z| <= 1.96)\n")
        f.write("\n")
        
        f.write("ENTROPY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"1-bit entropy:  {entropy_1bit:.6f} bits/bit (max = 1.0)\n")
        f.write(f"4-bit entropy:  {entropy_4bit:.6f} bits/bit (max = 1.0)\n")
        f.write(f"8-bit entropy:  {entropy_8bit:.6f} bits/bit (max = 1.0)\n")
        if entropy_8bit < 0.95:
            f.write("*** LOW ENTROPY: Sequence has significant structure ***\n")
        else:
            f.write("Entropy is consistent with random (>= 0.95)\n")
        f.write("\n")
        
        f.write("AUTOCORRELATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Max |autocorrelation| (lags 1-100): {max_autocorr:.6f}\n")
        f.write(f"Threshold for significance (2σ):    {2/np.sqrt(len(all_bits)):.6f}\n")
        if len(significant_lags) > 0:
            f.write(f"Significant lags: {significant_lags[:20].tolist()}")
            if len(significant_lags) > 20:
                f.write(f" ... and {len(significant_lags)-20} more")
            f.write("\n")
            f.write("*** SIGNIFICANT: Sequence shows autocorrelation ***\n")
        else:
            f.write("No significant autocorrelation detected\n")
        f.write("\n")
        
        f.write("FIRST 100 DECODED BITS\n")
        f.write("-" * 40 + "\n")
        for i in range(0, min(100, len(all_bits)), 8):
            byte_bits = all_bits[i:i+8]
            byte_str = ''.join(str(b) for b in byte_bits)
            ascii_val = int(byte_str, 2) if len(byte_bits) == 8 else 0
            char = chr(ascii_val) if 32 <= ascii_val < 127 else '.'
            f.write(f"  {byte_str}  ({ascii_val:3d})  '{char}'\n")
        f.write("\n")
        
        f.write("FIRST 50 BLOCKS DETAIL\n")
        f.write("-" * 40 + "\n")
        f.write("Block    Error     Lattice Point                              Bits\n")
        for i in range(min(50, n_blocks)):
            lp = lattice_points[i]
            lp_str = '[' + ', '.join(f'{x:5.1f}' for x in lp) + ']'
            bits = all_bits[i*4:i*4+4]
            bits_str = ''.join(str(b) for b in bits)
            marker = '*' if errors[i] >= E8_PACKING_RADIUS else ' '
            f.write(f"{i:5d}  {errors[i]:.4f}{marker}  {lp_str}  {bits_str}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n")
        f.write("""
If the NULL HYPOTHESIS is true (primes encode no message):
- Decoded bits should be statistically indistinguishable from random
- Nibble chi-square p-value should be > 0.05
- Runs test |z| should be < 1.96
- Entropy should be > 0.95
- No significant autocorrelation

If the ALTERNATIVE HYPOTHESIS is true (primes encode a message):
- Look for patterns in the decoded bits
- Check if bits form valid ASCII/UTF-8
- Look for mathematical constants (pi, e, etc.)
- Check for periodicity or low entropy

The error statistics relate to the RIEMANN HYPOTHESIS:
- If all errors < packing radius (0.7071), consistent with RH
- Errors exceeding the threshold suggest "uncorrectable" blocks
""")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nStatistics saved to {STATS_FILE}")
    
    # ==========================================================================
    # PRINT SUMMARY TO CONSOLE
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Blocks processed:    {n_blocks:,}")
    print(f"Correctable:         {correctable_count:,} ({100*correctable_count/n_blocks:.2f}%)")
    print(f"Uncorrectable:       {uncorrectable_count:,} ({100*uncorrectable_count/n_blocks:.2f}%)")
    print(f"Mean error:          {mean_error:.4f} (threshold: {E8_PACKING_RADIUS:.4f})")
    print(f"Bit ratio (1s):      {bit_ratio:.4f} (expected: 0.5)")
    print(f"8-bit entropy:       {entropy_8bit:.4f} (max: 1.0)")
    print(f"Chi-square p-value:  {p_value:.4f}")
    print(f"Runs test z-score:   {runs_z:+.4f}")
    print("=" * 60)
    
    # Verdict
    is_random = (p_value > 0.05 and abs(runs_z) < 1.96 and entropy_8bit > 0.95)
    print("\nVERDICT:", "Consistent with RANDOM (null hypothesis)" if is_random 
          else "Shows STRUCTURE (alternative hypothesis may hold)")
    
    return errors, all_bits, nibble_counts

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("E8 Prime Decoder - Experimental Protocol")
    print("=" * 60)
    
    try:
        errors, bits, nibbles = run_experiment()
        print("\nExperiment completed successfully.")
    except FileNotFoundError as e:
        print(f"\nError: Could not find prime data files.")
        print(f"Expected location: {PRIMES_DIR}")
        print(f"Please ensure primes1.txt through primes{NUM_FILES}.txt exist.")
        raise
    except Exception as e:
        print(f"\nError during experiment: {e}")
        raise
