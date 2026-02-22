#!/usr/bin/env python3
"""
Convert prime text files from t5k.org format to CSV format for E8-PRIME-DECODE.

Input format: Space-separated primes with header lines
Output format: Rank,Num,Interval
"""

import os
import sys
from pathlib import Path

def parse_prime_file(filepath):
    """Parse a single prime file, skipping header lines."""
    primes = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header lines (contain letters)
            if any(c.isalpha() for c in line):
                continue
            # Parse space-separated integers
            for num_str in line.split():
                try:
                    primes.append(int(num_str))
                except ValueError:
                    continue
    return primes

def convert_to_csv(input_dir, output_file, max_primes=None):
    """Convert all prime files to a single CSV."""
    input_path = Path(input_dir)

    # Find all prime files and sort numerically
    prime_files = sorted(
        input_path.glob('primes*.txt'),
        key=lambda x: int(x.stem.replace('primes', ''))
    )

    print(f"Found {len(prime_files)} prime files")

    all_primes = []
    for i, pfile in enumerate(prime_files):
        print(f"Reading {pfile.name}...", end=' ', flush=True)
        primes = parse_prime_file(pfile)
        print(f"{len(primes):,} primes")
        all_primes.extend(primes)

        if max_primes and len(all_primes) >= max_primes:
            all_primes = all_primes[:max_primes]
            print(f"Reached limit of {max_primes:,} primes")
            break

    # Sort to ensure order (should already be sorted)
    all_primes.sort()

    print(f"\nTotal primes: {len(all_primes):,}")
    print(f"Range: {all_primes[0]} to {all_primes[-1]}")

    # Write CSV
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        f.write('Rank,Num,Interval\n')
        for i, p in enumerate(all_primes):
            if i < len(all_primes) - 1:
                gap = all_primes[i + 1] - p
            else:
                gap = 0
            f.write(f'{i+1},{p},{gap}\n')

    print(f"Done! CSV written to {output_file}")
    return len(all_primes)

if __name__ == '__main__':
    input_dir = '/home/john/mynotes/HodgedeRham'

    if len(sys.argv) > 1:
        max_primes = int(sys.argv[1])
        output_file = f'primes_{max_primes//1000000}m.csv'
    else:
        max_primes = None
        output_file = 'primes_50m.csv'

    convert_to_csv(input_dir, output_file, max_primes)
