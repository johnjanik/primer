# Prime Visualization & E8 Analysis — Program Catalog

**42 programs** (15 C, 24 Python, 3 headers) totaling ~24,700 lines.

All C programs depend on `e8_common.h` for shared E8/F4 routines, prime sieve,
Ulam spiral, and colormaps.  Image output uses `stb_image_write.h` (public domain).

---

## 1. Core Visualizers

### `e8_viz_v3.c` (1,545 lines)
**Streaming Multi-Lattice Prime Gap Visualizer v3.0**

2-pass streaming architecture for memory-efficient rendering at any scale
(~8 GB max).  Pass 1 accumulates bounds and top-K vertex heaps; Pass 2 renders.
Supports all Lie group filters (E8, E7, E6, F4, G2, SO(16), S(16)).  Embeds
SHA-256 metadata in PNG tEXt chunks for reproducibility.

```bash
gcc -O3 -march=native -fopenmp -o e8_viz_v3 e8_viz_v3.c -lm
./e8_viz_v3 --primes 100000000 --lattice E8 --mode jordan --output e8_100M.png
```

Depends on: `e8_common.h`, `e8_metadata.h`, `stb_image_write.h`

### `e8_viz_v2.c` / `e8_viz_v2_2BPrimesMax.c` (810 lines each)
**Multi-Lattice Prime Gap Visualizer v2.0**

In-memory variant of v3; renders Ulam spiral of primes filtered by Lie group
structure.  Supports jordan, tiered, and other rendering modes.  v2_2BPrimesMax
extends prime limit to 2 billion.

```bash
gcc -O3 -march=native -fopenmp -o e8_viz_v2 e8_viz_v2.c -lm
./e8_viz_v2 --primes 50000000 --lattice F4 --mode tiered --output f4_50M.png
```

### `exceptional_grid.c` (806 lines)
**7-Panel Exceptional Chain Visualization**

Renders separate PPM images for all exceptional Lie groups: E8 (240 roots),
E7 (126), E6 (72), F4 (48), G2 (12), S16 (128 half-spinor), SO16 (112 D8).

```bash
gcc -O3 -march=native -fopenmp -o exceptional_grid exceptional_grid.c -lm
./exceptional_grid --primes 10000000
# Produces: e8_10M.ppm, e7_10M.ppm, ..., g2_10M.ppm
```

### `f4_crystalline_grid.c` (930 lines)
**F4 Crystalline Grid Visualization (PPM)**

E8 root assignment of prime gaps with F4 filtering, Jordan trace coloring
(plasma colormap), crystalline vertex extraction on Ulam spiral.

```bash
gcc -O3 -march=native -fopenmp -o f4_crystalline_grid f4_crystalline_grid.c -lm
./f4_crystalline_grid --primes 10000000 --output f4_grid.ppm
```

### `e8_f4_viz.c` (454 lines)
**F4 Crystalline Grid Visualization (compact)**

Lighter variant of `f4_crystalline_grid.c` focused on F4-filtered primes
with crystalline vertex highlighting via cosine similarity ≥ 0.7.

```bash
gcc -O3 -fopenmp -o e8_f4_viz e8_f4_viz.c -lm
./e8_f4_viz --primes 5000000
```

### `e8_slope_viz.c` (291 lines)
**E8 Projection Slope Visualization**

Produces PNG of primes in Ulam spiral colored by E8 root projection slope.
C/OpenMP replacement for the slower Python version (`e8_slope_coloring.py`).

```bash
gcc -O3 -fopenmp -o e8_slope_viz e8_slope_viz.c -lm
./e8_slope_viz --primes 100000000 --output slope_100M.png
```

---

## 2. Decoders & Path Analysis

### `path_decoder.c` (1,305 lines)
**Crystalline Path Decoder**

Extracts top-K crystalline vertices (highest triplet coherence κ from E8 root
assignments), connects them via Hamiltonian path, decodes E8/F4/G2 root
transitions and inner products along the path.

```bash
gcc -O3 -march=native -fopenmp -o path_decoder path_decoder.c -lm
./path_decoder --primes 100000000 --vertices 500 --output path_500.csv
```

### `vertex_path_decoder.py` (890 lines)
**Geodesic Decoding of the Crystalline Path (HPD algorithm)**

For consecutive crystalline vertex pairs, computes geodesic angle between E8
root vectors, maps angle → base-18 alphabet character.  Post-processing stage
for `path_decoder.c` output.

```bash
python3 vertex_path_decoder.py --input path_500.csv --output decoded_path.txt
```

### `monstrous_assembler.py` (749 lines)
**Run-Length Assembler for the Crystalline Path**

Compresses Hamiltonian path into (Root, Duration, Operator) triplets using
Phase-Length Modulation encoding.  Produces the "assembled sentence" — a
string representation of the prime sequence through E8 root transitions.

```bash
python3 monstrous_assembler.py --input path_500.csv --output sentence.txt
```

### `base18_decoder.c` (796 lines)
**EBD Pass-7 Lagrangian Decoder**

Streaming high-performance decoder: primes → E8 → base-18 symbols.  Uses mmap
checkpoints, OpenMP parallelism, and triality validation.  Designed for
100+ trillion primes.

```bash
gcc -O3 -march=native -fopenmp -o base18_decoder base18_decoder.c -lm
./base18_decoder --primes 1000000000 --output base18_1B.ebd
```

### `apsk_decoder.c` (617 lines)
**Mersenne-Sync Decoder (MSD)**

Treats prime numbers as a broadcast medium.  Mersenne primes act as
synchronization pulses; the 248 primes following each Mersenne prime form a
"listening window" whose E8 phase shifts decode to 8-bit bytes.

```bash
gcc -O3 -fopenmp -o apsk_decoder apsk_decoder.c -lm
./apsk_decoder --primes 50000000
```

### `e8_verify.c` (507 lines)
**Self-Verification Tool for E8 Visualizer PNG Metadata**

Reads PNG tEXt metadata embedded by `e8_viz_v3`, verifies integrity hash
(SHA-256), vertex path decode, and base-18 stream hash.

```bash
gcc -O3 -o e8_verify e8_verify.c -lm
./e8_verify e8_100M.png
```

Depends on: `e8_metadata.h`

---

## 3. Spectral & Statistical Verifiers

### `srv_verify.c` (1,063 lines)
**Spectral Rigidity Verifier (SRV Pass-9)**

Verifies three predicted "physical constants of arithmetic" against data up to
10^11 primes: Spectral Variance Λ_J, Monstrous Ratio R_M = gap6/gap2, and
Phase-Sync Mandala Ψ.  Streaming segmented sieve with binary mmap checkpoints.

```bash
gcc -O3 -march=native -fopenmp -o srv_verify srv_verify.c -lm
./srv_verify --primes 1000000000 --output srv_state_1B.ebd
./srv_verify --resume srv_state_1B.ebd --primes 10000000000  # resume
```

### `monstrous_governor.c` (1,011 lines)
**Monstrous Governor Scan (MGS Pass-10)**

Searches for Monster group spectral signature in prime gaps.  Four modules:
Goertzel DFT resonance at f = k/196,883; sliding variance window of width
196,883; Hardy-Littlewood residual mapping; j-function coefficient correlation.

```bash
gcc -O3 -march=native -fopenmp -o monstrous_governor monstrous_governor.c -lm
./monstrous_governor --primes 1000000000 --output mgs_state_1B.ebd
```

### `monstrous_correlator.c` (1,000 lines)
**MC Pass-8 Moonshine Decoder**

Isolates "transcendental" triplets (3 consecutive prime gaps with highly
coherent E8 root vectors, κ > 2.5), correlates spectral density with j-function
(Monster group moonshine) coefficients.

```bash
gcc -O3 -march=native -fopenmp -o monstrous_correlator monstrous_correlator.c -lm
./monstrous_correlator --primes 100000000000 --output mc_state.ebd
```

### `monstrous_linguistics.c` (740 lines)
**N-gram & Cryptographic Profiling of the F4 String (C)**

Treats F4-root-index-mod-26 encoding of prime gaps as linguistic corpus.
Computes index of coincidence, Kasiski test, Shannon entropy, N-gram
frequencies, and keyword matching.

```bash
gcc -O3 -fopenmp -o monstrous_linguistics monstrous_linguistics.c -lm
./monstrous_linguistics --primes 10000000
```

---

## 4. Python Analysis & Lattice Libraries

### `e8_prime_decode.sage.py` (996 lines)
**E8-PRIME-DECODE: Arithmetic Decoding (SageMath)**

Treats prime sequence as holographic signal.  Embeds prime gaps into E8 root
lattice, performs Exceptional Fourier Transform (EFT) over the 240-root
spectral basis.  Requires SageMath.

```bash
sage e8_prime_decode.sage.py --primes 50000000
```

### `e8_prime_decode_v2.sage.py` (715 lines)
**E8-PRIME-DECODE v2.0 (optimized SageMath)**

NumPy-vectorized variant with pre-computed lookup tables for O(1) embedding.
50-100x speedup over v1.

```bash
sage e8_prime_decode_v2.sage.py --primes 50000000
```

### `e8_prime_decoder.py` (707 lines)
**E8 Prime Decoder: Experimental Protocol**

Pure Python implementation.  Embeds 8 consecutive primes into R^8, decodes via
E8 lattice (CVP), measures errors, extracts logical bits via E8/2E8 ≅ Hamming(8,4).

```bash
python3 e8_prime_decoder.py --primes 1000000
```

### `e8_f4_prime_analysis.py` (571 lines)
**E8-F4 Prime Analysis Complete Pipeline**

Combines E8 root analysis with F4 sub-harmonic extraction.  Produces concentric
ring (E8) and discrete vertex (F4) visualizations.

```bash
python3 e8_f4_prime_analysis.py --primes 1000000
```

### `e8_multi_decoder.py` (964 lines)
**E8 Multi-Method Decoder**

Tries ALL extraction methods simultaneously: Hamming 4-bit, sign bits,
sublattice projection, parity check.  Outputs readable text for evaluation.

```bash
python3 e8_multi_decoder.py --primes 1000000
```

### `exceptional_analysis.py` (419 lines)
**Exceptional Chain Analysis: G2 < F4 < E6 < E7 < E8 + S(16)**

Multi-lattice orchestrator producing 3×2 panel visualization across the
complete exceptional Lie group chain.

```bash
python3 exceptional_analysis.py --primes 1000000
```

### `monstrous_linguistics.py` (426 lines)
**Monstrous Linguistics (Python)**

Python variant of the C version.  Same IC/Kasiski/entropy/N-gram analysis
on the F4 string encoding of prime gaps.

```bash
python3 monstrous_linguistics.py --primes 1000000
```

### `salem_jordan.py` (342 lines)
**Salem-Jordan Kernel**

Modified Salem filter for F4 sub-harmonics using the F4 character χ_F4.
Filters "topological noise" of E8 spinor sectors.

```bash
python3 salem_jordan.py --input e8_results.pkl
```

### `f4_eft.py` (346 lines)
**F4 Exceptional Fourier Transform**

Restricts E8-EFT to 48-component F4 sublattice, extracting the
Jordan-algebraic core of the prime signal.

```bash
python3 f4_eft.py --primes 1000000
```

### `jordan_algebra.py` (383 lines)
**Jordan Algebra Module — The Albert Algebra J₃(O)**

3×3 Hermitian matrices over Octonions; F4 is its automorphism group.
Library module used by other scripts.

```python
from jordan_algebra import AlbertAlgebra
```

---

## 5. Lattice Sublattice Libraries

Each constructs the root system as a set of vectors in R^8, provides membership
testing, nearest-root lookup, and inner product computation.

| File | Lattice | Roots | Rank | Dim(Lie alg) | Lines |
|------|---------|-------|------|--------------|-------|
| `g2_lattice.py` | G₂ | 12 | 2 | 14 | 306 |
| `f4_lattice.py` | F₄ | 48 | 4 | 52 | 423 |
| `e6_lattice.py` | E₆ | 72 | 6 | 78 | 321 |
| `e7_lattice.py` | E₇ | 126 | 7 | 133 | 317 |
| `s16_lattice.py` | S(16) | 128 | 8 | 120 | 284 |

All are importable as Python modules:
```python
from f4_lattice import F4Lattice
f4 = F4Lattice()
nearest_root = f4.nearest(vector)
```

---

## 6. Utilities

### `convert_primes.py` (83 lines)
Converts prime text files from t5k.org format to CSV (Rank, Num, Interval).

```bash
python3 convert_primes.py primes1.txt primes1.csv
```

### `split_primes.py` (72 lines)
Splits `all_primes.txt` into 50 files of 2M primes each in t5k.org format.

```bash
python3 split_primes.py
```

### `e8_visualizer.py` (183 lines)
Matplotlib visualization tools for E8 prime decoding results.  Loads pickle
files from the decoders.

### `e8_decoder_companion.py` (183 lines)
Companion plotting utilities for the main E8 decoder.

### `e8_decoderring.py` (426 lines)
Interactive multi-method decoder combining all extraction methods.

### `e8_slope_coloring.py` (241 lines)
Python version of `e8_slope_viz.c` — slower but useful for prototyping.

### `__init__.py` (24 lines)
Package initialization; exports lattice modules.

---

## 7. Shared Headers

### `e8_common.h` (1,225 lines)
Core shared library for all C programs:
- **E8 root system**: 240 roots, norm, inner product, nearest-root lookup
- **Segmented sieve**: streaming prime generation with OpenMP
- **Ulam spiral**: coordinate mapping for visualization
- **Colormaps**: plasma, viridis, inferno, magma palettes
- **Min-heap**: top-K extraction for crystalline vertices
- **Formatting**: `fmt_comma()`, `tic()`/`toc()` timing

### `e8_metadata.h` (391 lines)
Metadata and verification infrastructure:
- SHA-256 implementation (no OpenSSL dependency)
- Base-18 symbol mapping
- Vertex path decoder
- PNG tEXt chunk embedding

### `stb_image_write.h` (1,724 lines)
Public domain single-header image writer by Sean Barrett.
Supports PNG, BMP, TGA, JPEG, HDR output.

---

## Build (all C programs)

```bash
# Individual build
gcc -O3 -march=native -fopenmp -o PROGRAM PROGRAM.c -lm

# All programs at once
for f in *.c; do
  gcc -O3 -march=native -fopenmp -o "${f%.c}" "$f" -lm 2>/dev/null
done
```

## Dependencies

- **C**: GCC with OpenMP, standard math library (`-lm`)
- **Python**: NumPy, Matplotlib, SciPy (for most scripts)
- **SageMath**: Required only for `*.sage.py` files
- **Hardware**: 24-core Intel recommended; most programs use OpenMP
