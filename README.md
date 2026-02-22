# E8 Prime Visualization & Analysis Toolkit

**42 programs** (15 C, 24 Python, 3 headers) totaling ~26,400 lines of source code.

This toolkit embeds prime gap sequences into the E8 root lattice and its
exceptional sub-lattices (F4, E6, E7, G2, S(16), SO(16)), producing
large-scale visualizations, spectral analyses, and algebraic decoders.
The central idea: consecutive prime gaps define vectors in R^8 whose
nearest E8 roots carry rich arithmetic structure visible in Ulam-spiral
renderings at scales of 10^8--10^11 primes.

---

## Mathematical Background

### The E8 Root Lattice

E8 is the unique even unimodular lattice in 8 dimensions.  Its 240 minimal
vectors (roots) decompose into:

- **Type I** (112 roots): +/-e_i +/- e_j for 1 <= i < j <= 8
- **Type II** (128 roots): (1/2)(+/-1, +/-1, ..., +/-1) with an even
  number of minus signs

All roots have norm sqrt(2).  The automorphism group |W(E8)| = 696,729,600
is the largest exceptional Weyl group.

### Prime Gap Embedding

Given 8 consecutive primes p_n, ..., p_{n+7}, the gap vector
g = (p_{n+1}-p_n, p_{n+2}-p_{n+1}, ..., p_{n+7}-p_{n+6}) is a point in
R^8.  The **E8 root assignment** maps g to its nearest E8 root r(g) via
closest-vector projection.  The Jordan trace Tr(r) = sum of coordinates
and the projection slope atan2(r_2, r_1) encode gap magnitude and
directional structure.

### Exceptional Lie Group Chain

The programs explore the full exceptional chain and its branching:

```
G2(12) < F4(48) < E6(72) < E7(126) < E8(240)
                                         |
                                       S(16)(128)  [half-spinor]
                                       SO(16)(112)  [D8 roots]
```

Numbers in parentheses are root counts.  F4 is the automorphism group
of the exceptional Jordan algebra J_3(O) (3x3 Hermitian octonion
matrices); G2 is the automorphism group of the octonions.

### Spectral Predictions

The E8 Diamond framework predicts three "physical constants of
arithmetic" governing normalized prime gaps g~ = (p_{n+1}-p_n)/ln(p_n):

1. **Spectral Variance** Lambda_J: stabilizes near the E8-predicted value
2. **Monstrous Ratio** R_M = gap_6/gap_2: ratio of 6th to 2nd gap moments
3. **Phase-Sync Mandala** Psi: E8 phase coherence measure

These are verified numerically to 10^11 primes by `srv_verify.c`.

---

## Repository Structure

```
primer/
|-- *.c                 15 C source files (high-performance pipeline)
|-- *.py                24 Python scripts (analysis, lattice libraries)
|-- *.h                  3 shared headers
|-- PROGRAM_CATALOG.md   Detailed per-program documentation
|-- README.md            This file
|-- __init__.py          Package initialization
|-- docs/
|   |-- spectral_rigidity-of_primes.tex   "The Singular Series as Classical Limit"
|   |-- e8_prime_structure.tex            "Prime Gap Structure via Exceptional Lattice Projections"
|   |-- algebraic_geometry_e8_synthesis_v1.tex  "The E8 Lattice Framework and Arithmetic Geometry"
|   |-- entropy_production_rate.tex       "The Entropy Production Rate of the Prime Flow"
|   |-- lean_formalization_notes.md       Lean 4 formalization roadmap
|   `-- spiral_outputs/                   Saved output logs (SRV 100B, etc.)
```

---

## Build

All C programs share the same build command:

```bash
gcc -O3 -march=native -fopenmp -o PROGRAM PROGRAM.c -lm
```

Build everything at once:

```bash
for f in *.c; do
  gcc -O3 -march=native -fopenmp -o "${f%.c}" "$f" -lm 2>/dev/null && \
    echo "  OK  $f" || echo "FAIL  $f"
done
```

### Dependencies

| Component | Requirement |
|-----------|-------------|
| C compiler | GCC with OpenMP (`-fopenmp`) |
| Math library | `-lm` (standard) |
| Python | 3.8+ with NumPy, Matplotlib, SciPy |
| SageMath | Required only for `*.sage.py` files |
| Hardware | 24-core recommended; all C programs use OpenMP |

No external C libraries are needed.  Image output uses the bundled
`stb_image_write.h` (public domain, Sean Barrett).

---

## Pipeline Overview

The toolkit forms a pipeline from raw prime generation through
visualization, decoding, and statistical verification:

```
 GENERATE          EMBED & RENDER         DECODE            VERIFY
+---------+     +----------------+     +-----------+     +----------+
| Sieve   | --> | E8 root assign | --> | Path      | --> | Spectral |
| (OpenMP |     | Ulam spiral    |     | Decoder   |     | Rigidity |
|  seg.)  |     | Lie group      |     | Base-18   |     | Monster  |
|         |     | filter         |     | APSK      |     | Governor |
+---------+     +----------------+     +-----------+     +----------+
  e8_common.h    e8_viz_v3.c           path_decoder.c    srv_verify.c
  (sieve)        exceptional_grid.c    base18_decoder.c   monstrous_governor.c
                 f4_crystalline_grid.c  apsk_decoder.c    monstrous_correlator.c
                 e8_slope_viz.c         vertex_path_*.py  monstrous_linguistics.c
```

---

## Quick Start

### 1. Visualize 100M primes through the E8 lattice

```bash
gcc -O3 -march=native -fopenmp -o e8_viz_v3 e8_viz_v3.c -lm
./e8_viz_v3 --primes 100000000 --lattice E8 --mode jordan --output e8_100M.png
```

Produces a 4000x4000 PNG of primes on the Ulam spiral, colored by Jordan
trace of their E8 root assignment.  Plasma colormap.  Runtime: ~30 seconds
on 24 cores.

### 2. Render the full exceptional chain

```bash
gcc -O3 -march=native -fopenmp -o exceptional_grid exceptional_grid.c -lm
./exceptional_grid --primes 10000000
# Produces: e8_10M.ppm, e7_10M.ppm, e6_10M.ppm, f4_10M.ppm, g2_10M.ppm, s16_10M.ppm, so16_10M.ppm
```

Seven separate PPM images, one per exceptional Lie group.

### 3. Extract and decode the crystalline path

```bash
gcc -O3 -march=native -fopenmp -o path_decoder path_decoder.c -lm
./path_decoder --primes 100000000 --vertices 500 --output path_500.csv

python3 vertex_path_decoder.py --input path_500.csv --output decoded_path.txt
python3 monstrous_assembler.py --input path_500.csv --output sentence.txt
```

Extracts the top-500 crystalline vertices (highest triplet coherence kappa),
connects via Hamiltonian path, decodes root transitions to a base-18 alphabet.

### 4. Verify spectral predictions to 10^9

```bash
gcc -O3 -march=native -fopenmp -o srv_verify srv_verify.c -lm
./srv_verify --primes 1000000000 --output srv_state_1B.ebd
```

Streaming segmented sieve with binary checkpoints.  Tests Lambda_J, R_M,
and Psi against predicted values.  Resume with `--resume srv_state_1B.ebd`.

### 5. Python lattice analysis

```bash
python3 e8_f4_prime_analysis.py --primes 1000000
python3 exceptional_analysis.py --primes 1000000
```

Produces Matplotlib figures: concentric ring (E8) and discrete vertex (F4)
visualizations, multi-panel exceptional chain analysis.

---

## Programs by Category

### Core Visualizers (6 programs)

| Program | Lines | Description |
|---------|-------|-------------|
| `e8_viz_v3.c` | 1,545 | Streaming 2-pass renderer, all Lie groups, PNG with SHA-256 metadata |
| `e8_viz_v2.c` | 810 | In-memory variant, up to ~50M primes |
| `e8_viz_v2_2BPrimesMax.c` | 810 | Extended to 2 billion primes |
| `exceptional_grid.c` | 806 | 7-panel PPM output for all exceptional groups |
| `f4_crystalline_grid.c` | 930 | F4-filtered crystalline vertex extraction |
| `e8_slope_viz.c` | 291 | Projection slope coloring (C replacement for Python) |

### Decoders & Path Analysis (7 programs)

| Program | Lines | Description |
|---------|-------|-------------|
| `path_decoder.c` | 1,305 | Crystalline path extraction + Hamiltonian connection |
| `vertex_path_decoder.py` | 890 | Geodesic angle decoding (HPD algorithm) |
| `monstrous_assembler.py` | 749 | Run-length Phase-Length Modulation encoding |
| `base18_decoder.c` | 796 | Streaming base-18 decoder with mmap checkpoints |
| `apsk_decoder.c` | 617 | Mersenne-sync decoder (MSD) |
| `e8_verify.c` | 507 | PNG metadata integrity verification |
| `e8_f4_viz.c` | 454 | Compact F4 crystalline visualization |

### Spectral & Statistical Verifiers (4 programs)

| Program | Lines | Description |
|---------|-------|-------------|
| `srv_verify.c` | 1,063 | Lambda_J, R_M, Psi verification to 10^11 |
| `monstrous_governor.c` | 1,011 | Monster group spectral signature search (Goertzel DFT at f=k/196883) |
| `monstrous_correlator.c` | 1,000 | Moonshine j-function coefficient correlation |
| `monstrous_linguistics.c` | 740 | N-gram, IC, Kasiski, Shannon entropy profiling |

### Python Analysis & Transforms (12 programs)

| Program | Lines | Description |
|---------|-------|-------------|
| `e8_prime_decode.sage.py` | 996 | SageMath holographic signal decoder |
| `e8_multi_decoder.py` | 964 | All extraction methods simultaneously |
| `e8_prime_decode_v2.sage.py` | 715 | NumPy-vectorized SageMath decoder (50-100x faster) |
| `e8_prime_decoder.py` | 707 | Pure Python CVP decoder, Hamming(8,4) bits |
| `e8_f4_prime_analysis.py` | 571 | Complete E8-F4 analysis pipeline |
| `e8_decoderring.py` | 426 | Interactive multi-method decoder |
| `monstrous_linguistics.py` | 426 | Python variant of C linguistics |
| `exceptional_analysis.py` | 419 | 3x2 panel exceptional chain analysis |
| `jordan_algebra.py` | 383 | Albert algebra J_3(O) library |
| `f4_eft.py` | 346 | F4 Exceptional Fourier Transform |
| `salem_jordan.py` | 342 | Modified Salem filter for F4 sub-harmonics |
| `e8_slope_coloring.py` | 241 | Prototype slope visualizer (Matplotlib) |

### Lattice Sub-Libraries (5 modules)

| Module | Lattice | Roots | Rank | Lines |
|--------|---------|-------|------|-------|
| `f4_lattice.py` | F4 | 48 | 4 | 423 |
| `e6_lattice.py` | E6 | 72 | 6 | 321 |
| `e7_lattice.py` | E7 | 126 | 7 | 317 |
| `g2_lattice.py` | G2 | 12 | 2 | 306 |
| `s16_lattice.py` | S(16) | 128 | 8 | 284 |

Each provides: root system construction, membership test, nearest-root
lookup, inner product computation.  Import as:

```python
from f4_lattice import F4Lattice
f4 = F4Lattice()
nearest = f4.nearest(vector)
```

### Utilities (5 programs)

| Program | Lines | Description |
|---------|-------|-------------|
| `e8_visualizer.py` | 183 | Matplotlib visualization for decoder results |
| `e8_decoder_companion.py` | 183 | Companion plotting utilities |
| `convert_primes.py` | 83 | t5k.org format to CSV |
| `split_primes.py` | 72 | Split prime files into 2M-prime chunks |
| `__init__.py` | 24 | Package initialization |

---

## Shared Headers

### `e8_common.h` (1,225 lines)

Core library included by all C programs:

- **E8 root system**: 240 roots in R^8 (112 Type I + 128 Type II), nearest-root
  lookup via brute-force inner product, norm, slope computation
- **Segmented sieve**: streaming prime generation with OpenMP parallelism,
  handles up to 10^11+ primes with bounded memory
- **Ulam spiral**: integer-to-spiral coordinate mapping for visualization
- **Colormaps**: plasma, viridis, inferno, magma palettes (256-entry LUTs)
- **Min-heap**: top-K extraction for crystalline vertex selection
- **Formatting**: `fmt_comma()` for human-readable numbers, `tic()`/`toc()`
  for wall-clock timing via `omp_get_wtime()`

### `e8_metadata.h` (391 lines)

Metadata and verification infrastructure:

- SHA-256 implementation (no OpenSSL dependency)
- Base-18 symbol alphabet mapping
- Vertex path decoder routines
- PNG tEXt chunk embedding for reproducibility

### `stb_image_write.h` (1,724 lines)

Public domain single-header image writer by Sean Barrett.  Supports
PNG, BMP, TGA, JPEG, HDR output.  Used by all visualizers that produce
PNG output.

---

## Rendering Modes

The visualizers support several coloring modes selected via `--mode`:

| Mode | Description |
|------|-------------|
| `jordan` | Color by Jordan trace Tr(r) = sum of E8 root coordinates |
| `tiered` | Discrete color tiers based on gap magnitude |
| `slope` | Projection slope atan2(r_2, r_1) of E8 root |
| `crystalline` | Highlight high-coherence triplet vertices (kappa > threshold) |

All modes render primes on the Ulam spiral.  Non-prime positions are
black.  F4-filtered modes show only primes whose E8 root assignment
lies in the 48-root F4 sub-lattice.

---

## Streaming Architecture (v3)

`e8_viz_v3.c` uses a 2-pass streaming design for memory efficiency:

- **Pass 1**: Segmented sieve generates primes in blocks.  Accumulates
  global bounds (min/max Jordan trace) and top-K vertex heaps.  Memory:
  O(sqrt(N)) for sieve + O(K) for heaps.  No image buffer allocated.

- **Pass 2**: Re-sieves from scratch.  With bounds known, maps each prime
  to a pixel color and writes directly to the image buffer.  Memory:
  O(W^2) for image + O(sqrt(N)) for sieve.

This allows rendering up to ~8 GB images (32K x 32K at 8 bytes/pixel)
with bounded working memory.

---

## Checkpoint & Resume

Long-running programs (`srv_verify.c`, `monstrous_governor.c`,
`monstrous_correlator.c`) support binary checkpoints:

```bash
# Start a 10B verification
./srv_verify --primes 10000000000 --output srv_state.ebd

# Resume after interruption
./srv_verify --resume srv_state.ebd --primes 100000000000
```

Checkpoints are saved atomically (write to `.tmp`, then rename) to
survive power failures.  State includes sieve position, all accumulated
statistics, and verification counters.

---

## Companion Papers

| Paper | File |
|-------|------|
| The Singular Series as Classical Limit | `docs/spectral_rigidity-of_primes.tex` |
| Prime Gap Structure via Exceptional Lattice Projections | `docs/e8_prime_structure.tex` |
| The E8 Lattice Framework and Arithmetic Geometry | `docs/algebraic_geometry_e8_synthesis_v1.tex` |
| The Entropy Production Rate of the Prime Flow | `docs/entropy_production_rate.tex` |

---

## Lean 4 Formalization

A parallel Lean 4 formalization effort verifies key claims constructively.
See `docs/lean_formalization_notes.md` for the full roadmap.  Key files
in the companion Lean 4 project:

- **E8Lattice.lean**: Constructive E8 root system, 3 axioms proved
- **Lehmer.lean**: Lehmer's polynomial, E10 Coxeter connection, Smyth bound
  (14 theorems by `native_decide`)
- **RiemannHypothesis.lean**: Hodge-de Rham approach via Salem integrals

---

## Performance Notes

Tested on Intel Core Ultra 9 275HX (24 cores, no HT), 128 GB DDR5:

| Program | Scale | Time | Throughput |
|---------|-------|------|------------|
| `e8_viz_v3` | 100M primes | ~30s | ~3.3M primes/s |
| `srv_verify` | 1B primes | ~5 min | ~3.3M primes/s |
| `srv_verify` | 100B primes | ~8 hrs | ~3.5M primes/s |
| `path_decoder` | 100M, K=500 | ~45s | ~2.2M primes/s |
| `exceptional_grid` | 10M primes | ~10s | ~1M primes/s |

The segmented sieve in `e8_common.h` scales linearly with core count
up to ~20 cores; beyond that, memory bandwidth for the Ulam spiral
coordinate mapping becomes the bottleneck.

---

## License

The source code in this repository is the work of John A. Janik.
`stb_image_write.h` is public domain (Sean Barrett).

---

## Author

**John A. Janik**
john.janik@gmail.com
