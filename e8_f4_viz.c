/*
 * e8_f4_viz.c — F4 Crystalline Grid Visualization (C/OpenMP)
 *
 * Reproduces f4_crystalline_grid.png: F4-filtered primes in Ulam spiral,
 * colored by Jordan trace (plasma cmap), with crystalline vertices highlighted.
 *
 * Pipeline:
 *   1. Generate primes via sieve
 *   2. E8 root assignment
 *   3. F4 filtering (cosine similarity >= 0.7)
 *   4. Jordan trace computation
 *   5. F4-EFT spectrum + Salem-Jordan filter
 *   6. Crystalline vertex extraction (top 500 by F4 resonance score)
 *   7. Rasterize: F4 primes colored by Jordan trace, vertices in white
 *   8. Write PNG
 *
 * Usage:
 *   ./e8_f4_viz [--max-primes N] [--dpi D] [--size S] [--vertices V]
 *
 * Build: gcc -O3 -march=native -fopenmp -o e8_f4_viz e8_f4_viz.c -lm -lpng
 */

#include "e8_common.h"
#include <png.h>

/* ================================================================
 * Configuration
 * ================================================================ */

typedef struct {
    int64_t max_primes;
    int     dpi;
    int     fig_inches;
    int     n_vertices;
} Config;

static Config parse_args(int argc, char **argv)
{
    Config cfg = {
        .max_primes = 100000000,
        .dpi        = 1200,
        .fig_inches = 16,
        .n_vertices = 500,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-primes") && i+1 < argc) cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dpi") && i+1 < argc) cfg.dpi = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc) cfg.fig_inches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vertices") && i+1 < argc) cfg.n_vertices = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
    }
    return cfg;
}

/* ================================================================
 * PNG Writer (same as slope viz)
 * ================================================================ */

static void write_png(const char *path, const uint8_t *canvas, int width, int height)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); exit(1); }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fprintf(stderr, "png_create_write_struct failed\n"); exit(1); }
    png_infop info = png_create_info_struct(png);
    if (!info) { fprintf(stderr, "png_create_info_struct failed\n"); exit(1); }
    if (setjmp(png_jmpbuf(png))) { fprintf(stderr, "PNG write error\n"); exit(1); }

    png_init_io(png, fp);
    png_set_compression_level(png, 6);
    png_set_filter(png, 0, PNG_FILTER_SUB);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    int64_t row_bytes = (int64_t)width * 3;
    for (int y = 0; y < height; y++)
        png_write_row(png, (png_const_bytep)(canvas + y * row_bytes));

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

/* ================================================================
 * F4-EFT Spectrum Computation
 * ================================================================ */

typedef struct {
    double spectrum_re[F4_NUM_ROOTS];
    double spectrum_im[F4_NUM_ROOTS];
    double power[F4_NUM_ROOTS];
    double f4_fraction;
} F4EFTResult;

static void compute_f4_eft(
    const int *e8_assignments,  /* E8 root index per gap */
    const double *norm_gaps,    /* normalized gaps */
    int64_t n_gaps,
    const F4Lattice *f4,
    int apply_salem,            /* 1 = apply Salem-Jordan filter */
    F4EFTResult *result)
{
    memset(result->spectrum_re, 0, sizeof(result->spectrum_re));
    memset(result->spectrum_im, 0, sizeof(result->spectrum_im));

    int64_t f4_count = 0;
    double inv_n = 1.0 / (double)n_gaps;

    /* Accumulate F4-EFT (serial — 48-bin accumulation) */
    for (int64_t n = 0; n < n_gaps; n++) {
        int e8_idx = e8_assignments[n];
        if (!f4->e8_is_f4[e8_idx]) continue;

        int f4_idx = f4->e8_to_f4[e8_idx];
        if (f4_idx < 0) continue;

        f4_count++;
        double fluct = norm_gaps[n] - 1.0;
        double chi = f4->characters[f4_idx];
        double root_norm = f4->norms[f4_idx];

        double phase = 2.0 * M_PI * root_norm / sqrt(2.0);
        double time_phase = phase * (double)n * inv_n;

        double weighted = fluct * chi;
        result->spectrum_re[f4_idx] += weighted * cos(time_phase);
        result->spectrum_im[f4_idx] += weighted * sin(time_phase);
    }

    result->f4_fraction = (double)f4_count / (double)n_gaps;

    /* Optionally apply Salem-Jordan filter */
    if (apply_salem) {
        double tau = 0.5;
        for (int i = 0; i < F4_NUM_ROOTS; i++) {
            double mag = sqrt(result->spectrum_re[i] * result->spectrum_re[i]
                            + result->spectrum_im[i] * result->spectrum_im[i]);
            /* Fermi-Dirac: 1/(exp(mag/tau) + 1) */
            double x = mag / tau;
            if (x > 50.0) x = 50.0;
            if (x < -50.0) x = -50.0;
            double fermi = 1.0 / (exp(x) + 1.0);
            /* F4 character: (52 - 4*mag^2) / 52 */
            double chi_norm = (52.0 - 4.0 * mag * mag) / 52.0;
            double kernel = chi_norm * fermi;
            result->spectrum_re[i] *= kernel;
            result->spectrum_im[i] *= kernel;
        }
    }

    /* Power spectrum */
    for (int i = 0; i < F4_NUM_ROOTS; i++) {
        result->power[i] = result->spectrum_re[i] * result->spectrum_re[i]
                         + result->spectrum_im[i] * result->spectrum_im[i];
    }
}

typedef struct { int64_t idx; double score; } ScoreEntry;

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);
    int canvas_px = cfg.dpi * cfg.fig_inches;

    char buf1[32];
    printf("F4 Crystalline Grid Visualization (C/OpenMP)\n");
    printf("==================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, buf1, sizeof(buf1)));
    printf("Canvas     : %d x %d px (%d DPI x %d in)\n", canvas_px, canvas_px, cfg.dpi, cfg.fig_inches);
    printf("Vertices   : %d\n", cfg.n_vertices);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("==================================================\n\n");

    /* ---- Step 1: Generate primes ---- */
    printf("Generating primes via sieve...\n");
    tic();
    int64_t n_primes;
    int64_t *primes = sieve_primes(cfg.max_primes, &n_primes);
    printf("  Generated %s primes in %.2fs\n", fmt_comma(n_primes, buf1, sizeof(buf1)), toc());
    printf("  Range: 2 to %s\n", fmt_comma(primes[n_primes - 1], buf1, sizeof(buf1)));

    int64_t n_gaps = n_primes - 1;

    /* ---- Step 2: Initialize E8 + F4 ---- */
    E8Lattice e8;
    e8_init(&e8);
    F4Lattice f4;
    f4_init(&f4, &e8);

    /* Count how many E8 roots map to F4 */
    int e8_f4_count = 0;
    for (int i = 0; i < E8_NUM_ROOTS; i++) if (f4.e8_is_f4[i]) e8_f4_count++;
    printf("  E8 roots mapping to F4: %d/240\n", e8_f4_count);

    /* ---- Step 3: Compute E8 assignments + normalized gaps ---- */
    printf("Computing E8/F4 assignments (%s gaps)...\n", fmt_comma(n_gaps, buf1, sizeof(buf1)));
    tic();

    int *e8_assignments = (int *)malloc(n_gaps * sizeof(int));
    double *norm_gaps = (double *)malloc(n_gaps * sizeof(double));
    if (!e8_assignments || !norm_gaps) { fprintf(stderr, "malloc failed\n"); exit(1); }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i + 1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        norm_gaps[i] = gap / log_p;
        e8_assignments[i] = e8_assign_root(&e8, norm_gaps[i]);
    }

    /* Compute per-prime data: is_f4, jordan_trace */
    /* Index convention: prime[i+1] gets data from gap[i] */
    uint8_t *is_f4_prime = (uint8_t *)calloc(n_primes, 1);
    float *jordan_vals = (float *)calloc(n_primes, sizeof(float));

    int64_t f4_prime_count = 0;
    #pragma omp parallel for schedule(static) reduction(+:f4_prime_count)
    for (int64_t i = 0; i < n_gaps; i++) {
        int e8_idx = e8_assignments[i];
        if (f4.e8_is_f4[e8_idx]) {
            int f4_idx = f4.e8_to_f4[e8_idx];
            if (f4_idx >= 0) {
                is_f4_prime[i + 1] = 1;
                jordan_vals[i + 1] = (float)f4.jordan_traces[f4_idx];
                f4_prime_count++;
            }
        }
    }
    printf("  Done in %.2fs\n", toc());
    printf("  F4-mapped gaps: %s (%.1f%%)\n",
           fmt_comma(f4_prime_count, buf1, sizeof(buf1)),
           100.0 * f4_prime_count / n_gaps);

    /* ---- Step 4: Compute F4-EFT + crystalline vertices ---- */
    printf("Computing F4-EFT spectrum...\n");
    tic();
    F4EFTResult eft;
    compute_f4_eft(e8_assignments, norm_gaps, n_gaps, &f4, 1 /* apply Salem */, &eft);
    printf("  F4 fraction: %.2f%%\n", eft.f4_fraction * 100.0);
    printf("  Done in %.2fs\n", toc());

    printf("Extracting crystalline vertices (top %d)...\n", cfg.n_vertices);
    tic();

    /* Score each gap by its F4 power spectrum contribution */
    ScoreEntry *scores = (ScoreEntry *)malloc(n_gaps * sizeof(ScoreEntry));

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_gaps; i++) {
        scores[i].idx = i;
        scores[i].score = 0.0;
        int e8_idx = e8_assignments[i];
        if (!f4.e8_is_f4[e8_idx]) continue;
        int f4_idx = f4.e8_to_f4[e8_idx];
        if (f4_idx < 0) continue;

        scores[i].score = eft.power[f4_idx];

        /* Boost idempotent-type roots (|jordan_trace| ≈ 1) */
        double jt = f4.jordan_traces[f4_idx];
        if (fabs(fabs(jt) - 1.0) < 0.2)
            scores[i].score *= 2.0;
    }

    /* Partial selection: keep top N scores using a min-heap of size N.
     * O(n_gaps * log(N)) instead of O(n_gaps * log(n_gaps)) for full sort. */
    int heap_cap = cfg.n_vertices * 2;  /* oversize to account for non-F4 */
    if (heap_cap > n_gaps) heap_cap = (int)n_gaps;
    ScoreEntry *heap = (ScoreEntry *)malloc(heap_cap * sizeof(ScoreEntry));
    int heap_size = 0;

    for (int64_t i = 0; i < n_gaps; i++) {
        if (scores[i].score <= 0.0) continue;
        if (heap_size < heap_cap) {
            heap[heap_size++] = scores[i];
            /* Bubble up (min-heap by score) */
            int k = heap_size - 1;
            while (k > 0) {
                int parent = (k - 1) / 2;
                if (heap[parent].score > heap[k].score) {
                    ScoreEntry tmp = heap[parent]; heap[parent] = heap[k]; heap[k] = tmp;
                    k = parent;
                } else break;
            }
        } else if (scores[i].score > heap[0].score) {
            heap[0] = scores[i];
            /* Sift down */
            int k = 0;
            for (;;) {
                int left = 2*k+1, right = 2*k+2, smallest = k;
                if (left < heap_size && heap[left].score < heap[smallest].score) smallest = left;
                if (right < heap_size && heap[right].score < heap[smallest].score) smallest = right;
                if (smallest == k) break;
                ScoreEntry tmp = heap[k]; heap[k] = heap[smallest]; heap[smallest] = tmp;
                k = smallest;
            }
        }
    }

    uint8_t *is_vertex = (uint8_t *)calloc(n_primes, 1);
    int actual_vertices = 0;
    for (int i = 0; i < heap_size && actual_vertices < cfg.n_vertices; i++) {
        int64_t gap_idx = heap[i].idx;
        int64_t prime_idx = gap_idx + 1;
        if (prime_idx < n_primes && is_f4_prime[prime_idx]) {
            is_vertex[prime_idx] = 1;
            actual_vertices++;
        }
    }
    free(heap);
    printf("  Marked %d crystalline vertices in %.2fs\n", actual_vertices, toc());
    free(scores);

    /* ---- Step 5: Compute Ulam coordinates ---- */
    printf("Computing Ulam spiral coordinates...\n");
    tic();
    int32_t *coord_x = (int32_t *)malloc(n_primes * sizeof(int32_t));
    int32_t *coord_y = (int32_t *)malloc(n_primes * sizeof(int32_t));

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_primes; i++) {
        ulam_coord(primes[i], &coord_x[i], &coord_y[i]);
    }
    printf("  Done in %.2fs\n", toc());

    /* Coordinate range (only for F4 primes) */
    int32_t min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    int first = 1;
    for (int64_t i = 1; i < n_primes; i++) {
        if (!is_f4_prime[i]) continue;
        if (first) { min_x = max_x = coord_x[i]; min_y = max_y = coord_y[i]; first = 0; continue; }
        if (coord_x[i] < min_x) min_x = coord_x[i];
        if (coord_x[i] > max_x) max_x = coord_x[i];
        if (coord_y[i] < min_y) min_y = coord_y[i];
        if (coord_y[i] > max_y) max_y = coord_y[i];
    }
    printf("  F4 coord range: x=[%d, %d] y=[%d, %d]\n", min_x, max_x, min_y, max_y);

    /* ---- Step 6: Rasterize ---- */
    printf("Rasterizing %s F4 primes to %dx%d canvas...\n",
           fmt_comma(f4_prime_count, buf1, sizeof(buf1)), canvas_px, canvas_px);
    tic();

    int64_t canvas_bytes = (int64_t)canvas_px * canvas_px * 3;
    uint8_t *canvas = (uint8_t *)calloc(canvas_bytes, 1);  /* black background */
    if (!canvas) {
        fprintf(stderr, "Failed to allocate canvas (%.1f GB)\n", canvas_bytes / 1e9);
        exit(1);
    }

    double range_x = (double)(max_x - min_x);
    double range_y = (double)(max_y - min_y);
    double range = fmax(range_x, range_y);
    double margin_frac = 0.02;
    double scale = (canvas_px * (1.0 - 2.0 * margin_frac)) / range;
    double offset_x = canvas_px * margin_frac - min_x * scale + (range - range_x) * scale * 0.5;
    double offset_y = canvas_px * margin_frac - min_y * scale + (range - range_y) * scale * 0.5;

    /* Pass 1: Plot F4 primes colored by Jordan trace */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n_primes; i++) {
        if (!is_f4_prime[i]) continue;
        int px = (int)(coord_x[i] * scale + offset_x);
        int py = (int)(coord_y[i] * scale + offset_y);
        py = canvas_px - 1 - py;
        if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;
        RGB c = jordan_to_color((double)jordan_vals[i]);
        int64_t off = ((int64_t)py * canvas_px + px) * 3;
        canvas[off + 0] = c.r;
        canvas[off + 1] = c.g;
        canvas[off + 2] = c.b;
    }

    /* Pass 2: Overlay crystalline vertices in white (larger dot) */
    for (int64_t i = 1; i < n_primes; i++) {
        if (!is_vertex[i]) continue;
        int cx = (int)(coord_x[i] * scale + offset_x);
        int cy = (int)(coord_y[i] * scale + offset_y);
        cy = canvas_px - 1 - cy;

        /* Draw a small filled circle (radius ~2-3 px at 1200 DPI) */
        int radius = cfg.dpi >= 600 ? 3 : 2;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx*dx + dy*dy > radius*radius) continue;
                int px = cx + dx;
                int py = cy + dy;
                if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;
                int64_t off = ((int64_t)py * canvas_px + px) * 3;
                /* White center, yellow edge */
                if (dx*dx + dy*dy <= (radius-1)*(radius-1)) {
                    canvas[off + 0] = 255;
                    canvas[off + 1] = 255;
                    canvas[off + 2] = 255;
                } else {
                    canvas[off + 0] = 255;
                    canvas[off + 1] = 255;
                    canvas[off + 2] = 0;
                }
            }
        }
    }
    printf("  Rasterized in %.2fs\n", toc());

    /* ---- Step 7: Write PNG ---- */
    (void)!system("mkdir -p /home/john/mynotes/HodgedeRham/spiral_outputs");

    char suffix[64];
    if (cfg.max_primes >= 1000000)
        snprintf(suffix, sizeof(suffix), "%ldM", cfg.max_primes / 1000000);
    else
        snprintf(suffix, sizeof(suffix), "%ldk", cfg.max_primes / 1000);

    char png_path[512];
    snprintf(png_path, sizeof(png_path),
             "/home/john/mynotes/HodgedeRham/spiral_outputs/f4_crystalline_grid_%s.png", suffix);
    printf("Writing PNG (%dx%d)...\n", canvas_px, canvas_px);
    tic();
    write_png(png_path, canvas, canvas_px, canvas_px);
    FILE *check = fopen(png_path, "rb");
    long png_size = 0;
    if (check) { fseek(check, 0, SEEK_END); png_size = ftell(check); fclose(check); }
    printf("  Written %s (%.1f MB) in %.2fs\n", png_path, png_size / 1e6, toc());

    /* ---- Summary ---- */
    printf("\n==================================================\n");
    printf("F4 Crystalline Grid\n");
    printf("  %s F4 primes (%.1f%% of gaps)\n",
           fmt_comma(f4_prime_count, buf1, sizeof(buf1)),
           100.0 * f4_prime_count / n_gaps);
    printf("  %d crystalline vertices\n", actual_vertices);
    printf("Output: %s\n", png_path);
    printf("==================================================\n");

    free(canvas);
    free(is_vertex);
    free(is_f4_prime);
    free(jordan_vals);
    free(e8_assignments);
    free(norm_gaps);
    free(coord_x);
    free(coord_y);
    free(primes);

    return 0;
}
