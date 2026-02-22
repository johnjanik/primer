/*
 * e8_viz_v3.c — Streaming Multi-Lattice Prime Gap Visualizer v3.0
 *
 * 2-pass streaming rewrite of v2: memory stays under ~8GB for any prime count.
 *
 * Architecture:
 *   Pass 1: Stream all sieve chunks, accumulate per-(lattice,mode) coordinate
 *           bounds and top-K vertex heaps.
 *   Pass 2: Reset sieve, stream again, render directly to pre-allocated canvases
 *           using bounds from Pass 1.
 *
 * Root systems: E8 (240), E7 (126), E6 (72), F4 (48), G2 (12), SO(16) (128)
 *
 * Rendering modes:
 *   jordan  — Classic: color by Jordan trace (plasma colormap)
 *   tiered  — Color by triplet coherence tier (Noise/Harmonic/Resonant/Transcendental)
 *   strobe  — Stroboscopic filter at Monster frequency (1/196,883)
 *   torsion — Color by projection quality (non-sublattice leakage)
 *
 * Features:
 *   - Crystalline vertex extraction + Hamiltonian path overlay (--path)
 *   - Multi-lattice: generates one PNG per selected lattice
 *   - Triplet coherence computation for tiered mode
 *   - SieveIter: replayable segmented sieve, 50M primes per chunk
 *
 * Build: gcc -O3 -march=native -fopenmp -o e8_viz_v3 e8_viz_v3.c -lm -lpng
 */

#include "e8_common.h"
#include "e8_metadata.h"
#include <png.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* ================================================================
 * Configuration
 * ================================================================ */

#define MONSTER_DIM 196883
#define CHUNK_SIZE  50000000   /* 50M primes per chunk */

enum RenderMode { MODE_JORDAN = 0, MODE_TIERED, MODE_STROBE, MODE_TORSION, MODE_COUNT };
enum LatticeID  { LAT_E8 = 0, LAT_E7, LAT_E6, LAT_F4, LAT_G2, LAT_SO16, LAT_COUNT };

static const char *mode_names[]    = {"jordan", "tiered", "strobe", "torsion"};
static const char *lattice_names[] = {"E8", "E7", "E6", "F4", "G2", "SO16"};

typedef struct {
    int64_t max_primes;
    int     dpi;
    int     fig_inches;
    int     n_vertices;
    int     mode;           /* -1 = all */
    int     lattice;        /* -1 = all */
    int     draw_path;      /* 1 = overlay Hamiltonian path lines */
    int     strobe_epsilon; /* half-width of strobe window */
    int     strobe_phase;   /* phase offset for strobe */
    char    output_dir[512];
} Config;

static Config parse_args(int argc, char **argv)
{
    Config cfg = {
        .max_primes      = 10000000000LL,
        .dpi             = 600,
        .fig_inches      = 16,
        .n_vertices      = 500,
        .mode            = -1,
        .lattice         = -1,
        .draw_path       = 0,
        .strobe_epsilon  = 100,
        .strobe_phase    = 0,
    };
    strncpy(cfg.output_dir, "/home/john/mynotes/HodgedeRham/spiral_outputs", sizeof(cfg.output_dir) - 1);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-primes") && i+1 < argc) cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dpi") && i+1 < argc) cfg.dpi = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc) cfg.fig_inches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vertices") && i+1 < argc) cfg.n_vertices = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--path")) cfg.draw_path = 1;
        else if (!strcmp(argv[i], "--strobe-epsilon") && i+1 < argc) cfg.strobe_epsilon = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--strobe-phase") && i+1 < argc) cfg.strobe_phase = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) strncpy(cfg.output_dir, argv[++i], sizeof(cfg.output_dir) - 1);
        else if (!strcmp(argv[i], "--mode") && i+1 < argc) {
            i++;
            if (!strcmp(argv[i], "all")) cfg.mode = -1;
            else {
                for (int m = 0; m < MODE_COUNT; m++) {
                    if (!strcmp(argv[i], mode_names[m])) { cfg.mode = m; break; }
                }
            }
        }
        else if (!strcmp(argv[i], "--lattice") && i+1 < argc) {
            i++;
            if (!strcmp(argv[i], "all") || !strcmp(argv[i], "ALL")) cfg.lattice = -1;
            else {
                for (int l = 0; l < LAT_COUNT; l++) {
                    if (!strcasecmp(argv[i], lattice_names[l])) { cfg.lattice = l; break; }
                }
            }
        }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
    }
    return cfg;
}

/* ================================================================
 * Generic Lattice View
 * ================================================================ */

typedef struct {
    const char *name;
    int         n_roots;
    const int  *e8_is_member;   /* [E8_NUM_ROOTS]: 1 if E8 root is in this lattice */
    const int  *e8_to_idx;      /* [E8_NUM_ROOTS]: E8 root -> sublattice index (-1 if none) */
    const double *jordan_traces;/* [n_roots] */
    const double *norms;        /* [n_roots] */
    double      jmin, jmax;     /* Jordan trace range for color scaling */
    double      quality[E8_NUM_ROOTS]; /* cosine similarity to nearest sublattice root */
} LatticeView;

/* Compute Jordan trace range and quality scores */
static void lattice_view_finalize(LatticeView *lv, const E8Lattice *e8)
{
    /* Jordan trace range */
    lv->jmin = 1e30;
    lv->jmax = -1e30;
    for (int i = 0; i < lv->n_roots; i++) {
        double j = lv->jordan_traces[i];
        if (j < lv->jmin) lv->jmin = j;
        if (j > lv->jmax) lv->jmax = j;
    }

    /* Quality: for E8 itself, all roots are exact members */
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        if (lv->e8_is_member[ei]) {
            lv->quality[ei] = 1.0;
        } else if (lv->e8_to_idx[ei] >= 0) {
            lv->quality[ei] = 0.5;  /* placeholder; overridden per lattice below */
        } else {
            lv->quality[ei] = 0.0;
        }
    }
}

/* ================================================================
 * PNG Writer
 * ================================================================ */

/* write_png_meta: Write PNG with embedded tEXt/zTXt metadata chunks */
static void write_png_meta(const char *path, const uint8_t *canvas,
                            int width, int height, const MetadataBundle *meta)
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

    /* Embed metadata as tEXt/zTXt chunks */
    png_text text_chunks[6];
    memset(text_chunks, 0, sizeof(text_chunks));
    int n_chunks = 0;

    /* params: compressed (can be large) */
    text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_zTXt;
    text_chunks[n_chunks].key = (png_charp)"e8v3:params";
    text_chunks[n_chunks].text = (png_charp)meta->params;
    text_chunks[n_chunks].text_length = strlen(meta->params);
    n_chunks++;

    /* stats: compressed */
    text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_zTXt;
    text_chunks[n_chunks].key = (png_charp)"e8v3:stats";
    text_chunks[n_chunks].text = (png_charp)meta->stats;
    text_chunks[n_chunks].text_length = strlen(meta->stats);
    n_chunks++;

    /* vertices: compressed (largest chunk, ~15KB) */
    if (meta->vertices && meta->vertices[0]) {
        text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_zTXt;
        text_chunks[n_chunks].key = (png_charp)"e8v3:vertices";
        text_chunks[n_chunks].text = (png_charp)meta->vertices;
        text_chunks[n_chunks].text_length = strlen(meta->vertices);
        n_chunks++;
    }

    /* vertex_decode: uncompressed (short string) */
    text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_NONE;
    text_chunks[n_chunks].key = (png_charp)"e8v3:vertex_decode";
    text_chunks[n_chunks].text = (png_charp)meta->vertex_decode;
    text_chunks[n_chunks].text_length = strlen(meta->vertex_decode);
    n_chunks++;

    /* base18_hash: uncompressed (64 hex chars) */
    text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_NONE;
    text_chunks[n_chunks].key = (png_charp)"e8v3:base18_hash";
    text_chunks[n_chunks].text = (png_charp)meta->base18_hash;
    text_chunks[n_chunks].text_length = strlen(meta->base18_hash);
    n_chunks++;

    /* integrity: uncompressed (64 hex chars) */
    text_chunks[n_chunks].compression = PNG_TEXT_COMPRESSION_NONE;
    text_chunks[n_chunks].key = (png_charp)"e8v3:sha256";
    text_chunks[n_chunks].text = (png_charp)meta->integrity;
    text_chunks[n_chunks].text_length = strlen(meta->integrity);
    n_chunks++;

    png_set_text(png, info, text_chunks, n_chunks);

    png_write_info(png, info);

    int64_t row_bytes = (int64_t)width * 3;
    for (int y = 0; y < height; y++)
        png_write_row(png, (png_const_bytep)(canvas + y * row_bytes));

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

/* write_jpeg: Write JPEG preview via stb_image_write */
static void write_jpeg(const char *path, const uint8_t *canvas,
                        int width, int height, int quality)
{
    int ok = stbi_write_jpg(path, width, height, 3, canvas, quality);
    if (!ok) {
        fprintf(stderr, "Warning: JPEG write failed for %s\n", path);
    }
}

/* ================================================================
 * Bresenham Line Drawing (for Hamiltonian path overlay)
 * ================================================================ */

static void draw_line(uint8_t *canvas, int W, int H,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            int64_t off = ((int64_t)y0 * W + x0) * 3;
            canvas[off + 0] = r;
            canvas[off + 1] = g;
            canvas[off + 2] = b;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

/* ================================================================
 * Tiered Coloring: map coherence to RGB
 * ================================================================ */

static RGB tier_color(double coherence)
{
    if (coherence > 2.50) return (RGB){255, 255, 255};  /* Tier 3: Transcendental (white) */
    if (coherence > 2.00) return (RGB){255, 215,   0};  /* Tier 2: Resonant (gold) */
    if (coherence > 0.90) return (RGB){  0, 160, 255};  /* Tier 1: Harmonic (blue) */
    return (RGB){ 40,  40,  45};                        /* Tier 0: Noise (dark grey) */
}

/* ================================================================
 * Torsion Coloring: map quality [0,1] to color
 *   quality=1 (in lattice): bright green
 *   quality=0 (far from lattice): dark red
 * ================================================================ */

static RGB torsion_color(double quality)
{
    double torsion = 1.0 - quality;
    int r_val = (int)(180 * torsion + 40);
    int g_val = (int)(180 * (1.0 - torsion) + 40);
    int b_val = 30;
    if (r_val > 255) r_val = 255;
    if (g_val > 255) g_val = 255;
    return (RGB){(uint8_t)r_val, (uint8_t)g_val, (uint8_t)b_val};
}

/* ================================================================
 * Compute per-E8-root quality scores for each sublattice
 * ================================================================ */

static void compute_quality_f4(LatticeView *lv, const F4Lattice *f4, const E8Lattice *e8)
{
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        if (f4->e8_is_f4[ei]) {
            lv->quality[ei] = 1.0;
        } else {
            double proj[4];
            memcpy(proj, e8->roots[ei], sizeof(double) * 4);
            double pn = 0;
            for (int d = 0; d < 4; d++) pn += proj[d] * proj[d];
            pn = sqrt(pn);
            if (pn < 0.01) { lv->quality[ei] = 0.0; continue; }

            int fi = f4->e8_to_f4[ei];
            if (fi < 0) { lv->quality[ei] = 0.0; continue; }

            double dot = 0;
            for (int d = 0; d < 4; d++) dot += proj[d] * f4->roots[fi][d];
            double sim = fabs(dot) / (pn * f4->norms[fi]);
            lv->quality[ei] = sim * sim;
        }
    }
}

static void compute_quality_generic(LatticeView *lv, const int *is_member,
                                     const double roots[][8], int n_sub,
                                     const double *sub_norms,
                                     const E8Lattice *e8, int sub_dim)
{
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        if (is_member[ei]) {
            lv->quality[ei] = 1.0;
        } else {
            double e8n = 0;
            for (int d = 0; d < sub_dim; d++)
                e8n += e8->roots[ei][d] * e8->roots[ei][d];
            e8n = sqrt(e8n);
            if (e8n < 0.01) { lv->quality[ei] = 0.0; continue; }

            double best = 0;
            for (int si = 0; si < n_sub; si++) {
                double dot = 0;
                for (int d = 0; d < sub_dim; d++)
                    dot += e8->roots[ei][d] * roots[si][d];
                double sim = fabs(dot) / (e8n * sub_norms[si]);
                if (sim > best) best = sim;
            }
            lv->quality[ei] = best * best;
        }
    }
}

/* ================================================================
 * SieveIter: Replayable streaming segmented sieve
 * ================================================================ */

#define SIEVE_SEG_SIZE (1 << 18)  /* 256K segment */

typedef struct {
    int64_t *base_primes;
    int64_t  n_base;
    int64_t  limit;
    int64_t  max_primes;
    int64_t  seg_lo;
    int64_t  total_emitted;
    uint8_t *seg_buf;
    int64_t *seg_primes;  /* primes found in current segment */
    int64_t  seg_count;
    int64_t  seg_pos;
} SieveIter;

/* Compute base primes up to sqrt(limit) */
static void sieve_iter_init(SieveIter *si, int64_t max_primes)
{
    si->max_primes = max_primes;
    si->limit = prime_upper_bound(max_primes);

    /* Phase 1: simple sieve up to sqrt(limit) for base primes */
    int64_t sqrt_limit = (int64_t)sqrt((double)si->limit) + 1;
    uint8_t *is_composite_small = (uint8_t *)calloc(sqrt_limit + 1, 1);
    if (!is_composite_small) { fprintf(stderr, "sieve_iter_init: malloc failed\n"); exit(1); }

    int64_t base_cap = (int64_t)(sqrt_limit / (log((double)sqrt_limit) - 1.1)) + 100;
    si->base_primes = (int64_t *)malloc(base_cap * sizeof(int64_t));
    if (!si->base_primes) { fprintf(stderr, "sieve_iter_init: malloc failed\n"); exit(1); }
    si->n_base = 0;

    for (int64_t i = 2; i <= sqrt_limit; i++) {
        if (!is_composite_small[i]) {
            si->base_primes[si->n_base++] = i;
            for (int64_t j = i * i; j <= sqrt_limit; j += i)
                is_composite_small[j] = 1;
        }
    }
    free(is_composite_small);

    /* Allocate working buffers */
    si->seg_buf = (uint8_t *)malloc(SIEVE_SEG_SIZE);
    si->seg_primes = (int64_t *)malloc(SIEVE_SEG_SIZE * sizeof(int64_t));
    if (!si->seg_buf || !si->seg_primes) {
        fprintf(stderr, "sieve_iter_init: malloc failed for seg buffers\n");
        exit(1);
    }

    /* Initialize to start of sieve */
    si->seg_lo = 2;
    si->total_emitted = 0;
    si->seg_count = 0;
    si->seg_pos = 0;
}

/* Yield up to capacity primes into out[]. Returns number actually yielded. */
static int64_t sieve_iter_next_chunk(SieveIter *si, int64_t *out, int64_t capacity)
{
    int64_t filled = 0;

    while (filled < capacity && si->total_emitted < si->max_primes) {
        /* If current segment exhausted, sieve the next one */
        if (si->seg_pos >= si->seg_count) {
            if (si->seg_lo > si->limit) break;

            int64_t hi = si->seg_lo + SIEVE_SEG_SIZE - 1;
            if (hi > si->limit) hi = si->limit;

            memset(si->seg_buf, 0, SIEVE_SEG_SIZE);

            /* Mark composites */
            for (int64_t b = 0; b < si->n_base; b++) {
                int64_t p = si->base_primes[b];
                int64_t start = ((si->seg_lo + p - 1) / p) * p;
                if (start < p * p) start = p * p;
                if (start > hi) continue;
                for (int64_t j = start; j <= hi; j += p)
                    si->seg_buf[j - si->seg_lo] = 1;
            }

            /* Collect primes from segment */
            si->seg_count = 0;
            for (int64_t i = 0; i <= hi - si->seg_lo; i++) {
                if (!si->seg_buf[i]) {
                    si->seg_primes[si->seg_count++] = si->seg_lo + i;
                }
            }
            si->seg_pos = 0;
            si->seg_lo = hi + 1;
        }

        /* Copy from segment buffer to output */
        int64_t avail = si->seg_count - si->seg_pos;
        int64_t want = capacity - filled;
        int64_t remain_total = si->max_primes - si->total_emitted;
        int64_t n = avail;
        if (n > want) n = want;
        if (n > remain_total) n = remain_total;

        memcpy(out + filled, si->seg_primes + si->seg_pos, n * sizeof(int64_t));
        si->seg_pos += n;
        si->total_emitted += n;
        filled += n;
    }

    return filled;
}

/* Reset sieve to beginning, keep base primes */
static void sieve_iter_reset(SieveIter *si)
{
    si->seg_lo = 2;
    si->total_emitted = 0;
    si->seg_count = 0;
    si->seg_pos = 0;
}

/* Free all resources */
static void sieve_iter_free(SieveIter *si)
{
    free(si->base_primes);
    free(si->seg_buf);
    free(si->seg_primes);
    si->base_primes = NULL;
    si->seg_buf = NULL;
    si->seg_primes = NULL;
}

/* ================================================================
 * StreamState: boundary state between chunks
 * ================================================================ */

typedef struct {
    int64_t last_prime;
    int     last_assigns[2];  /* last 2 e8_assignments from prev chunk */
    int64_t global_prime_idx; /* running prime index counter */
    int     has_prev;         /* 0 for first chunk */
} StreamState;

/* ================================================================
 * VertexEntry and LMStats: per (lattice, mode) statistics
 * ================================================================ */

typedef struct {
    int64_t gap_idx;
    float   score;
    int32_t cx, cy;
    int16_t e8_root;  /* E8 root index [0,239] */
} VertexEntry;

typedef struct {
    int32_t min_x, max_x, min_y, max_y;
    int64_t plot_count;
    int     bounds_init;
    VertexEntry *heap;
    int     heap_size, heap_cap;
} LMStats;

/* Min-heap operations on VertexEntry heap (keyed by score) */
static void vertex_heap_push(LMStats *st, VertexEntry entry)
{
    if (st->heap_size < st->heap_cap) {
        st->heap[st->heap_size] = entry;
        st->heap_size++;
        /* Bubble up */
        int k = st->heap_size - 1;
        while (k > 0) {
            int parent = (k - 1) / 2;
            if (st->heap[parent].score > st->heap[k].score) {
                VertexEntry tmp = st->heap[parent];
                st->heap[parent] = st->heap[k];
                st->heap[k] = tmp;
                k = parent;
            } else break;
        }
    } else if (entry.score > st->heap[0].score) {
        st->heap[0] = entry;
        /* Sift down */
        int k = 0;
        for (;;) {
            int left = 2*k+1, right = 2*k+2, smallest = k;
            if (left < st->heap_size && st->heap[left].score < st->heap[smallest].score)
                smallest = left;
            if (right < st->heap_size && st->heap[right].score < st->heap[smallest].score)
                smallest = right;
            if (smallest == k) break;
            VertexEntry tmp = st->heap[k];
            st->heap[k] = st->heap[smallest];
            st->heap[smallest] = tmp;
            k = smallest;
        }
    }
}

/* ================================================================
 * process_chunk: compute Ulam coords, gaps, E8 assignments,
 *                triplet coherence for one chunk of primes
 *
 * Inputs:
 *   primes[0..n-1] — the chunk's primes
 *   n              — number of primes in this chunk
 *   ss             — stream state from previous chunk
 *   e8             — E8 lattice
 *
 * Outputs (pre-allocated by caller):
 *   cx[0..n-1], cy[0..n-1]   — Ulam coordinates
 *   assigns[0..n_gaps-1]     — E8 root assignments per gap
 *   coherence[0..n_gaps-1]   — triplet coherence per gap
 *   norm_gaps[0..n_gaps-1]   — normalized gaps
 *
 * n_gaps = n - 1  if !has_prev (first chunk)
 * n_gaps = n      if has_prev  (gap[0] = primes[0] - last_prime)
 *
 * Returns the number of gaps computed.
 * ================================================================ */

static int64_t process_chunk(const int64_t *primes, int64_t n,
                              const StreamState *ss,
                              const E8Lattice *e8,
                              int32_t *cx, int32_t *cy,
                              int *assigns, float *coherence,
                              double *norm_gaps)
{
    /* --- Ulam coordinates (fully parallel) --- */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        ulam_coord(primes[i], &cx[i], &cy[i]);
    }

    /* --- Gaps and E8 assignments --- */
    int64_t n_gaps;

    if (ss->has_prev) {
        /* Gap 0: boundary gap from previous chunk's last prime */
        n_gaps = n;
        {
            double gap = (double)(primes[0] - ss->last_prime);
            double log_p = log((double)ss->last_prime);
            if (log_p < 1.0) log_p = 1.0;
            norm_gaps[0] = gap / log_p;
            assigns[0] = e8_assign_root(e8, norm_gaps[0]);
        }
        /* Interior gaps: parallel */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n; i++) {
            double gap = (double)(primes[i] - primes[i - 1]);
            double log_p = log((double)primes[i - 1]);
            if (log_p < 1.0) log_p = 1.0;
            norm_gaps[i] = gap / log_p;
            assigns[i] = e8_assign_root(e8, norm_gaps[i]);
        }
    } else {
        /* First chunk: no previous prime */
        n_gaps = n - 1;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n_gaps; i++) {
            double gap = (double)(primes[i + 1] - primes[i]);
            double log_p = log((double)primes[i]);
            if (log_p < 1.0) log_p = 1.0;
            norm_gaps[i] = gap / log_p;
            assigns[i] = e8_assign_root(e8, norm_gaps[i]);
        }
    }

    /* --- Triplet coherence --- */
    /* Initialize all to 0 */
    memset(coherence, 0, n_gaps * sizeof(float));

    if (ss->has_prev) {
        /* Boundary triplet at gap 0: uses last_assigns[0], last_assigns[1], assigns[0] */
        /* But we need 3 consecutive assignments. last_assigns[1] is the second-to-last
         * assignment from previous chunk, last_assigns[0] is the last assignment.
         * For gap index 0 in this chunk:
         *   triplet = (last_assigns[0], assigns[0], assigns[1 if exists])
         * Actually, coherence[i] uses triplet (assigns[i-1], assigns[i], assigns[i+1]).
         * For i=0 with has_prev: assigns[-1] = last_assigns[0]
         * For i=1 with has_prev: assigns[0], assigns[1], assigns[2] — all interior
         */

        /* Gap 0: triplet uses last_assigns[0], assigns[0], assigns[1] */
        if (n_gaps >= 2) {
            int r0 = ss->last_assigns[0];
            int r1 = assigns[0];
            int r2 = assigns[1];
            double sum[E8_DIM];
            for (int d = 0; d < E8_DIM; d++)
                sum[d] = e8->roots[r0][d] + e8->roots[r1][d] + e8->roots[r2][d];
            double sum_sq = 0;
            for (int d = 0; d < E8_DIM; d++) sum_sq += sum[d] * sum[d];
            coherence[0] = (float)(sum_sq / 6.0);
        }

        /* Interior: parallel */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n_gaps - 1; i++) {
            int r0 = assigns[i - 1];
            int r1 = assigns[i];
            int r2 = assigns[i + 1];
            double sum[E8_DIM];
            for (int d = 0; d < E8_DIM; d++)
                sum[d] = e8->roots[r0][d] + e8->roots[r1][d] + e8->roots[r2][d];
            double sum_sq = 0;
            for (int d = 0; d < E8_DIM; d++) sum_sq += sum[d] * sum[d];
            coherence[i] = (float)(sum_sq / 6.0);
        }
    } else {
        /* First chunk: coherence[0] stays 0 (no triplet for first gap) */
        /* Interior: parallel */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < n_gaps - 1; i++) {
            int r0 = assigns[i - 1];
            int r1 = assigns[i];
            int r2 = assigns[i + 1];
            double sum[E8_DIM];
            for (int d = 0; d < E8_DIM; d++)
                sum[d] = e8->roots[r0][d] + e8->roots[r1][d] + e8->roots[r2][d];
            double sum_sq = 0;
            for (int d = 0; d < E8_DIM; d++) sum_sq += sum[d] * sum[d];
            coherence[i] = (float)(sum_sq / 6.0);
        }
    }

    return n_gaps;
}

/* ================================================================
 * GlobalStats: coherence statistics accumulated in pass1
 * ================================================================ */

typedef struct {
    double  coh_sum, coh_max;
    int64_t tier_counts[4];
    int64_t total_triplets;
    char    base18_hash_hex[65];
} GlobalStats;

/* ================================================================
 * pass1_stats: Stream all chunks, accumulate per-(lattice,mode)
 *              coordinate bounds and top-K vertex heaps.
 *              Also accumulate global coherence stats.
 * ================================================================ */

static void pass1_stats(SieveIter *si,
                         const Config *cfg,
                         const LatticeView views[LAT_COUNT],
                         const E8Lattice *e8,
                         LMStats lm_stats[LAT_COUNT][MODE_COUNT],
                         StreamState *final_ss,
                         GlobalStats *gstats)
{
    printf("Pass 1: Streaming statistics...\n");
    tic();

    /* Allocate chunk buffers */
    int64_t chunk_cap = CHUNK_SIZE + 1; /* +1 for safety */
    int64_t *primes   = (int64_t *)malloc(chunk_cap * sizeof(int64_t));
    int     *assigns  = (int *)malloc(chunk_cap * sizeof(int));
    float   *coherence = (float *)malloc(chunk_cap * sizeof(float));
    int32_t *cx       = (int32_t *)malloc(chunk_cap * sizeof(int32_t));
    int32_t *cy       = (int32_t *)malloc(chunk_cap * sizeof(int32_t));
    double  *norm_gaps = (double *)malloc(chunk_cap * sizeof(double));
    if (!primes || !assigns || !coherence || !cx || !cy || !norm_gaps) {
        fprintf(stderr, "pass1_stats: malloc failed for chunk buffers\n");
        exit(1);
    }

    /* Initialize LMStats */
    for (int l = 0; l < LAT_COUNT; l++) {
        for (int m = 0; m < MODE_COUNT; m++) {
            LMStats *st = &lm_stats[l][m];
            st->min_x = st->max_x = st->min_y = st->max_y = 0;
            st->plot_count = 0;
            st->bounds_init = 0;

            /* Vertex heap */
            int hcap = cfg->n_vertices * 2;
            if (hcap < 1) hcap = 1;
            st->heap_cap = hcap;
            st->heap_size = 0;
            st->heap = (VertexEntry *)malloc(hcap * sizeof(VertexEntry));
            if (!st->heap) { fprintf(stderr, "pass1_stats: malloc failed for heap\n"); exit(1); }
        }
    }

    /* Global coherence accumulators */
    double coh_sum = 0, coh_max = 0;
    int64_t tier_counts[4] = {0, 0, 0, 0};
    int64_t total_triplets = 0;

    /* Base-18 streaming SHA-256 */
    SHA256_CTX base18_ctx;
    sha256_init(&base18_ctx);

    /* Stream state */
    StreamState ss = {0};
    ss.has_prev = 0;
    ss.global_prime_idx = 0;
    ss.last_prime = 0;
    ss.last_assigns[0] = 0;
    ss.last_assigns[1] = 0;

    int chunk_num = 0;

    for (;;) {
        int64_t n = sieve_iter_next_chunk(si, primes, CHUNK_SIZE);
        if (n <= 0) break;

        int64_t n_gaps = process_chunk(primes, n, &ss, e8, cx, cy, assigns, coherence, norm_gaps);

        /* --- Accumulate coherence stats + base-18 streaming hash --- */
        for (int64_t i = 0; i < n_gaps; i++) {
            double c = (double)coherence[i];
            coh_sum += c;
            if (c > coh_max) coh_max = c;
            if (c > 2.50) tier_counts[3]++;
            else if (c > 2.00) tier_counts[2]++;
            else if (c > 0.90) tier_counts[1]++;
            else tier_counts[0]++;
            total_triplets++;

            /* Base-18 streaming hash: one char per gap */
            int8_t b18 = g_root_to_base18[assigns[i]];
            uint8_t ch = (b18 >= 0) ? (uint8_t)ALPHABET_18[(int)b18] : (uint8_t)'.';
            sha256_update(&base18_ctx, &ch, 1);
        }

        /* --- For each active (lattice, mode): update bounds and vertex heaps --- */
        for (int l = 0; l < LAT_COUNT; l++) {
            if (cfg->lattice >= 0 && cfg->lattice != l) continue;
            const LatticeView *lv = &views[l];

            for (int m = 0; m < MODE_COUNT; m++) {
                if (cfg->mode >= 0 && cfg->mode != m) continue;
                LMStats *st = &lm_stats[l][m];

                /* Iterate over gaps in this chunk.
                 * Gap g maps to:
                 *   - If has_prev: prime at index g in this chunk's primes,
                 *     global prime index = ss.global_prime_idx + g
                 *   - If !has_prev (first chunk): prime at index g+1 in primes,
                 *     global prime index = g + 1
                 */
                for (int64_t g = 0; g < n_gaps; g++) {
                    int64_t pi_local;  /* index into this chunk's primes/cx/cy */
                    int64_t pi_global; /* global prime index */

                    if (ss.has_prev) {
                        pi_local = g;
                        pi_global = ss.global_prime_idx + g;
                    } else {
                        pi_local = g + 1;
                        pi_global = g + 1;
                    }

                    int e8_idx = assigns[g];

                    /* Lattice membership filter */
                    int in_lattice = lv->e8_is_member[e8_idx];
                    int plot = (lv->n_roots == 240) ? 1 : in_lattice;

                    /* Strobe filter */
                    if (m == MODE_STROBE) {
                        int phase = (int)(pi_global % MONSTER_DIM);
                        int dist = abs(phase - cfg->strobe_phase);
                        if (dist > MONSTER_DIM / 2) dist = MONSTER_DIM - dist;
                        if (dist > cfg->strobe_epsilon) plot = 0;
                    }

                    if (!plot) continue;

                    int32_t x = cx[pi_local];
                    int32_t y = cy[pi_local];

                    /* Update bounds */
                    if (!st->bounds_init) {
                        st->min_x = st->max_x = x;
                        st->min_y = st->max_y = y;
                        st->bounds_init = 1;
                    } else {
                        if (x < st->min_x) st->min_x = x;
                        if (x > st->max_x) st->max_x = x;
                        if (y < st->min_y) st->min_y = y;
                        if (y > st->max_y) st->max_y = y;
                    }
                    st->plot_count++;

                    /* Push to vertex heap if coherence > 0 */
                    double score = (double)coherence[g];
                    if (score > 0.0 && (cfg->draw_path || cfg->n_vertices > 0)) {
                        VertexEntry ve;
                        ve.gap_idx = (ss.has_prev)
                            ? ss.global_prime_idx + g
                            : g + 1;
                        ve.score = (float)score;
                        ve.cx = x;
                        ve.cy = y;
                        ve.e8_root = (int16_t)e8_idx;
                        vertex_heap_push(st, ve);
                    }
                }
            }
        }

        /* Update stream state for next chunk */
        ss.last_prime = primes[n - 1];
        if (n_gaps >= 2) {
            ss.last_assigns[1] = assigns[n_gaps - 2];
            ss.last_assigns[0] = assigns[n_gaps - 1];
        } else if (n_gaps == 1) {
            ss.last_assigns[1] = ss.last_assigns[0];
            ss.last_assigns[0] = assigns[0];
        }
        if (!ss.has_prev) {
            ss.global_prime_idx = n;
        } else {
            ss.global_prime_idx += n;
        }
        ss.has_prev = 1;

        chunk_num++;
        if (chunk_num % 4 == 0) {
            char b1[32];
            printf("  Chunk %d: %s primes streamed (%.1fs)\n",
                   chunk_num, fmt_comma(ss.global_prime_idx, b1, sizeof(b1)), toc());
        }
    }

    /* Finalize base-18 hash */
    {
        uint8_t b18_hash[32];
        sha256_final(&base18_ctx, b18_hash);
        sha256_hex(b18_hash, gstats->base18_hash_hex);
    }

    /* Populate gstats */
    gstats->coh_sum = coh_sum;
    gstats->coh_max = coh_max;
    for (int i = 0; i < 4; i++) gstats->tier_counts[i] = tier_counts[i];
    gstats->total_triplets = total_triplets;

    /* Print coherence stats */
    {
        char b1[32];
        printf("\n  Global coherence statistics (%s triplets):\n",
               fmt_comma(total_triplets, b1, sizeof(b1)));
        if (total_triplets > 0) {
            printf("    Mean coherence: %.4f (max: %.4f)\n",
                   coh_sum / total_triplets, coh_max);
            printf("    Tier 0 (Noise):           %s (%.1f%%)\n",
                   fmt_comma(tier_counts[0], b1, sizeof(b1)),
                   100.0 * tier_counts[0] / total_triplets);
            printf("    Tier 1 (Harmonic):        %s (%.1f%%)\n",
                   fmt_comma(tier_counts[1], b1, sizeof(b1)),
                   100.0 * tier_counts[1] / total_triplets);
            printf("    Tier 2 (Resonant):        %s (%.1f%%)\n",
                   fmt_comma(tier_counts[2], b1, sizeof(b1)),
                   100.0 * tier_counts[2] / total_triplets);
            printf("    Tier 3 (Transcendental):  %s (%.1f%%)\n",
                   fmt_comma(tier_counts[3], b1, sizeof(b1)),
                   100.0 * tier_counts[3] / total_triplets);
        }
        printf("    Base-18 stream hash: %s\n", gstats->base18_hash_hex);
    }

    printf("  Pass 1 complete in %.2fs\n\n", toc());

    /* Save final stream state */
    *final_ss = ss;

    /* Free chunk buffers */
    free(primes);
    free(assigns);
    free(coherence);
    free(cx);
    free(cy);
    free(norm_gaps);
}

/* ================================================================
 * pass2_render: Reset sieve, stream again, render directly to
 *               pre-allocated canvases using bounds from Pass 1.
 * ================================================================ */

static void pass2_render(SieveIter *si,
                          const Config *cfg,
                          const LatticeView views[LAT_COUNT],
                          const E8Lattice *e8,
                          LMStats lm_stats[LAT_COUNT][MODE_COUNT],
                          const GlobalStats *gstats)
{
    printf("Pass 2: Streaming render...\n");
    tic();

    int canvas_px = cfg->dpi * cfg->fig_inches;

    /* Compute scale/offset from Pass 1 bounds for each active (lattice, mode) */
    double lm_scale[LAT_COUNT][MODE_COUNT];
    double lm_off_x[LAT_COUNT][MODE_COUNT];
    double lm_off_y[LAT_COUNT][MODE_COUNT];

    /* Allocate canvases */
    uint8_t *canvases[LAT_COUNT][MODE_COUNT];
    memset(canvases, 0, sizeof(canvases));

    int64_t canvas_bytes = (int64_t)canvas_px * canvas_px * 3;

    for (int l = 0; l < LAT_COUNT; l++) {
        if (cfg->lattice >= 0 && cfg->lattice != l) continue;
        for (int m = 0; m < MODE_COUNT; m++) {
            if (cfg->mode >= 0 && cfg->mode != m) continue;
            LMStats *st = &lm_stats[l][m];

            if (st->plot_count == 0) continue;

            double range_x = (double)(st->max_x - st->min_x);
            double range_y = (double)(st->max_y - st->min_y);
            double range = fmax(range_x, range_y);
            if (range < 1.0) range = 1.0;
            double margin_frac = 0.02;
            double scale = (canvas_px * (1.0 - 2.0 * margin_frac)) / range;
            double offset_x = canvas_px * margin_frac - st->min_x * scale
                              + (range - range_x) * scale * 0.5;
            double offset_y = canvas_px * margin_frac - st->min_y * scale
                              + (range - range_y) * scale * 0.5;

            lm_scale[l][m] = scale;
            lm_off_x[l][m] = offset_x;
            lm_off_y[l][m] = offset_y;

            canvases[l][m] = (uint8_t *)calloc(canvas_bytes, 1);
            if (!canvases[l][m]) {
                fprintf(stderr, "pass2_render: failed to allocate canvas for %s/%s (%.1f MB)\n",
                        views[l].name, mode_names[m], canvas_bytes / 1e6);
                continue;
            }

            {
                char b1[32];
                printf("  %s / %s: %s plotted primes, canvas %d x %d\n",
                       views[l].name, mode_names[m],
                       fmt_comma(st->plot_count, b1, sizeof(b1)),
                       canvas_px, canvas_px);
            }
        }
    }

    /* Allocate chunk buffers */
    int64_t chunk_cap = CHUNK_SIZE + 1;
    int64_t *primes   = (int64_t *)malloc(chunk_cap * sizeof(int64_t));
    int     *assigns  = (int *)malloc(chunk_cap * sizeof(int));
    float   *coherence = (float *)malloc(chunk_cap * sizeof(float));
    int32_t *cx       = (int32_t *)malloc(chunk_cap * sizeof(int32_t));
    int32_t *cy       = (int32_t *)malloc(chunk_cap * sizeof(int32_t));
    double  *norm_gaps = (double *)malloc(chunk_cap * sizeof(double));
    if (!primes || !assigns || !coherence || !cx || !cy || !norm_gaps) {
        fprintf(stderr, "pass2_render: malloc failed for chunk buffers\n");
        exit(1);
    }

    /* Stream state */
    StreamState ss = {0};
    ss.has_prev = 0;
    ss.global_prime_idx = 0;
    ss.last_prime = 0;
    ss.last_assigns[0] = 0;
    ss.last_assigns[1] = 0;

    int chunk_num = 0;

    for (;;) {
        int64_t n = sieve_iter_next_chunk(si, primes, CHUNK_SIZE);
        if (n <= 0) break;

        int64_t n_gaps = process_chunk(primes, n, &ss, e8, cx, cy, assigns, coherence, norm_gaps);

        /* --- For each active canvas: parallel loop over gaps, compute pixel, write color --- */
        for (int l = 0; l < LAT_COUNT; l++) {
            if (cfg->lattice >= 0 && cfg->lattice != l) continue;
            const LatticeView *lv = &views[l];

            for (int m = 0; m < MODE_COUNT; m++) {
                if (cfg->mode >= 0 && cfg->mode != m) continue;
                if (!canvases[l][m]) continue;

                double scale = lm_scale[l][m];
                double offset_x = lm_off_x[l][m];
                double offset_y = lm_off_y[l][m];
                uint8_t *canvas = canvases[l][m];

                #pragma omp parallel for schedule(static)
                for (int64_t g = 0; g < n_gaps; g++) {
                    int64_t pi_local;
                    int64_t pi_global;

                    if (ss.has_prev) {
                        pi_local = g;
                        pi_global = ss.global_prime_idx + g;
                    } else {
                        pi_local = g + 1;
                        pi_global = g + 1;
                    }

                    int e8_idx = assigns[g];

                    /* Lattice membership filter */
                    int in_lattice = lv->e8_is_member[e8_idx];
                    int plot = (lv->n_roots == 240) ? 1 : in_lattice;

                    /* Strobe filter */
                    if (m == MODE_STROBE) {
                        int phase = (int)(pi_global % MONSTER_DIM);
                        int dist = abs(phase - cfg->strobe_phase);
                        if (dist > MONSTER_DIM / 2) dist = MONSTER_DIM - dist;
                        if (dist > cfg->strobe_epsilon) plot = 0;
                    }

                    if (!plot) continue;

                    int px = (int)(cx[pi_local] * scale + offset_x);
                    int py = (int)(cy[pi_local] * scale + offset_y);
                    py = canvas_px - 1 - py;
                    if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;

                    int sub_idx = lv->e8_to_idx[e8_idx];

                    RGB c;
                    switch (m) {
                        case MODE_JORDAN:
                            if (sub_idx >= 0) {
                                c = jordan_to_color_range(lv->jordan_traces[sub_idx],
                                                          lv->jmin, lv->jmax);
                            } else {
                                c = (RGB){30, 30, 30};
                            }
                            break;

                        case MODE_TIERED:
                            c = tier_color((double)coherence[g]);
                            break;

                        case MODE_STROBE:
                            if (sub_idx >= 0) {
                                c = jordan_to_color_range(lv->jordan_traces[sub_idx],
                                                          lv->jmin, lv->jmax);
                            } else {
                                c = (RGB){100, 100, 100};
                            }
                            break;

                        case MODE_TORSION:
                            c = torsion_color(lv->quality[e8_idx]);
                            break;

                        default:
                            c = (RGB){128, 128, 128};
                    }

                    int64_t off = ((int64_t)py * canvas_px + px) * 3;
                    canvas[off + 0] = c.r;
                    canvas[off + 1] = c.g;
                    canvas[off + 2] = c.b;
                }
            }
        }

        /* Update stream state */
        ss.last_prime = primes[n - 1];
        if (n_gaps >= 2) {
            ss.last_assigns[1] = assigns[n_gaps - 2];
            ss.last_assigns[0] = assigns[n_gaps - 1];
        } else if (n_gaps == 1) {
            ss.last_assigns[1] = ss.last_assigns[0];
            ss.last_assigns[0] = assigns[0];
        }
        if (!ss.has_prev) {
            ss.global_prime_idx = n;
        } else {
            ss.global_prime_idx += n;
        }
        ss.has_prev = 1;

        chunk_num++;
        if (chunk_num % 4 == 0) {
            char b1[32];
            printf("  Chunk %d rendered (%s primes, %.1fs)\n",
                   chunk_num, fmt_comma(ss.global_prime_idx, b1, sizeof(b1)), toc());
        }
    }

    /* Free chunk buffers */
    free(primes);
    free(assigns);
    free(coherence);
    free(cx);
    free(cy);
    free(norm_gaps);

    printf("  Pixel rendering complete in %.2fs\n", toc());

    /* --- Overlay vertices + paths on each canvas, write PNGs --- */
    printf("  Writing PNGs...\n");

    /* Build suffix string */
    char suffix[64];
    if (cfg->max_primes >= 1000000000LL) {
        int64_t billions = cfg->max_primes / 1000000000LL;
        snprintf(suffix, sizeof(suffix), "%ldB", (long)billions);
    } else if (cfg->max_primes >= 1000000LL) {
        snprintf(suffix, sizeof(suffix), "%ldM", (long)(cfg->max_primes / 1000000LL));
    } else {
        snprintf(suffix, sizeof(suffix), "%ldk", (long)(cfg->max_primes / 1000));
    }

    int n_renders = 0;

    for (int l = 0; l < LAT_COUNT; l++) {
        if (cfg->lattice >= 0 && cfg->lattice != l) continue;
        const LatticeView *lv = &views[l];

        for (int m = 0; m < MODE_COUNT; m++) {
            if (cfg->mode >= 0 && cfg->mode != m) continue;
            if (!canvases[l][m]) continue;

            LMStats *st = &lm_stats[l][m];
            uint8_t *canvas = canvases[l][m];
            double scale = lm_scale[l][m];
            double offset_x = lm_off_x[l][m];
            double offset_y = lm_off_y[l][m];

            /* --- Vertex overlay --- */
            if (cfg->draw_path || cfg->n_vertices > 0) {
                /* Sort vertices by gap_idx for path drawing */
                for (int i = 0; i < st->heap_size - 1; i++) {
                    for (int j = i + 1; j < st->heap_size; j++) {
                        if (st->heap[j].gap_idx < st->heap[i].gap_idx) {
                            VertexEntry tmp = st->heap[i];
                            st->heap[i] = st->heap[j];
                            st->heap[j] = tmp;
                        }
                    }
                }

                int actual_vertices = st->heap_size < cfg->n_vertices
                                      ? st->heap_size : cfg->n_vertices;

                /* Draw Hamiltonian path (thin cyan lines between consecutive vertices) */
                if (cfg->draw_path && actual_vertices > 1) {
                    for (int v = 0; v < actual_vertices - 1; v++) {
                        int x0 = (int)(st->heap[v].cx * scale + offset_x);
                        int y0 = canvas_px - 1 - (int)(st->heap[v].cy * scale + offset_y);
                        int x1 = (int)(st->heap[v + 1].cx * scale + offset_x);
                        int y1 = canvas_px - 1 - (int)(st->heap[v + 1].cy * scale + offset_y);
                        draw_line(canvas, canvas_px, canvas_px, x0, y0, x1, y1, 0, 255, 200);
                    }
                }

                /* Overlay vertices as white dots */
                int radius = cfg->dpi >= 600 ? 3 : 2;
                for (int v = 0; v < actual_vertices; v++) {
                    int vcx = (int)(st->heap[v].cx * scale + offset_x);
                    int vcy = canvas_px - 1 - (int)(st->heap[v].cy * scale + offset_y);

                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            if (dx*dx + dy*dy > radius*radius) continue;
                            int px = vcx + dx;
                            int py = vcy + dy;
                            if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;
                            int64_t off = ((int64_t)py * canvas_px + px) * 3;
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

                printf("    %s / %s: %d crystalline vertices",
                       lv->name, mode_names[m], actual_vertices);
                if (cfg->draw_path) printf(" + path");
                printf("\n");
            }

            /* --- Build metadata bundle --- */
            MetadataBundle meta;
            metadata_init(&meta);

            build_params_string(&meta, cfg->max_primes, lv->name, mode_names[m],
                                cfg->dpi, cfg->fig_inches, cfg->n_vertices,
                                cfg->draw_path, cfg->strobe_epsilon, cfg->strobe_phase);

            {
                double coh_mean = gstats->total_triplets > 0
                    ? gstats->coh_sum / gstats->total_triplets : 0.0;
                build_stats_string(&meta, coh_mean, gstats->coh_max,
                                    gstats->tier_counts[0], gstats->tier_counts[1],
                                    gstats->tier_counts[2], gstats->tier_counts[3],
                                    gstats->total_triplets, st->plot_count,
                                    st->min_x, st->max_x, st->min_y, st->max_y);
            }

            /* Build vertices string + vertex path decode */
            {
                int actual_v = st->heap_size < cfg->n_vertices
                               ? st->heap_size : cfg->n_vertices;

                /* Convert VertexEntry heap to MetaVertexEntry array */
                MetaVertexEntry *mverts = (MetaVertexEntry *)malloc(
                    actual_v * sizeof(MetaVertexEntry));
                if (mverts) {
                    for (int v = 0; v < actual_v; v++) {
                        mverts[v].gap_idx = st->heap[v].gap_idx;
                        mverts[v].score   = st->heap[v].score;
                        mverts[v].cx      = st->heap[v].cx;
                        mverts[v].cy      = st->heap[v].cy;
                        mverts[v].e8_root = st->heap[v].e8_root;
                    }

                    build_vertices_string(&meta, mverts, actual_v);

                    if (actual_v > 1) {
                        int decode_len = 0;
                        decode_vertex_path(mverts, actual_v, e8->roots,
                                            meta.vertex_decode, &decode_len);
                    }

                    free(mverts);
                }
            }

            /* Copy base18 hash from gstats */
            memcpy(meta.base18_hash, gstats->base18_hash_hex, 65);

            /* Compute integrity hash over all fields */
            compute_integrity_hash(&meta);

            /* --- Write PNG with metadata --- */
            char png_path[1024];
            snprintf(png_path, sizeof(png_path), "%s/viz3_%s_%s_%s.png",
                     cfg->output_dir, lv->name, mode_names[m], suffix);

            write_png_meta(png_path, canvas, canvas_px, canvas_px, &meta);

            FILE *check = fopen(png_path, "rb");
            long png_size = 0;
            if (check) { fseek(check, 0, SEEK_END); png_size = ftell(check); fclose(check); }
            printf("    -> %s (%.1f MB)\n", png_path, png_size / 1e6);

            /* --- Write JPEG preview --- */
            char jpg_path[1024];
            snprintf(jpg_path, sizeof(jpg_path), "%s/viz3_%s_%s_%s_preview.jpg",
                     cfg->output_dir, lv->name, mode_names[m], suffix);

            write_jpeg(jpg_path, canvas, canvas_px, canvas_px, 85);

            FILE *jcheck = fopen(jpg_path, "rb");
            long jpg_size = 0;
            if (jcheck) { fseek(jcheck, 0, SEEK_END); jpg_size = ftell(jcheck); fclose(jcheck); }
            printf("    -> %s (%.1f MB)\n", jpg_path, jpg_size / 1e6);

            printf("    Metadata: integrity=%s\n", meta.integrity);

            metadata_free(&meta);

            n_renders++;

            /* Free canvas */
            free(canvases[l][m]);
            canvases[l][m] = NULL;
        }
    }

    /* Free remaining vertex heaps */
    for (int l = 0; l < LAT_COUNT; l++) {
        for (int m = 0; m < MODE_COUNT; m++) {
            if (lm_stats[l][m].heap) {
                free(lm_stats[l][m].heap);
                lm_stats[l][m].heap = NULL;
            }
            if (canvases[l][m]) {
                free(canvases[l][m]);
                canvases[l][m] = NULL;
            }
        }
    }

    printf("  Pass 2 complete: %d PNG(s) rendered in %.2fs\n\n", n_renders, toc());
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);
    int canvas_px = cfg.dpi * cfg.fig_inches;
    char b1[32], b2[32];

    printf("================================================================\n");
    printf("  Streaming Multi-Lattice Prime Gap Visualizer v3.0\n");
    printf("================================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, b1, sizeof(b1)));
    printf("Chunk size : %s primes per chunk\n", fmt_comma((int64_t)CHUNK_SIZE, b2, sizeof(b2)));
    printf("Canvas     : %d x %d px (%d DPI x %d in)\n", canvas_px, canvas_px, cfg.dpi, cfg.fig_inches);
    printf("Vertices   : %d%s\n", cfg.n_vertices, cfg.draw_path ? " + path" : "");
    printf("Mode       : %s\n", cfg.mode < 0 ? "ALL" : mode_names[cfg.mode]);
    printf("Lattice    : %s\n", cfg.lattice < 0 ? "ALL" : lattice_names[cfg.lattice]);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("================================================================\n\n");

    /* ---- Step 1: Initialize sieve iterator ---- */
    printf("Step 1: Initializing sieve iterator...\n");
    tic();
    SieveIter si;
    sieve_iter_init(&si, cfg.max_primes);
    printf("  %s base primes computed (sieve limit: %s) in %.2fs\n",
           fmt_comma(si.n_base, b1, sizeof(b1)),
           fmt_comma(si.limit, b2, sizeof(b2)), toc());

    /* ---- Step 2: Initialize all lattices ---- */
    printf("\nStep 2: Initializing lattices...\n");
    tic();

    E8Lattice e8;  e8_init(&e8);
    E7Lattice e7;  e7_init(&e7, &e8);
    E6Lattice e6;  e6_init(&e6, &e8);
    F4Lattice f4;  f4_init(&f4, &e8);
    G2Lattice g2;  g2_init(&g2, &e8);
    S16Lattice s16; s16_init(&s16, &e8);
    D8Lattice d8;  d8_init(&d8, &e8);
    (void)d8;  /* D8 initialized but not used in views */

    /* Build generic views */
    LatticeView views[LAT_COUNT];

    /* E8 -- all roots are members */
    {
        static int e8_all_member[E8_NUM_ROOTS];
        static int e8_identity[E8_NUM_ROOTS];
        static double e8_jt[E8_NUM_ROOTS];
        static double e8_norms[E8_NUM_ROOTS];
        for (int i = 0; i < E8_NUM_ROOTS; i++) {
            e8_all_member[i] = 1;
            e8_identity[i] = i;
            double t = 0, n2 = 0;
            for (int d = 0; d < E8_DIM; d++) {
                t  += e8.roots[i][d];
                n2 += e8.roots[i][d] * e8.roots[i][d];
            }
            e8_jt[i] = t;
            e8_norms[i] = sqrt(n2);
        }
        views[LAT_E8] = (LatticeView){
            .name = "E8", .n_roots = 240,
            .e8_is_member = e8_all_member, .e8_to_idx = e8_identity,
            .jordan_traces = e8_jt, .norms = e8_norms
        };
        lattice_view_finalize(&views[LAT_E8], &e8);
        for (int i = 0; i < E8_NUM_ROOTS; i++) views[LAT_E8].quality[i] = 1.0;
    }

    /* E7 */
    views[LAT_E7] = (LatticeView){
        .name = "E7", .n_roots = E7_NUM_ROOTS,
        .e8_is_member = e7.e8_is_e7, .e8_to_idx = e7.e8_to_e7,
        .jordan_traces = e7.jordan_traces, .norms = e7.norms
    };
    lattice_view_finalize(&views[LAT_E7], &e8);
    compute_quality_generic(&views[LAT_E7], e7.e8_is_e7,
                            (const double (*)[8])e7.roots, E7_NUM_ROOTS, e7.norms, &e8, E8_DIM);

    /* E6 */
    views[LAT_E6] = (LatticeView){
        .name = "E6", .n_roots = E6_NUM_ROOTS,
        .e8_is_member = e6.e8_is_e6, .e8_to_idx = e6.e8_to_e6,
        .jordan_traces = e6.jordan_traces, .norms = e6.norms
    };
    lattice_view_finalize(&views[LAT_E6], &e8);
    compute_quality_generic(&views[LAT_E6], e6.e8_is_e6,
                            (const double (*)[8])e6.roots, E6_NUM_ROOTS, e6.norms, &e8, E8_DIM);

    /* F4 -- uses 4D projection, needs special quality computation */
    {
        views[LAT_F4] = (LatticeView){
            .name = "F4", .n_roots = F4_NUM_ROOTS,
            .e8_is_member = f4.e8_is_f4, .e8_to_idx = f4.e8_to_f4,
            .jordan_traces = f4.jordan_traces, .norms = f4.norms
        };
        lattice_view_finalize(&views[LAT_F4], &e8);
        compute_quality_f4(&views[LAT_F4], &f4, &e8);
    }

    /* G2 -- lives in 2D */
    views[LAT_G2] = (LatticeView){
        .name = "G2", .n_roots = G2_NUM_ROOTS,
        .e8_is_member = g2.e8_is_g2, .e8_to_idx = g2.e8_to_g2,
        .jordan_traces = g2.jordan_traces, .norms = g2.norms
    };
    lattice_view_finalize(&views[LAT_G2], &e8);
    /* G2 quality: cosine similarity in first 2 coords */
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        if (g2.e8_is_g2[ei]) {
            views[LAT_G2].quality[ei] = 1.0;
        } else {
            double px = e8.roots[ei][0], py = e8.roots[ei][1];
            double pn = sqrt(px*px + py*py);
            if (pn < 0.01) { views[LAT_G2].quality[ei] = 0.0; continue; }
            int gi = g2.e8_to_g2[ei];
            if (gi < 0) { views[LAT_G2].quality[ei] = 0.0; continue; }
            double dot = px * g2.roots[gi][0] + py * g2.roots[gi][1];
            double sim = fabs(dot) / (pn * g2.norms[gi]);
            views[LAT_G2].quality[ei] = sim * sim;
        }
    }

    /* SO(16) / S16 */
    views[LAT_SO16] = (LatticeView){
        .name = "SO16", .n_roots = S16_NUM_ROOTS,
        .e8_is_member = s16.e8_is_s16, .e8_to_idx = s16.e8_to_s16,
        .jordan_traces = s16.jordan_traces, .norms = s16.norms
    };
    lattice_view_finalize(&views[LAT_SO16], &e8);
    compute_quality_generic(&views[LAT_SO16], s16.e8_is_s16,
                            (const double (*)[8])s16.roots, S16_NUM_ROOTS, s16.norms, &e8, E8_DIM);

    /* Print lattice membership counts */
    for (int l = 0; l < LAT_COUNT; l++) {
        int cnt = 0;
        for (int i = 0; i < E8_NUM_ROOTS; i++) if (views[l].e8_is_member[i]) cnt++;
        printf("  %4s: %3d/%d E8 roots, Jordan trace [%.2f, %.2f]\n",
               views[l].name, cnt, E8_NUM_ROOTS, views[l].jmin, views[l].jmax);
    }
    printf("  Lattices initialized in %.2fs\n", toc());

    /* ---- mkdir output dir ---- */
    {
        char mkdir_cmd[600];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", cfg.output_dir);
        (void)!system(mkdir_cmd);
    }

    /* ---- Initialize base-18 mapping for metadata ---- */
    init_base18_mapping();

    /* ---- Pass 1: Statistics ---- */
    printf("\n");
    LMStats lm_stats[LAT_COUNT][MODE_COUNT];
    StreamState final_ss;
    GlobalStats gstats;
    memset(&gstats, 0, sizeof(gstats));
    pass1_stats(&si, &cfg, views, &e8, lm_stats, &final_ss, &gstats);

    /* ---- Reset sieve ---- */
    sieve_iter_reset(&si);

    /* ---- Pass 2: Render ---- */
    pass2_render(&si, &cfg, views, &e8, lm_stats, &gstats);

    /* ---- Summary ---- */
    {
        int n_renders = 0;
        for (int l = 0; l < LAT_COUNT; l++) {
            if (cfg.lattice >= 0 && cfg.lattice != l) continue;
            for (int m = 0; m < MODE_COUNT; m++) {
                if (cfg.mode >= 0 && cfg.mode != m) continue;
                if (lm_stats[l][m].plot_count > 0) n_renders++;
            }
        }

        printf("================================================================\n");
        printf("  %d PNG(s) + JPEG preview(s) rendered\n", n_renders);
        printf("  Output directory: %s\n", cfg.output_dir);
        printf("  Each PNG embeds e8v3:* metadata (verify with e8_verify)\n");
        printf("================================================================\n");
    }

    /* Cleanup */
    sieve_iter_free(&si);

    return 0;
}
