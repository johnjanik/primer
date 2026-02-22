/*
 * e8_viz_v2.c — Multi-Lattice Prime Gap Visualizer v2.0
 *
 * Renders Ulam spiral of primes, filtered and colored by Lie group structure.
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
 *
 * Build: gcc -O3 -march=native -fopenmp -o e8_viz_v2 e8_viz_v2.c -lm -lpng
 */

#include "e8_common.h"
#include <png.h>

/* ================================================================
 * Configuration
 * ================================================================ */

#define MONSTER_DIM 196883

enum RenderMode { MODE_JORDAN = 0, MODE_TIERED, MODE_STROBE, MODE_TORSION, MODE_COUNT };
enum LatticeID  { LAT_E8 = 0, LAT_E7, LAT_E6, LAT_F4, LAT_G2, LAT_SO16, LAT_COUNT };

static const char *mode_names[]    = {"jordan", "tiered", "strobe", "torsion"};
static const char *lattice_names[] = {"E8", "E7", "E6", "F4", "G2", "SO16"};
/* static const int lattice_nroots[] = {240, 126, 72, 48, 12, 128}; */

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
        .max_primes      = 2000000000,
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
    const int  *e8_to_idx;      /* [E8_NUM_ROOTS]: E8 root → sublattice index (-1 if none) */
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
 * Tiered Coloring: map coherence to RGBA
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
    /* quality: 1.0 = in lattice (bright), 0.0 = pure leakage (dim) */
    double torsion = 1.0 - quality;  /* 0 = no torsion, 1 = max torsion */
    /* Map torsion [0,1] to red→green gradient */
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
            /* Cosine similarity of first-4 projection to nearest F4 root */
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
            lv->quality[ei] = sim * sim;  /* square for sharper transition */
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
            /* Find best cosine similarity to any sublattice root */
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
 * Render one (lattice, mode) combination
 * ================================================================ */

typedef struct { int64_t idx; double score; } ScoreEntry;

static void render(const Config *cfg,
                   const LatticeView *lv,
                   int mode,
                   const int64_t *primes, int64_t n_primes,
                   const int *e8_assignments,
                   const double *norm_gaps,
                   const float *coherence,     /* per-gap triplet coherence */
                   const int32_t *coord_x,
                   const int32_t *coord_y,
                   const E8Lattice *e8)
{
    int64_t n_gaps = n_primes - 1;
    int canvas_px = cfg->dpi * cfg->fig_inches;
    char buf[32];

    printf("  Rendering %s / %s...\n", lv->name, mode_names[mode]);
    tic();

    /* --- Determine which primes to plot --- */
    /* For lattice filtering: prime[i+1] inherits gap[i]'s E8 assignment */
    uint8_t *plot_mask = (uint8_t *)calloc(n_primes, 1);
    int64_t plot_count = 0;

    #pragma omp parallel for schedule(static) reduction(+:plot_count)
    for (int64_t i = 0; i < n_gaps; i++) {
        int64_t pi = i + 1;  /* prime index */
        int e8_idx = e8_assignments[i];

        /* Lattice membership: for E8, all roots are members */
        int in_lattice = lv->e8_is_member[e8_idx];

        /* For E8 itself (n_roots==240), plot everything */
        int plot = (lv->n_roots == 240) ? 1 : in_lattice;

        /* Strobe filter */
        if (mode == MODE_STROBE) {
            int phase = (int)(pi % MONSTER_DIM);
            int dist = abs(phase - cfg->strobe_phase);
            if (dist > MONSTER_DIM / 2) dist = MONSTER_DIM - dist;
            if (dist > cfg->strobe_epsilon) plot = 0;
        }

        if (plot) {
            plot_mask[pi] = 1;
            plot_count++;
        }
    }

    printf("    Plotting %s primes (%.1f%%)\n",
           fmt_comma(plot_count, buf, sizeof(buf)),
           100.0 * plot_count / n_primes);

    /* --- Compute coordinate range (over plotted primes) --- */
    int32_t min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    int first = 1;
    for (int64_t i = 1; i < n_primes; i++) {
        if (!plot_mask[i]) continue;
        if (first) { min_x = max_x = coord_x[i]; min_y = max_y = coord_y[i]; first = 0; continue; }
        if (coord_x[i] < min_x) min_x = coord_x[i];
        if (coord_x[i] > max_x) max_x = coord_x[i];
        if (coord_y[i] < min_y) min_y = coord_y[i];
        if (coord_y[i] > max_y) max_y = coord_y[i];
    }

    double range_x = (double)(max_x - min_x);
    double range_y = (double)(max_y - min_y);
    double range = fmax(range_x, range_y);
    if (range < 1.0) range = 1.0;
    double margin_frac = 0.02;
    double scale = (canvas_px * (1.0 - 2.0 * margin_frac)) / range;
    double offset_x = canvas_px * margin_frac - min_x * scale + (range - range_x) * scale * 0.5;
    double offset_y = canvas_px * margin_frac - min_y * scale + (range - range_y) * scale * 0.5;

    /* --- Allocate canvas (black background) --- */
    int64_t canvas_bytes = (int64_t)canvas_px * canvas_px * 3;
    uint8_t *canvas = (uint8_t *)calloc(canvas_bytes, 1);
    if (!canvas) {
        fprintf(stderr, "Failed to allocate canvas (%.1f MB)\n", canvas_bytes / 1e6);
        free(plot_mask);
        return;
    }

    /* --- Pass 1: Plot primes with selected coloring --- */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n_primes; i++) {
        if (!plot_mask[i]) continue;

        int px = (int)(coord_x[i] * scale + offset_x);
        int py = (int)(coord_y[i] * scale + offset_y);
        py = canvas_px - 1 - py;
        if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;

        int e8_idx = e8_assignments[i - 1];
        int sub_idx = lv->e8_to_idx[e8_idx];

        RGB c;
        switch (mode) {
            case MODE_JORDAN:
                if (sub_idx >= 0) {
                    c = jordan_to_color_range(lv->jordan_traces[sub_idx], lv->jmin, lv->jmax);
                } else {
                    c = (RGB){30, 30, 30};
                }
                break;

            case MODE_TIERED:
                c = tier_color((double)coherence[i - 1]);
                break;

            case MODE_STROBE:
                /* Color strobed primes by Jordan trace */
                if (sub_idx >= 0) {
                    c = jordan_to_color_range(lv->jordan_traces[sub_idx], lv->jmin, lv->jmax);
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

    /* --- Pass 2 (optional): Extract crystalline vertices + path --- */
    if (cfg->draw_path || cfg->n_vertices > 0) {
        /* Score each plotted gap by coherence */
        int heap_cap = cfg->n_vertices * 2;
        if (heap_cap > n_gaps) heap_cap = (int)n_gaps;
        if (heap_cap < 1) heap_cap = 1;
        ScoreEntry *heap = (ScoreEntry *)malloc(heap_cap * sizeof(ScoreEntry));
        int heap_size = 0;

        for (int64_t i = 0; i < n_gaps; i++) {
            if (!plot_mask[i + 1]) continue;
            double score = (double)coherence[i];
            if (score <= 0.0) continue;

            if (heap_size < heap_cap) {
                heap[heap_size].idx = i;
                heap[heap_size].score = score;
                heap_size++;
                /* Bubble up (min-heap) */
                int k = heap_size - 1;
                while (k > 0) {
                    int parent = (k - 1) / 2;
                    if (heap[parent].score > heap[k].score) {
                        ScoreEntry tmp = heap[parent]; heap[parent] = heap[k]; heap[k] = tmp;
                        k = parent;
                    } else break;
                }
            } else if (score > heap[0].score) {
                heap[0].idx = i;
                heap[0].score = score;
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

        /* Sort vertices by prime index for path drawing */
        for (int i = 0; i < heap_size - 1; i++) {
            for (int j = i + 1; j < heap_size; j++) {
                if (heap[j].idx < heap[i].idx) {
                    ScoreEntry tmp = heap[i]; heap[i] = heap[j]; heap[j] = tmp;
                }
            }
        }

        int actual_vertices = heap_size < cfg->n_vertices ? heap_size : cfg->n_vertices;

        /* Draw Hamiltonian path (thin cyan lines between consecutive vertices) */
        if (cfg->draw_path && actual_vertices > 1) {
            for (int v = 0; v < actual_vertices - 1; v++) {
                int64_t pi0 = heap[v].idx + 1;
                int64_t pi1 = heap[v + 1].idx + 1;
                int x0 = (int)(coord_x[pi0] * scale + offset_x);
                int y0 = canvas_px - 1 - (int)(coord_y[pi0] * scale + offset_y);
                int x1 = (int)(coord_x[pi1] * scale + offset_x);
                int y1 = canvas_px - 1 - (int)(coord_y[pi1] * scale + offset_y);
                draw_line(canvas, canvas_px, canvas_px, x0, y0, x1, y1, 0, 255, 200);
            }
        }

        /* Overlay vertices as white dots */
        int radius = cfg->dpi >= 600 ? 3 : 2;
        for (int v = 0; v < actual_vertices; v++) {
            int64_t pi = heap[v].idx + 1;
            int cx = (int)(coord_x[pi] * scale + offset_x);
            int cy = canvas_px - 1 - (int)(coord_y[pi] * scale + offset_y);

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx*dx + dy*dy > radius*radius) continue;
                    int px = cx + dx;
                    int py = cy + dy;
                    if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;
                    int64_t off = ((int64_t)py * canvas_px + px) * 3;
                    if (dx*dx + dy*dy <= (radius-1)*(radius-1)) {
                        canvas[off + 0] = 255; canvas[off + 1] = 255; canvas[off + 2] = 255;
                    } else {
                        canvas[off + 0] = 255; canvas[off + 1] = 255; canvas[off + 2] = 0;
                    }
                }
            }
        }

        printf("    %d crystalline vertices", actual_vertices);
        if (cfg->draw_path) printf(" + path");
        printf("\n");
        free(heap);
    }

    /* --- Write PNG --- */
    char suffix[64];
    if (cfg->max_primes >= 1000000)
        snprintf(suffix, sizeof(suffix), "%ldM", cfg->max_primes / 1000000);
    else
        snprintf(suffix, sizeof(suffix), "%ldk", cfg->max_primes / 1000);

    char png_path[1024];
    snprintf(png_path, sizeof(png_path), "%s/viz2_%s_%s_%s.png",
             cfg->output_dir, lattice_names[lv->n_roots == 240 ? 0 :
                                            lv->n_roots == 126 ? 1 :
                                            lv->n_roots == 72  ? 2 :
                                            lv->n_roots == 48  ? 3 :
                                            lv->n_roots == 12  ? 4 : 5],
             mode_names[mode], suffix);

    /* Use lattice name from the view */
    snprintf(png_path, sizeof(png_path), "%s/viz2_%s_%s_%s.png",
             cfg->output_dir, lv->name, mode_names[mode], suffix);

    write_png(png_path, canvas, canvas_px, canvas_px);

    FILE *check = fopen(png_path, "rb");
    long png_size = 0;
    if (check) { fseek(check, 0, SEEK_END); png_size = ftell(check); fclose(check); }
    printf("    → %s (%.1f MB, %.2fs)\n", png_path, png_size / 1e6, toc());

    free(canvas);
    free(plot_mask);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);
    int canvas_px = cfg.dpi * cfg.fig_inches;
    char b1[32];

    printf("================================================================\n");
    printf("  Multi-Lattice Prime Gap Visualizer v2.0\n");
    printf("================================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, b1, sizeof(b1)));
    printf("Canvas     : %d x %d px (%d DPI x %d in)\n", canvas_px, canvas_px, cfg.dpi, cfg.fig_inches);
    printf("Vertices   : %d%s\n", cfg.n_vertices, cfg.draw_path ? " + path" : "");
    printf("Mode       : %s\n", cfg.mode < 0 ? "ALL" : mode_names[cfg.mode]);
    printf("Lattice    : %s\n", cfg.lattice < 0 ? "ALL" : lattice_names[cfg.lattice]);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("================================================================\n\n");

    /* ---- Step 1: Generate primes ---- */
    printf("Step 1: Generating primes...\n");
    tic();
    int64_t n_primes;
    int64_t *primes = sieve_primes(cfg.max_primes, &n_primes);
    printf("  %s primes in %.2fs (range: 2 to %s)\n",
           fmt_comma(n_primes, b1, sizeof(b1)), toc(),
           fmt_comma(primes[n_primes - 1], b1, sizeof(b1)));

    int64_t n_gaps = n_primes - 1;

    /* ---- Step 2: Initialize all lattices ---- */
    printf("\nStep 2: Initializing lattices...\n");
    tic();

    E8Lattice e8;  e8_init(&e8);
    E7Lattice e7;  e7_init(&e7, &e8);
    E6Lattice e6;  e6_init(&e6, &e8);
    F4Lattice f4;  f4_init(&f4, &e8);
    G2Lattice g2;  g2_init(&g2, &e8);
    S16Lattice s16; s16_init(&s16, &e8);

    /* Build generic views */
    LatticeView views[LAT_COUNT];

    /* E8 — all roots are members */
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

    /* F4 — uses 4D projection, needs special quality computation */
    {
        /* F4 jordan_traces and norms live in f4 struct but are 4D.
         * We need 8D versions for the generic interface. Use the existing ones. */
        views[LAT_F4] = (LatticeView){
            .name = "F4", .n_roots = F4_NUM_ROOTS,
            .e8_is_member = f4.e8_is_f4, .e8_to_idx = f4.e8_to_f4,
            .jordan_traces = f4.jordan_traces, .norms = f4.norms
        };
        lattice_view_finalize(&views[LAT_F4], &e8);
        compute_quality_f4(&views[LAT_F4], &f4, &e8);
    }

    /* G2 — lives in 2D */
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

    /* ---- Step 3: E8 root assignments + normalized gaps ---- */
    printf("\nStep 3: Computing E8 assignments (%s gaps)...\n", fmt_comma(n_gaps, b1, sizeof(b1)));
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
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 4: Compute triplet coherence ---- */
    printf("\nStep 4: Computing triplet coherence...\n");
    tic();

    float *coherence = (float *)calloc(n_gaps, sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n_gaps - 1; i++) {
        /* Triplet: gaps i-1, i, i+1 */
        int r0 = e8_assignments[i - 1];
        int r1 = e8_assignments[i];
        int r2 = e8_assignments[i + 1];

        /* Sum vector in R^8 */
        double sum[E8_DIM];
        for (int d = 0; d < E8_DIM; d++) {
            sum[d] = e8.roots[r0][d] + e8.roots[r1][d] + e8.roots[r2][d];
        }

        /* Coherence = |sum|² / (|r0|² + |r1|² + |r2|²) = |sum|² / 6.0 */
        double sum_sq = 0;
        for (int d = 0; d < E8_DIM; d++) sum_sq += sum[d] * sum[d];
        coherence[i] = (float)(sum_sq / 6.0);
    }

    /* Stats */
    double coh_sum = 0, coh_max = 0;
    int64_t tier_counts[4] = {0, 0, 0, 0};
    for (int64_t i = 1; i < n_gaps - 1; i++) {
        double c = (double)coherence[i];
        coh_sum += c;
        if (c > coh_max) coh_max = c;
        if (c > 2.50) tier_counts[3]++;
        else if (c > 2.00) tier_counts[2]++;
        else if (c > 0.90) tier_counts[1]++;
        else tier_counts[0]++;
    }
    int64_t n_triplets = n_gaps - 2;
    printf("  Mean coherence: %.4f (max: %.4f)\n", coh_sum / n_triplets, coh_max);
    printf("  Tier 0 (Noise):           %s (%.1f%%)\n", fmt_comma(tier_counts[0], b1, sizeof(b1)),
           100.0 * tier_counts[0] / n_triplets);
    printf("  Tier 1 (Harmonic):        %s (%.1f%%)\n", fmt_comma(tier_counts[1], b1, sizeof(b1)),
           100.0 * tier_counts[1] / n_triplets);
    printf("  Tier 2 (Resonant):        %s (%.1f%%)\n", fmt_comma(tier_counts[2], b1, sizeof(b1)),
           100.0 * tier_counts[2] / n_triplets);
    printf("  Tier 3 (Transcendental):  %s (%.1f%%)\n", fmt_comma(tier_counts[3], b1, sizeof(b1)),
           100.0 * tier_counts[3] / n_triplets);
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 5: Compute Ulam spiral coordinates ---- */
    printf("\nStep 5: Computing Ulam coordinates...\n");
    tic();

    int32_t *coord_x = (int32_t *)malloc(n_primes * sizeof(int32_t));
    int32_t *coord_y = (int32_t *)malloc(n_primes * sizeof(int32_t));
    if (!coord_x || !coord_y) { fprintf(stderr, "malloc failed\n"); exit(1); }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_primes; i++) {
        ulam_coord(primes[i], &coord_x[i], &coord_y[i]);
    }
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 6: Render all selected (lattice, mode) combinations ---- */
    printf("\nStep 6: Rendering...\n");
    char mkdir_cmd[600];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", cfg.output_dir);
    (void)!system(mkdir_cmd);

    int n_renders = 0;
    for (int l = 0; l < LAT_COUNT; l++) {
        if (cfg.lattice >= 0 && cfg.lattice != l) continue;
        for (int m = 0; m < MODE_COUNT; m++) {
            if (cfg.mode >= 0 && cfg.mode != m) continue;
            render(&cfg, &views[l], m, primes, n_primes,
                   e8_assignments, norm_gaps, coherence,
                   coord_x, coord_y, &e8);
            n_renders++;
        }
    }

    /* ---- Summary ---- */
    printf("\n================================================================\n");
    printf("  %d PNG(s) rendered\n", n_renders);
    printf("  Output directory: %s\n", cfg.output_dir);
    printf("================================================================\n");

    /* Cleanup */
    free(coord_x);
    free(coord_y);
    free(coherence);
    free(e8_assignments);
    free(norm_gaps);
    free(primes);

    return 0;
}
