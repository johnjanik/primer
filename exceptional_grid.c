/*
 * exceptional_grid.c — 7-Panel Exceptional Chain Visualization (PPM output)
 *
 * Renders separate images for all exceptional Lie groups plus SO(16):
 *   E8 (240 roots), E7 (126), E6 (72), F4 (48), G2 (12),
 *   S16 (128 half-spinor), SO16 (112 D8 vector roots)
 *
 * Color scales:
 *   auto    — trace8 for E8/E7/E6/S16/SO16, jordan for F4, trace2 for G2 (default)
 *   jordan  — sum of first 4 coords, range [-2,+2]
 *   trace8  — sum of 8 coords, range [-4,+4]
 *   trace2  — sum of 2 coords, range [-2.5,+2.5]
 *   norm    — root norm, range [1, sqrt(3)]
 *
 * Usage:
 *   ./exceptional_grid --max-primes 2000000 --n-vertices 38 [--size 6000]
 *       [--color-scale auto] [--output exceptional_grid.ppm]
 *
 * Build: gcc -O3 -march=native -Wall -fopenmp -o exceptional_grid exceptional_grid.c -lm
 */

#include "e8_common.h"

/* ================================================================
 * Configuration
 * ================================================================ */

typedef enum {
    SCALE_AUTO = 0,
    SCALE_JORDAN,
    SCALE_TRACE8,
    SCALE_TRACE2,
    SCALE_NORM,
} ColorScale;

typedef struct {
    int64_t   max_primes;
    int       n_vertices;
    int       img_size;        /* total image size (each panel = img_size/2) */
    ColorScale color_scale;
    double    zoom;            /* 0 < zoom <= 1.0; fraction of area to show (default 1.0 = full) */
    char      output[512];
    char      prime_dir[512];
} Config;

static ColorScale parse_scale(const char *s)
{
    if (!strcmp(s, "jordan"))  return SCALE_JORDAN;
    if (!strcmp(s, "trace8"))  return SCALE_TRACE8;
    if (!strcmp(s, "trace2"))  return SCALE_TRACE2;
    if (!strcmp(s, "norm"))    return SCALE_NORM;
    return SCALE_AUTO;
}

static const char *scale_name(ColorScale s)
{
    switch (s) {
        case SCALE_JORDAN: return "jordan";
        case SCALE_TRACE8: return "trace8";
        case SCALE_TRACE2: return "trace2";
        case SCALE_NORM:   return "norm";
        default:           return "auto";
    }
}

static Config parse_args(int argc, char **argv)
{
    Config cfg = {
        .max_primes  = 2000000,
        .n_vertices  = 38,
        .img_size    = 6000,
        .color_scale = SCALE_AUTO,
        .zoom        = 1.0,
    };
    snprintf(cfg.output, sizeof(cfg.output), "exceptional_grid.ppm");
    snprintf(cfg.prime_dir, sizeof(cfg.prime_dir), "/home/john/mynotes/HodgedeRham");

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-primes") && i+1 < argc)
            cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--n-vertices") && i+1 < argc)
            cfg.n_vertices = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc)
            cfg.img_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--color-scale") && i+1 < argc)
            cfg.color_scale = parse_scale(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc)
            snprintf(cfg.output, sizeof(cfg.output), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--prime-dir") && i+1 < argc)
            snprintf(cfg.prime_dir, sizeof(cfg.prime_dir), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--zoom") && i+1 < argc) {
            cfg.zoom = atof(argv[++i]);
            if (cfg.zoom <= 0.0 || cfg.zoom > 1.0) {
                fprintf(stderr, "Error: --zoom must be in (0, 1.0] (area fraction)\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --max-primes N        (default 2000000)\n");
            printf("  --n-vertices N        (default 38)\n");
            printf("  --size N              total image size in px (default 6000)\n");
            printf("  --color-scale SCALE   auto|jordan|trace8|trace2|norm\n");
            printf("  --zoom FRAC           area fraction to show, e.g. 0.10 for central 10%% (default 1.0)\n");
            printf("  --output FILE         (default exceptional_grid.ppm)\n");
            printf("  --prime-dir DIR       (default /home/john/mynotes/HodgedeRham)\n");
            exit(0);
        }
    }
    return cfg;
}

/* ================================================================
 * Per-panel trace computation
 * ================================================================ */

/* Get the color value for a root, given lattice type and color scale */
static double panel_trace_e7(const E7Lattice *e7, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_TRACE8)
        return e7->jordan_traces[root_idx];  /* sum of 8 coords */
    if (scale == SCALE_JORDAN)
        return e7->roots[root_idx][0] + e7->roots[root_idx][1]
             + e7->roots[root_idx][2] + e7->roots[root_idx][3];
    if (scale == SCALE_TRACE2)
        return e7->roots[root_idx][0] + e7->roots[root_idx][1];
    if (scale == SCALE_NORM)
        return e7->norms[root_idx];
    return e7->jordan_traces[root_idx];
}

static double panel_trace_e6(const E6Lattice *e6, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_TRACE8)
        return e6->jordan_traces[root_idx];
    if (scale == SCALE_JORDAN)
        return e6->roots[root_idx][0] + e6->roots[root_idx][1]
             + e6->roots[root_idx][2] + e6->roots[root_idx][3];
    if (scale == SCALE_TRACE2)
        return e6->roots[root_idx][0] + e6->roots[root_idx][1];
    if (scale == SCALE_NORM)
        return e6->norms[root_idx];
    return e6->jordan_traces[root_idx];
}

static double panel_trace_f4(const F4Lattice *f4, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_JORDAN)
        return f4->jordan_traces[root_idx];  /* sum of 4 coords */
    if (scale == SCALE_TRACE8)
        return f4->jordan_traces[root_idx];
    if (scale == SCALE_TRACE2)
        return f4->roots[root_idx][0] + f4->roots[root_idx][1];
    if (scale == SCALE_NORM)
        return f4->norms[root_idx];
    return f4->jordan_traces[root_idx];
}

static double panel_trace_g2(const G2Lattice *g2, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_TRACE2)
        return g2->jordan_traces[root_idx];  /* sum of 2 coords */
    if (scale == SCALE_JORDAN || scale == SCALE_TRACE8)
        return g2->jordan_traces[root_idx];
    if (scale == SCALE_NORM)
        return g2->norms[root_idx];
    return g2->jordan_traces[root_idx];
}

static double panel_trace_s16(const S16Lattice *s16, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_TRACE8)
        return s16->jordan_traces[root_idx];  /* sum of 8 coords, range [-4,+4] */
    if (scale == SCALE_JORDAN)
        return s16->roots[root_idx][0] + s16->roots[root_idx][1]
             + s16->roots[root_idx][2] + s16->roots[root_idx][3];
    if (scale == SCALE_TRACE2)
        return s16->roots[root_idx][0] + s16->roots[root_idx][1];
    if (scale == SCALE_NORM)
        return s16->norms[root_idx];
    return s16->jordan_traces[root_idx];
}

static double panel_trace_e8(const E8Lattice *e8, int root_idx, ColorScale scale)
{
    double trace8 = 0;
    for (int d = 0; d < E8_DIM; d++)
        trace8 += e8->roots[root_idx][d];
    if (scale == SCALE_AUTO || scale == SCALE_TRACE8)
        return trace8;
    if (scale == SCALE_JORDAN)
        return e8->roots[root_idx][0] + e8->roots[root_idx][1]
             + e8->roots[root_idx][2] + e8->roots[root_idx][3];
    if (scale == SCALE_TRACE2)
        return e8->roots[root_idx][0] + e8->roots[root_idx][1];
    if (scale == SCALE_NORM) {
        double n2 = 0;
        for (int d = 0; d < E8_DIM; d++)
            n2 += e8->roots[root_idx][d] * e8->roots[root_idx][d];
        return sqrt(n2);
    }
    return trace8;
}

static double panel_trace_d8(const D8Lattice *d8, int root_idx, ColorScale scale)
{
    if (scale == SCALE_AUTO || scale == SCALE_TRACE8)
        return d8->jordan_traces[root_idx];
    if (scale == SCALE_JORDAN)
        return d8->roots[root_idx][0] + d8->roots[root_idx][1]
             + d8->roots[root_idx][2] + d8->roots[root_idx][3];
    if (scale == SCALE_TRACE2)
        return d8->roots[root_idx][0] + d8->roots[root_idx][1];
    if (scale == SCALE_NORM)
        return d8->norms[root_idx];
    return d8->jordan_traces[root_idx];
}

/* Color range for each panel under each scale */
static void panel_color_range(const char *panel, ColorScale scale,
                              double *jmin, double *jmax)
{
    if (scale == SCALE_JORDAN) {
        *jmin = -2.0; *jmax = 2.0;
    } else if (scale == SCALE_TRACE8) {
        *jmin = -4.0; *jmax = 4.0;
    } else if (scale == SCALE_TRACE2) {
        *jmin = -2.5; *jmax = 2.5;
    } else if (scale == SCALE_NORM) {
        *jmin = 1.0; *jmax = 1.8;
    } else {
        /* SCALE_AUTO: use natural range per panel */
        if (!strcmp(panel, "E8") || !strcmp(panel, "E7") || !strcmp(panel, "E6")
            || !strcmp(panel, "S16") || !strcmp(panel, "SO16")) {
            *jmin = -4.0; *jmax = 4.0;  /* trace8 */
        } else if (!strcmp(panel, "F4")) {
            *jmin = -2.0; *jmax = 2.0;  /* jordan */
        } else {
            *jmin = -2.5; *jmax = 2.5;  /* trace2 for G2 */
        }
    }
}

/* ================================================================
 * Draw a filled circle with edge (same as f4_crystalline_grid.c)
 * ================================================================ */

static void draw_circle(unsigned char *buf, int W, int H,
                        int cx, int cy, int radius,
                        unsigned char fr, unsigned char fg, unsigned char fb,
                        unsigned char er, unsigned char eg, unsigned char eb)
{
    int r2 = radius * radius;
    int inner_r2 = (radius - 1) * (radius - 1);
    for (int dy = -radius; dy <= radius; dy++) {
        int py = cy + dy;
        if (py < 0 || py >= H) continue;
        for (int dx = -radius; dx <= radius; dx++) {
            int px = cx + dx;
            if (px < 0 || px >= W) continue;
            int d2 = dx*dx + dy*dy;
            if (d2 <= r2) {
                int off = (py * W + px) * 3;
                if (d2 > inner_r2) {
                    buf[off] = er; buf[off+1] = eg; buf[off+2] = eb;
                } else {
                    buf[off] = fr; buf[off+1] = fg; buf[off+2] = fb;
                }
            }
        }
    }
}

/* ================================================================
 * Render one panel into a subregion of the image buffer
 * ================================================================ */

typedef struct {
    int      *is_member;       /* 1 if gap passes lattice filter */
    double   *trace_values;    /* trace value per gap (NAN if not member) */
    int64_t  n_gaps;
    int64_t  n_mapped;
    const char *name;
} PanelData;

static void render_panel(
    unsigned char *img, int total_w, int total_h,
    int panel_x0, int panel_y0, int panel_w, int panel_h,
    const PanelData *pd,
    const int32_t *ux, const int32_t *uy, int64_t n_primes,
    double jmin, double jmax,
    const int *vertex_gap_idx, int n_vertices,
    double zoom)
{
    /* Compute full coordinate range for member primes */
    int32_t min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    int first = 1;
    for (int64_t i = 0; i < pd->n_gaps; i++) {
        if (!pd->is_member[i]) continue;
        int64_t pidx = i + 1;
        if (first) {
            min_x = max_x = ux[pidx];
            min_y = max_y = uy[pidx];
            first = 0;
        } else {
            if (ux[pidx] < min_x) min_x = ux[pidx];
            if (ux[pidx] > max_x) max_x = ux[pidx];
            if (uy[pidx] < min_y) min_y = uy[pidx];
            if (uy[pidx] > max_y) max_y = uy[pidx];
        }
    }

    if (first) return;  /* No member primes */

    int coord_range_x = max_x - min_x + 1;
    int coord_range_y = max_y - min_y + 1;
    int coord_range = coord_range_x > coord_range_y ? coord_range_x : coord_range_y;
    if (coord_range < 1) coord_range = 1;

    /* Apply zoom: shrink visible coordinate range by sqrt(zoom) on each axis,
     * centered on the origin (Ulam coord 0,0 = integer 1).
     * zoom = area fraction, so linear scale factor = sqrt(zoom). */
    if (zoom < 1.0 && zoom > 0.0) {
        double lin = sqrt(zoom);
        int half_range = (int)(coord_range * lin * 0.5);
        if (half_range < 1) half_range = 1;
        coord_range = 2 * half_range;
        /* Center on origin (0,0) instead of bounding-box center */
        min_x = -half_range; max_x = half_range;
        min_y = -half_range; max_y = half_range;
    }

    double scale = (double)(panel_w - 20) / (double)coord_range;
    int cx_off = panel_x0 + panel_w / 2;
    int cy_off = panel_y0 + panel_h / 2;
    int mid_x = (min_x + max_x) / 2;
    int mid_y = (min_y + max_y) / 2;

    /* Plot member primes colored by trace */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < pd->n_gaps; i++) {
        if (!pd->is_member[i]) continue;
        int64_t pidx = i + 1;
        int px = (int)((ux[pidx] - mid_x) * scale) + cx_off;
        int py = (int)((uy[pidx] - mid_y) * scale) + cy_off;
        if (px < panel_x0 || px >= panel_x0 + panel_w) continue;
        if (py < panel_y0 || py >= panel_y0 + panel_h) continue;

        double tv = pd->trace_values[i];
        if (tv != tv) continue;  /* NaN check */

        RGB c = jordan_to_color_range(tv, jmin, jmax);

        int off = (py * total_w + px) * 3;
        unsigned char old_r = img[off], old_g = img[off+1], old_b = img[off+2];
        img[off]   = (unsigned char)(0.7 * c.r + 0.3 * old_r);
        img[off+1] = (unsigned char)(0.7 * c.g + 0.3 * old_g);
        img[off+2] = (unsigned char)(0.7 * c.b + 0.3 * old_b);
    }

    /* Overlay vertices (if any) */
    int radius = (panel_w > 2000) ? 3 : 2;
    for (int v = 0; v < n_vertices; v++) {
        int gi = vertex_gap_idx[v];
        if (gi < 0 || gi >= pd->n_gaps) continue;
        if (!pd->is_member[gi]) continue;
        int64_t pidx = gi + 1;
        if (pidx >= n_primes) continue;
        int px = (int)((ux[pidx] - mid_x) * scale) + cx_off;
        int py = (int)((uy[pidx] - mid_y) * scale) + cy_off;
        if (px < panel_x0 || px >= panel_x0 + panel_w) continue;
        if (py < panel_y0 || py >= panel_y0 + panel_h) continue;
        draw_circle(img, total_w, total_h, px, py, radius,
                    255, 255, 255, 255, 255, 0);
    }
}

/* ================================================================
 * Draw panel label (simple pixel text via small 5x7 font)
 * ================================================================ */

/* Minimal 5-wide bitmap font for A-Z, 0-9, space, parens, percent, period */
static const uint8_t font5x7[][7] = {
    /* space  */ {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    /* 0      */ {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E},
    /* 1      */ {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
    /* 2      */ {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F},
    /* 3      */ {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E},
    /* 4      */ {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02},
    /* 5      */ {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
    /* 6      */ {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E},
    /* 7      */ {0x1F,0x01,0x02,0x04,0x04,0x04,0x04},
    /* 8      */ {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E},
    /* 9      */ {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
    /* (      */ {0x02,0x04,0x08,0x08,0x08,0x04,0x02},
    /* )      */ {0x08,0x04,0x02,0x02,0x02,0x04,0x08},
    /* %      */ {0x18,0x19,0x02,0x04,0x08,0x13,0x03},
    /* .      */ {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C},
    /* A=15   */ {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11},
    /* B      */ {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E},
    /* C      */ {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E},
    /* D      */ {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C},
    /* E      */ {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F},
    /* F      */ {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10},
    /* G      */ {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F},
    /* H      */ {0x11,0x11,0x11,0x1F,0x11,0x11,0x11},
    /* I      */ {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E},
    /* J      */ {0x07,0x02,0x02,0x02,0x02,0x12,0x0C},
    /* K      */ {0x11,0x12,0x14,0x18,0x14,0x12,0x11},
    /* L      */ {0x10,0x10,0x10,0x10,0x10,0x10,0x1F},
    /* M      */ {0x11,0x1B,0x15,0x15,0x11,0x11,0x11},
    /* N      */ {0x11,0x19,0x15,0x13,0x11,0x11,0x11},
    /* O      */ {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E},
    /* P      */ {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10},
    /* Q      */ {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D},
    /* R      */ {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11},
    /* S      */ {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E},
    /* T      */ {0x1F,0x04,0x04,0x04,0x04,0x04,0x04},
    /* U      */ {0x11,0x11,0x11,0x11,0x11,0x11,0x0E},
    /* V      */ {0x11,0x11,0x11,0x11,0x0A,0x0A,0x04},
    /* W      */ {0x11,0x11,0x11,0x15,0x15,0x1B,0x11},
    /* X      */ {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11},
    /* Y      */ {0x11,0x11,0x0A,0x04,0x04,0x04,0x04},
    /* Z      */ {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F},
};

static int font_index(char c)
{
    if (c == ' ') return 0;
    if (c >= '0' && c <= '9') return 1 + (c - '0');
    if (c == '(') return 11;
    if (c == ')') return 12;
    if (c == '%') return 13;
    if (c == '.') return 14;
    if (c >= 'A' && c <= 'Z') return 15 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 15 + (c - 'a'); /* lowercase → uppercase */
    return 0; /* space for unknown */
}

static void draw_text(unsigned char *img, int W, int H,
                      int x0, int y0, const char *text, int scale_factor,
                      unsigned char r, unsigned char g, unsigned char b)
{
    int x = x0;
    for (const char *p = text; *p; p++) {
        int fi = font_index(*p);
        for (int row = 0; row < 7; row++) {
            uint8_t bits = font5x7[fi][row];
            for (int col = 0; col < 5; col++) {
                if (bits & (0x10 >> col)) {
                    for (int sy = 0; sy < scale_factor; sy++) {
                        for (int sx = 0; sx < scale_factor; sx++) {
                            int px = x + col * scale_factor + sx;
                            int py = y0 + row * scale_factor + sy;
                            if (px >= 0 && px < W && py >= 0 && py < H) {
                                int off = (py * W + px) * 3;
                                img[off] = r; img[off+1] = g; img[off+2] = b;
                            }
                        }
                    }
                }
            }
        }
        x += 6 * scale_factor;
    }
}

/* ================================================================
 * Simple EFT scoring for crystalline vertices
 * ================================================================ */

typedef struct { double score; int index; } ScoreEntry2;

static void heap_push2(ScoreEntry2 *heap, int *size, int cap, ScoreEntry2 e)
{
    if (*size < cap) {
        heap[*size] = e; (*size)++;
        int i = *size - 1;
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap[parent].score > heap[i].score) {
                ScoreEntry2 tmp = heap[parent]; heap[parent] = heap[i]; heap[i] = tmp;
                i = parent;
            } else break;
        }
    } else if (e.score > heap[0].score) {
        heap[0] = e;
        int i = 0;
        for (;;) {
            int l = 2*i+1, r = 2*i+2, s = i;
            if (l < cap && heap[l].score < heap[s].score) s = l;
            if (r < cap && heap[r].score < heap[s].score) s = r;
            if (s == i) break;
            ScoreEntry2 tmp = heap[i]; heap[i] = heap[s]; heap[s] = tmp;
            i = s;
        }
    }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);

    char buf1[32];
    printf("Exceptional Chain Grid Visualization (C/OpenMP)\n");
    printf("==================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, buf1, sizeof(buf1)));
    printf("Image      : %d x %d px (7 separate files)\n",
           cfg.img_size, cfg.img_size);
    printf("Color scale: %s\n", scale_name(cfg.color_scale));
    if (cfg.zoom < 1.0)
        printf("Zoom       : %.2f%% area (linear scale %.2f%%)\n",
               100.0 * cfg.zoom, 100.0 * sqrt(cfg.zoom));
    else
        printf("Zoom       : full (no zoom)\n");
    printf("Vertices   : %d\n", cfg.n_vertices);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("==================================================\n\n");

    /* ---- Step 1: Generate primes ---- */
    printf("Generating primes via sieve...\n");
    tic();
    int64_t n_primes;
    int64_t *primes = sieve_primes(cfg.max_primes, &n_primes);
    printf("  %s primes in %.2fs\n", fmt_comma(n_primes, buf1, sizeof(buf1)), toc());
    int64_t n_gaps = n_primes - 1;

    /* ---- Step 2: Initialize all lattices ---- */
    printf("Initializing E8 + E7 + E6 + F4 + G2 + S16 + SO16(D8)...\n");
    tic();
    E8Lattice  e8;   e8_init(&e8);
    E7Lattice  e7;   e7_init(&e7, &e8);
    E6Lattice  e6;   e6_init(&e6, &e8);
    F4Lattice  f4;   f4_init(&f4, &e8);
    G2Lattice  g2;   g2_init(&g2, &e8);
    S16Lattice s16;  s16_init(&s16, &e8);
    D8Lattice  d8;   d8_init(&d8, &e8);
    printf("  Done in %.2fs\n", toc());

    /* Count E8→sublattice mappings */
    int n_e7 = 0, n_e6 = 0, n_f4 = 0, n_g2 = 0, n_s16 = 0, n_d8 = 0;
    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        if (e7.e8_is_e7[i]) n_e7++;
        if (e6.e8_is_e6[i]) n_e6++;
        if (f4.e8_is_f4[i]) n_f4++;
        if (g2.e8_is_g2[i]) n_g2++;
        if (s16.e8_is_s16[i]) n_s16++;
        if (d8.e8_is_d8[i]) n_d8++;
    }
    printf("  E8→E7: %d/240, E8→E6: %d/240, E8→F4: %d/240\n", n_e7, n_e6, n_f4);
    printf("  E8→G2: %d/240, E8→S16: %d/240, E8→SO16(D8): %d/240\n", n_g2, n_s16, n_d8);

    /* ---- Step 3: E8 assignments + per-lattice filtering ---- */
    printf("Computing E8 assignments + lattice filtering...\n");
    tic();

    int *e8_assignments = (int *)malloc(n_gaps * sizeof(int));
    double *norm_gaps    = (double *)malloc(n_gaps * sizeof(double));

    /* Per-panel data */
    int *is_e8     = (int *)calloc(n_gaps, sizeof(int));
    int *is_e7     = (int *)calloc(n_gaps, sizeof(int));
    int *is_e6     = (int *)calloc(n_gaps, sizeof(int));
    int *is_f4     = (int *)calloc(n_gaps, sizeof(int));
    int *is_g2     = (int *)calloc(n_gaps, sizeof(int));
    int *is_s16    = (int *)calloc(n_gaps, sizeof(int));
    int *is_d8     = (int *)calloc(n_gaps, sizeof(int));
    double *tv_e8  = (double *)malloc(n_gaps * sizeof(double));
    double *tv_e7  = (double *)malloc(n_gaps * sizeof(double));
    double *tv_e6  = (double *)malloc(n_gaps * sizeof(double));
    double *tv_f4  = (double *)malloc(n_gaps * sizeof(double));
    double *tv_g2  = (double *)malloc(n_gaps * sizeof(double));
    double *tv_s16 = (double *)malloc(n_gaps * sizeof(double));
    double *tv_d8  = (double *)malloc(n_gaps * sizeof(double));

    int64_t cnt_e8 = 0, cnt_e7 = 0, cnt_e6 = 0, cnt_f4 = 0, cnt_g2 = 0, cnt_s16 = 0, cnt_d8 = 0;

    #pragma omp parallel for schedule(static) \
        reduction(+:cnt_e8,cnt_e7,cnt_e6,cnt_f4,cnt_g2,cnt_s16,cnt_d8)
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i+1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        norm_gaps[i] = gap / log_p;
        e8_assignments[i] = e8_assign_root(&e8, norm_gaps[i]);

        int ei = e8_assignments[i];

        /* NaN sentinel */
        tv_e8[i] = tv_e7[i] = tv_e6[i] = tv_f4[i] = tv_g2[i] = tv_s16[i] = tv_d8[i] = 0.0 / 0.0;

        /* E8: every gap maps to an E8 root (100% coverage) */
        is_e8[i] = 1; cnt_e8++;
        tv_e8[i] = panel_trace_e8(&e8, ei, cfg.color_scale);

        if (e7.e8_is_e7[ei] && e7.e8_to_e7[ei] >= 0) {
            is_e7[i] = 1; cnt_e7++;
            tv_e7[i] = panel_trace_e7(&e7, e7.e8_to_e7[ei], cfg.color_scale);
        }
        if (e6.e8_is_e6[ei] && e6.e8_to_e6[ei] >= 0) {
            is_e6[i] = 1; cnt_e6++;
            tv_e6[i] = panel_trace_e6(&e6, e6.e8_to_e6[ei], cfg.color_scale);
        }
        if (f4.e8_is_f4[ei] && f4.e8_to_f4[ei] >= 0) {
            is_f4[i] = 1; cnt_f4++;
            tv_f4[i] = panel_trace_f4(&f4, f4.e8_to_f4[ei], cfg.color_scale);
        }
        if (g2.e8_is_g2[ei] && g2.e8_to_g2[ei] >= 0) {
            is_g2[i] = 1; cnt_g2++;
            tv_g2[i] = panel_trace_g2(&g2, g2.e8_to_g2[ei], cfg.color_scale);
        }
        if (s16.e8_is_s16[ei] && s16.e8_to_s16[ei] >= 0) {
            is_s16[i] = 1; cnt_s16++;
            tv_s16[i] = panel_trace_s16(&s16, s16.e8_to_s16[ei], cfg.color_scale);
        }
        if (d8.e8_is_d8[ei] && d8.e8_to_d8[ei] >= 0) {
            is_d8[i] = 1; cnt_d8++;
            tv_d8[i] = panel_trace_d8(&d8, d8.e8_to_d8[ei], cfg.color_scale);
        }
    }
    printf("  Done in %.2fs\n", toc());
    {
        char b1[32], b2[32];
        printf("  E8: %s (%.1f%%), E7: %s (%.1f%%)\n",
               fmt_comma(cnt_e8, b1, sizeof(b1)), 100.0*cnt_e8/n_gaps,
               fmt_comma(cnt_e7, b2, sizeof(b2)), 100.0*cnt_e7/n_gaps);
        printf("  E6: %s (%.1f%%), F4: %s (%.1f%%)\n",
               fmt_comma(cnt_e6, b1, sizeof(b1)), 100.0*cnt_e6/n_gaps,
               fmt_comma(cnt_f4, b2, sizeof(b2)), 100.0*cnt_f4/n_gaps);
        printf("  G2: %s (%.1f%%), S16: %s (%.1f%%)\n",
               fmt_comma(cnt_g2, b1, sizeof(b1)), 100.0*cnt_g2/n_gaps,
               fmt_comma(cnt_s16, b2, sizeof(b2)), 100.0*cnt_s16/n_gaps);
        printf("  SO16(D8): %s (%.1f%%)\n",
               fmt_comma(cnt_d8, b1, sizeof(b1)), 100.0*cnt_d8/n_gaps);
    }

    /* ---- Step 4: Simple vertex scoring (F4 EFT based) ---- */
    printf("Computing F4-EFT vertices...\n");
    tic();

    double spec_re[F4_NUM_ROOTS] = {0}, spec_im[F4_NUM_ROOTS] = {0};
    for (int64_t n = 0; n < n_gaps; n++) {
        if (!is_f4[n]) continue;
        int fi = f4.e8_to_f4[e8_assignments[n]];
        if (fi < 0) continue;
        double fluct = norm_gaps[n] - 1.0;
        double chi = f4.characters[fi];
        double phase = 2.0 * M_PI * f4.norms[fi] / sqrt(2.0)
                      * (double)n / (double)n_gaps;
        spec_re[fi] += fluct * chi * cos(phase);
        spec_im[fi] += fluct * chi * sin(phase);
    }
    double power[F4_NUM_ROOTS];
    for (int i = 0; i < F4_NUM_ROOTS; i++)
        power[i] = spec_re[i]*spec_re[i] + spec_im[i]*spec_im[i];

    ScoreEntry2 *heap = (ScoreEntry2 *)malloc(cfg.n_vertices * sizeof(ScoreEntry2));
    int heap_size = 0;
    for (int64_t n = 0; n < n_gaps; n++) {
        if (!is_f4[n]) continue;
        int fi = f4.e8_to_f4[e8_assignments[n]];
        if (fi < 0) continue;
        double sc = power[fi];
        double jt = f4.jordan_traces[fi];
        if (fabs(fabs(jt) - 1.0) < 0.2) sc *= 2.0;
        if (sc > 0) {
            ScoreEntry2 entry = { sc, (int)n };
            heap_push2(heap, &heap_size, cfg.n_vertices, entry);
        }
    }
    int *vertex_idx = (int *)malloc(cfg.n_vertices * sizeof(int));
    int actual_vertices = heap_size;
    for (int i = 0; i < heap_size; i++)
        vertex_idx[i] = heap[i].index;
    free(heap);
    printf("  %d vertices in %.2fs\n", actual_vertices, toc());

    /* ---- Step 5: Ulam coordinates ---- */
    printf("Computing Ulam spiral coordinates...\n");
    tic();
    int32_t *ux = (int32_t *)malloc(n_primes * sizeof(int32_t));
    int32_t *uy = (int32_t *)malloc(n_primes * sizeof(int32_t));
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_primes; i++)
        ulam_coord(primes[i], &ux[i], &uy[i]);
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 6: Render 7 separate images ---- */
    struct {
        const char *name;
        int *is_member;
        double *trace_values;
        int64_t n_mapped;
    } panels[7] = {
        {"E8",   is_e8,  tv_e8,  cnt_e8},
        {"E7",   is_e7,  tv_e7,  cnt_e7},
        {"E6",   is_e6,  tv_e6,  cnt_e6},
        {"F4",   is_f4,  tv_f4,  cnt_f4},
        {"G2",   is_g2,  tv_g2,  cnt_g2},
        {"S16",  is_s16, tv_s16, cnt_s16},
        {"SO16", is_d8,  tv_d8,  cnt_d8},
    };
    int root_counts[7] = {E8_NUM_ROOTS, E7_NUM_ROOTS, E6_NUM_ROOTS, F4_NUM_ROOTS, G2_NUM_ROOTS, S16_NUM_ROOTS, D8_NUM_ROOTS};

    size_t buf_size = (size_t)cfg.img_size * cfg.img_size * 3;

    for (int p = 0; p < 7; p++) {
        printf("Rendering %s (%d roots, %s mapped) %dx%d...\n",
               panels[p].name, root_counts[p],
               fmt_comma(panels[p].n_mapped, buf1, sizeof(buf1)),
               cfg.img_size, cfg.img_size);
        tic();

        unsigned char *img = (unsigned char *)calloc(buf_size, 1);
        if (!img) { fprintf(stderr, "malloc failed for image\n"); exit(1); }

        PanelData pd = {
            .is_member    = panels[p].is_member,
            .trace_values = panels[p].trace_values,
            .n_gaps       = n_gaps,
            .n_mapped     = panels[p].n_mapped,
            .name         = panels[p].name,
        };

        double jmin, jmax;
        panel_color_range(panels[p].name, cfg.color_scale, &jmin, &jmax);

        render_panel(img, cfg.img_size, cfg.img_size,
                     0, 0, cfg.img_size, cfg.img_size,
                     &pd, ux, uy, n_primes, jmin, jmax,
                     vertex_idx, actual_vertices, cfg.zoom);

        /* Draw label */
        char label[64];
        snprintf(label, sizeof(label), "%s (%d) %.1f%%",
                 panels[p].name, root_counts[p],
                 100.0 * panels[p].n_mapped / n_gaps);
        int font_scale = (cfg.img_size > 4000) ? 3 : 2;
        draw_text(img, cfg.img_size, cfg.img_size,
                  10, 10, label, font_scale,
                  200, 200, 200);

        printf("  Rendered in %.2fs\n", toc());

        /* Build output filename: e.g. "exceptional_E7.ppm" or
         * from --output "foo.ppm" → "foo_E7.ppm" */
        char out_path[600];
        {
            /* Find last '.' in cfg.output */
            const char *dot = strrchr(cfg.output, '.');
            if (dot) {
                int prefix_len = (int)(dot - cfg.output);
                snprintf(out_path, sizeof(out_path), "%.*s_%s%s",
                         prefix_len, cfg.output, panels[p].name, dot);
            } else {
                snprintf(out_path, sizeof(out_path), "%s_%s.ppm",
                         cfg.output, panels[p].name);
            }
        }

        printf("  Writing %s...\n", out_path);
        tic();
        FILE *fp = fopen(out_path, "wb");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", out_path); exit(1); }
        fprintf(fp, "P6\n%d %d\n255\n", cfg.img_size, cfg.img_size);
        fwrite(img, 1, buf_size, fp);
        fclose(fp);
        printf("  Written in %.2fs (%zu bytes)\n", toc(), buf_size + 20);

        free(img);
    }

    /* ---- Summary ---- */
    printf("\n==================================================\n");
    printf("Exceptional Chain — 7 Separate Images\n");
    {
        char b1[32], b2[32];
        printf("  E8:  %s gaps (%.1f%%), E7: %s gaps (%.1f%%)\n",
               fmt_comma(cnt_e8, b1, sizeof(b1)), 100.0*cnt_e8/n_gaps,
               fmt_comma(cnt_e7, b2, sizeof(b2)), 100.0*cnt_e7/n_gaps);
        printf("  E6:  %s gaps (%.1f%%), F4: %s gaps (%.1f%%)\n",
               fmt_comma(cnt_e6, b1, sizeof(b1)), 100.0*cnt_e6/n_gaps,
               fmt_comma(cnt_f4, b2, sizeof(b2)), 100.0*cnt_f4/n_gaps);
        printf("  G2:  %s gaps (%.1f%%), S16: %s gaps (%.1f%%)\n",
               fmt_comma(cnt_g2, b1, sizeof(b1)), 100.0*cnt_g2/n_gaps,
               fmt_comma(cnt_s16, b2, sizeof(b2)), 100.0*cnt_s16/n_gaps);
        printf("  SO16(D8): %s gaps (%.1f%%)\n",
               fmt_comma(cnt_d8, b1, sizeof(b1)), 100.0*cnt_d8/n_gaps);
    }
    printf("  Vertices: %d\n", actual_vertices);
    printf("  Color: %s\n", scale_name(cfg.color_scale));
    printf("==================================================\n");

    /* Cleanup */
    free(vertex_idx);
    free(is_e8); free(is_e7); free(is_e6); free(is_f4); free(is_g2); free(is_s16); free(is_d8);
    free(tv_e8); free(tv_e7); free(tv_e6); free(tv_f4); free(tv_g2); free(tv_s16); free(tv_d8);
    free(e8_assignments); free(norm_gaps);
    free(ux); free(uy); free(primes);

    return 0;
}
