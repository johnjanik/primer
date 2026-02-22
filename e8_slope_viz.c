/*
 * e8_slope_viz.c — E8 Projection Slope Visualization (C/OpenMP)
 *
 * Produces a PNG image of primes in the Ulam spiral, colored by E8 root
 * projection slope. Replaces e8_slope_coloring.py for 100M+ primes.
 *
 * Usage:
 *   ./e8_slope_viz [--max-primes N] [--dpi D] [--size S] [--csv] [--ppm] [--from-files DIR]
 *
 * By default, generates primes via segmented sieve (no file dependency).
 * Use --from-files DIR to load from primes1.txt..primes50.txt instead.
 *
 * Output:
 *   spiral_outputs/e8_slope_<N>.png   (default)
 *   spiral_outputs/e8_slope_<N>.ppm   (if --ppm)
 *   spiral_outputs/e8_slope_data.csv  (if --csv)
 *
 * Build: gcc -O3 -march=native -fopenmp -o e8_slope_viz e8_slope_viz.c -lm -lpng
 */

#include "e8_common.h"
#include <png.h>

/* ================================================================
 * Configuration
 * ================================================================ */

typedef struct {
    const char *primes_dir;  /* NULL = use sieve (default) */
    int64_t     max_primes;
    int         dpi;
    int         fig_inches;
    int         emit_csv;
    int         emit_ppm;
} Config;

static Config parse_args(int argc, char **argv)
{
    Config cfg = {
        .primes_dir = NULL,  /* default: generate via sieve */
        .max_primes = 100000000,
        .dpi        = 1200,
        .fig_inches = 24,
        .emit_csv   = 0,
        .emit_ppm   = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--from-files") && i+1 < argc) cfg.primes_dir = argv[++i];
        else if (!strcmp(argv[i], "--max-primes") && i+1 < argc) cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dpi") && i+1 < argc) cfg.dpi = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc) cfg.fig_inches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")) cfg.emit_csv = 1;
        else if (!strcmp(argv[i], "--ppm")) cfg.emit_ppm = 1;
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
    }
    return cfg;
}

/* ================================================================
 * PNG Writer (streaming, row-at-a-time — no extra memory)
 * ================================================================ */

static void write_png(const char *path, const uint8_t *canvas, int width, int height)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno)); exit(1); }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fprintf(stderr, "png_create_write_struct failed\n"); exit(1); }

    png_infop info = png_create_info_struct(png);
    if (!info) { fprintf(stderr, "png_create_info_struct failed\n"); exit(1); }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "PNG write error\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(1);
    }

    png_init_io(png, fp);

    /* Best compression for large images */
    png_set_compression_level(png, 6);
    png_set_filter(png, 0, PNG_FILTER_SUB);

    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    /* Write row by row (no need to allocate row pointers array) */
    int64_t row_bytes = (int64_t)width * 3;
    for (int y = 0; y < height; y++) {
        png_write_row(png, (png_const_bytep)(canvas + y * row_bytes));
    }

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    Config cfg = parse_args(argc, argv);
    int canvas_px = cfg.dpi * cfg.fig_inches;

    char buf1[32];
    printf("E8 Slope Visualization (C/OpenMP)\n");
    printf("==================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, buf1, sizeof(buf1)));
    printf("Source     : %s\n", cfg.primes_dir ? cfg.primes_dir : "segmented sieve");
    printf("Canvas     : %d x %d px (%d DPI x %d in)\n", canvas_px, canvas_px, cfg.dpi, cfg.fig_inches);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("==================================================\n\n");

    /* ---- Step 1: Get primes ---- */
    int64_t n_primes;
    int64_t *primes;
    if (cfg.primes_dir) {
        printf("Loading primes from files...\n");
        tic();
        primes = load_primes(cfg.primes_dir, cfg.max_primes, &n_primes);
        printf("  Loaded %s primes in %.2fs\n", fmt_comma(n_primes, buf1, sizeof(buf1)), toc());
    } else {
        printf("Generating primes via sieve...\n");
        tic();
        primes = sieve_primes(cfg.max_primes, &n_primes);
        printf("  Generated %s primes in %.2fs\n", fmt_comma(n_primes, buf1, sizeof(buf1)), toc());
    }
    printf("  Range: 2 to %s\n", fmt_comma(primes[n_primes - 1], buf1, sizeof(buf1)));

    int64_t n_gaps = n_primes - 1;

    /* ---- Step 2: Initialize E8 lattice ---- */
    E8Lattice e8;
    e8_init(&e8);

    /* ---- Step 3: Compute gaps, normalized gaps, root assignments, slopes ---- */
    printf("Computing E8 root assignments (%s gaps)...\n", fmt_comma(n_gaps, buf1, sizeof(buf1)));
    tic();

    float *slopes = (float *)malloc(n_primes * sizeof(float));
    if (!slopes) { fprintf(stderr, "malloc failed for slopes\n"); exit(1); }
    slopes[0] = 0.0f;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i + 1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        double norm_gap = gap / log_p;
        int root_idx = e8_assign_root(&e8, norm_gap);
        slopes[i + 1] = (float)e8.slopes[root_idx];
    }
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 4: Compute Ulam spiral coordinates ---- */
    printf("Computing Ulam spiral coordinates...\n");
    tic();
    int32_t *coord_x = (int32_t *)malloc(n_primes * sizeof(int32_t));
    int32_t *coord_y = (int32_t *)malloc(n_primes * sizeof(int32_t));
    if (!coord_x || !coord_y) { fprintf(stderr, "malloc failed for coords\n"); exit(1); }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_primes; i++) {
        ulam_coord(primes[i], &coord_x[i], &coord_y[i]);
    }
    printf("  Done in %.2fs\n", toc());

    /* Find coordinate range for scaling */
    int32_t min_x = coord_x[0], max_x = coord_x[0];
    int32_t min_y = coord_y[0], max_y = coord_y[0];
    for (int64_t i = 1; i < n_primes; i++) {
        if (coord_x[i] < min_x) min_x = coord_x[i];
        if (coord_x[i] > max_x) max_x = coord_x[i];
        if (coord_y[i] < min_y) min_y = coord_y[i];
        if (coord_y[i] > max_y) max_y = coord_y[i];
    }
    printf("  Coord range: x=[%d, %d] y=[%d, %d]\n", min_x, max_x, min_y, max_y);

    /* ---- Step 5: Optional CSV output ---- */
    if (cfg.emit_csv) {
        printf("Writing CSV (first 100k entries)...\n");
        tic();
        (void)!system("mkdir -p /home/john/mynotes/HodgedeRham/spiral_outputs");
        FILE *csv = fopen("/home/john/mynotes/HodgedeRham/spiral_outputs/e8_slope_data.csv", "w");
        if (csv) {
            fprintf(csv, "prime,ulam_x,ulam_y,slope\n");
            int64_t csv_limit = n_primes < 100000 ? n_primes : 100000;
            for (int64_t i = 1; i < csv_limit; i++) {
                fprintf(csv, "%ld,%d,%d,%.6f\n", primes[i], coord_x[i], coord_y[i], slopes[i]);
            }
            fclose(csv);
            printf("  Done in %.2fs\n", toc());
        }
    }

    /* ---- Step 6: Rasterize into pixel buffer ---- */
    printf("Rasterizing %s primes to %dx%d canvas...\n",
           fmt_comma(n_primes - 1, buf1, sizeof(buf1)), canvas_px, canvas_px);
    tic();

    int64_t canvas_bytes = (int64_t)canvas_px * canvas_px * 3;
    uint8_t *canvas = (uint8_t *)calloc(canvas_bytes, 1);
    if (!canvas) {
        fprintf(stderr, "Failed to allocate canvas (%ld bytes = %.1f GB)\n",
                canvas_bytes, canvas_bytes / 1e9);
        exit(1);
    }

    double range_x = (double)(max_x - min_x);
    double range_y = (double)(max_y - min_y);
    double range = fmax(range_x, range_y);
    double margin_frac = 0.02;
    double scale = (canvas_px * (1.0 - 2.0 * margin_frac)) / range;
    double offset_x = canvas_px * margin_frac - min_x * scale + (range - range_x) * scale * 0.5;
    double offset_y = canvas_px * margin_frac - min_y * scale + (range - range_y) * scale * 0.5;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n_primes; i++) {
        int px = (int)(coord_x[i] * scale + offset_x);
        int py = (int)(coord_y[i] * scale + offset_y);
        py = canvas_px - 1 - py;  /* flip Y: math coords → image coords */
        if (px < 0 || px >= canvas_px || py < 0 || py >= canvas_px) continue;
        RGB c = slope_to_color((double)slopes[i]);
        int64_t off = ((int64_t)py * canvas_px + px) * 3;
        canvas[off + 0] = c.r;
        canvas[off + 1] = c.g;
        canvas[off + 2] = c.b;
    }
    printf("  Rasterized in %.2fs\n", toc());

    /* ---- Step 7: Write output ---- */
    (void)!system("mkdir -p /home/john/mynotes/HodgedeRham/spiral_outputs");

    char suffix[64];
    if (cfg.max_primes >= 1000000)
        snprintf(suffix, sizeof(suffix), "%ldM", cfg.max_primes / 1000000);
    else
        snprintf(suffix, sizeof(suffix), "%ldk", cfg.max_primes / 1000);

    /* Write PNG (default) */
    char png_path[512];
    snprintf(png_path, sizeof(png_path),
             "/home/john/mynotes/HodgedeRham/spiral_outputs/e8_slope_%s.png", suffix);
    printf("Writing PNG (%dx%d)...\n", canvas_px, canvas_px);
    tic();
    write_png(png_path, canvas, canvas_px, canvas_px);
    /* Get file size */
    FILE *check = fopen(png_path, "rb");
    long png_size = 0;
    if (check) { fseek(check, 0, SEEK_END); png_size = ftell(check); fclose(check); }
    printf("  Written %s (%.1f MB) in %.2fs\n", png_path, png_size / 1e6, toc());

    /* Optionally also write PPM */
    if (cfg.emit_ppm) {
        char ppm_path[512];
        snprintf(ppm_path, sizeof(ppm_path),
                 "/home/john/mynotes/HodgedeRham/spiral_outputs/e8_slope_%s.ppm", suffix);
        printf("Writing PPM...\n");
        tic();
        FILE *fp = fopen(ppm_path, "wb");
        if (!fp) { fprintf(stderr, "Cannot open %s: %s\n", ppm_path, strerror(errno)); exit(1); }
        fprintf(fp, "P6\n%d %d\n255\n", canvas_px, canvas_px);
        int64_t row_bytes = (int64_t)canvas_px * 3;
        for (int row = 0; row < canvas_px; row++) {
            fwrite(canvas + row * row_bytes, 1, row_bytes, fp);
        }
        fclose(fp);
        printf("  Written %s (%.2f GB) in %.2fs\n", ppm_path, canvas_bytes / 1e9, toc());
    }

    /* ---- Summary ---- */
    printf("\n==================================================\n");
    printf("Output: %s\n", png_path);
    printf("==================================================\n");

    free(canvas);
    free(slopes);
    free(coord_x);
    free(coord_y);
    free(primes);

    return 0;
}
