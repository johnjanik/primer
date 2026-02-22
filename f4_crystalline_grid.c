/*
 * f4_crystalline_grid.c — F4 Crystalline Grid Visualization (PPM output)
 *
 * Reproduces the F4 Crystalline Grid image: E8 root assignment of prime gaps,
 * F4 filtering, Jordan trace coloring (plasma colormap), and crystalline
 * vertex extraction on an Ulam spiral.
 *
 * Usage:
 *   ./f4_crystalline_grid --max-primes 2000000 --n-vertices 38 [--output grid.ppm] [--size 6000]
 *
 * Pipeline (matches Python e8_f4_prime_analysis.py exactly):
 *   1. Load primes from primes1.txt–primes50.txt
 *   2. Compute gaps and normalized gaps
 *   3. Generate 240 E8 roots, 48 F4 roots
 *   4. Build E8→F4 mapping (cosine similarity)
 *   5. Assign E8 roots to gaps via phase
 *   6. F4 filter (threshold 0.7) + Jordan trace
 *   7. F4-EFT spectrum (48 complex components)
 *   8. Crystalline vertex scoring + selection
 *   9. Ulam spiral coordinates
 *  10. Render PPM with plasma colormap
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o f4_crystalline_grid f4_crystalline_grid.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>

/* ========================================================================== */
/* Plasma colormap LUT (256 entries, matches matplotlib 'plasma')             */
/* ========================================================================== */
static const unsigned char plasma_lut[256][3] = {
  {12, 7, 134},
  {16, 7, 135},
  {19, 6, 137},
  {21, 6, 138},
  {24, 6, 139},
  {27, 6, 140},
  {29, 6, 141},
  {31, 5, 142},
  {33, 5, 143},
  {35, 5, 144},
  {37, 5, 145},
  {39, 5, 146},
  {41, 5, 147},
  {43, 5, 148},
  {45, 4, 148},
  {47, 4, 149},
  {49, 4, 150},
  {51, 4, 151},
  {52, 4, 152},
  {54, 4, 152},
  {56, 4, 153},
  {58, 4, 154},
  {59, 3, 154},
  {61, 3, 155},
  {63, 3, 156},
  {64, 3, 156},
  {66, 3, 157},
  {68, 3, 158},
  {69, 3, 158},
  {71, 2, 159},
  {73, 2, 159},
  {74, 2, 160},
  {76, 2, 161},
  {78, 2, 161},
  {79, 2, 162},
  {81, 1, 162},
  {82, 1, 163},
  {84, 1, 163},
  {86, 1, 163},
  {87, 1, 164},
  {89, 1, 164},
  {90, 0, 165},
  {92, 0, 165},
  {94, 0, 165},
  {95, 0, 166},
  {97, 0, 166},
  {98, 0, 166},
  {100, 0, 167},
  {101, 0, 167},
  {103, 0, 167},
  {104, 0, 167},
  {106, 0, 167},
  {108, 0, 168},
  {109, 0, 168},
  {111, 0, 168},
  {112, 0, 168},
  {114, 0, 168},
  {115, 0, 168},
  {117, 0, 168},
  {118, 1, 168},
  {120, 1, 168},
  {121, 1, 168},
  {123, 2, 168},
  {124, 2, 167},
  {126, 3, 167},
  {127, 3, 167},
  {129, 4, 167},
  {130, 4, 167},
  {132, 5, 166},
  {133, 6, 166},
  {134, 7, 166},
  {136, 7, 165},
  {137, 8, 165},
  {139, 9, 164},
  {140, 10, 164},
  {142, 12, 164},
  {143, 13, 163},
  {144, 14, 163},
  {146, 15, 162},
  {147, 16, 161},
  {149, 17, 161},
  {150, 18, 160},
  {151, 19, 160},
  {153, 20, 159},
  {154, 21, 158},
  {155, 23, 158},
  {157, 24, 157},
  {158, 25, 156},
  {159, 26, 155},
  {160, 27, 155},
  {162, 28, 154},
  {163, 29, 153},
  {164, 30, 152},
  {165, 31, 151},
  {167, 33, 151},
  {168, 34, 150},
  {169, 35, 149},
  {170, 36, 148},
  {172, 37, 147},
  {173, 38, 146},
  {174, 39, 145},
  {175, 40, 144},
  {176, 42, 143},
  {177, 43, 143},
  {178, 44, 142},
  {180, 45, 141},
  {181, 46, 140},
  {182, 47, 139},
  {183, 48, 138},
  {184, 50, 137},
  {185, 51, 136},
  {186, 52, 135},
  {187, 53, 134},
  {188, 54, 133},
  {189, 55, 132},
  {190, 56, 131},
  {191, 57, 130},
  {192, 59, 129},
  {193, 60, 128},
  {194, 61, 128},
  {195, 62, 127},
  {196, 63, 126},
  {197, 64, 125},
  {198, 65, 124},
  {199, 66, 123},
  {200, 68, 122},
  {201, 69, 121},
  {202, 70, 120},
  {203, 71, 119},
  {204, 72, 118},
  {205, 73, 117},
  {206, 74, 117},
  {207, 75, 116},
  {208, 77, 115},
  {209, 78, 114},
  {209, 79, 113},
  {210, 80, 112},
  {211, 81, 111},
  {212, 82, 110},
  {213, 83, 109},
  {214, 85, 109},
  {215, 86, 108},
  {215, 87, 107},
  {216, 88, 106},
  {217, 89, 105},
  {218, 90, 104},
  {219, 91, 103},
  {220, 93, 102},
  {220, 94, 102},
  {221, 95, 101},
  {222, 96, 100},
  {223, 97, 99},
  {223, 98, 98},
  {224, 100, 97},
  {225, 101, 96},
  {226, 102, 96},
  {227, 103, 95},
  {227, 104, 94},
  {228, 106, 93},
  {229, 107, 92},
  {229, 108, 91},
  {230, 109, 90},
  {231, 110, 90},
  {232, 112, 89},
  {232, 113, 88},
  {233, 114, 87},
  {234, 115, 86},
  {234, 116, 85},
  {235, 118, 84},
  {236, 119, 84},
  {236, 120, 83},
  {237, 121, 82},
  {237, 123, 81},
  {238, 124, 80},
  {239, 125, 79},
  {239, 126, 78},
  {240, 128, 77},
  {240, 129, 77},
  {241, 130, 76},
  {242, 132, 75},
  {242, 133, 74},
  {243, 134, 73},
  {243, 135, 72},
  {244, 137, 71},
  {244, 138, 71},
  {245, 139, 70},
  {245, 141, 69},
  {246, 142, 68},
  {246, 143, 67},
  {246, 145, 66},
  {247, 146, 65},
  {247, 147, 65},
  {248, 149, 64},
  {248, 150, 63},
  {248, 152, 62},
  {249, 153, 61},
  {249, 154, 60},
  {250, 156, 59},
  {250, 157, 58},
  {250, 159, 58},
  {250, 160, 57},
  {251, 162, 56},
  {251, 163, 55},
  {251, 164, 54},
  {252, 166, 53},
  {252, 167, 53},
  {252, 169, 52},
  {252, 170, 51},
  {252, 172, 50},
  {252, 173, 49},
  {253, 175, 49},
  {253, 176, 48},
  {253, 178, 47},
  {253, 179, 46},
  {253, 181, 45},
  {253, 182, 45},
  {253, 184, 44},
  {253, 185, 43},
  {253, 187, 43},
  {253, 188, 42},
  {253, 190, 41},
  {253, 192, 41},
  {253, 193, 40},
  {253, 195, 40},
  {253, 196, 39},
  {253, 198, 38},
  {252, 199, 38},
  {252, 201, 38},
  {252, 203, 37},
  {252, 204, 37},
  {252, 206, 37},
  {251, 208, 36},
  {251, 209, 36},
  {251, 211, 36},
  {250, 213, 36},
  {250, 214, 36},
  {250, 216, 36},
  {249, 217, 36},
  {249, 219, 36},
  {248, 221, 36},
  {248, 223, 36},
  {247, 224, 36},
  {247, 226, 37},
  {246, 228, 37},
  {246, 229, 37},
  {245, 231, 38},
  {245, 233, 38},
  {244, 234, 38},
  {243, 236, 38},
  {243, 238, 38},
  {242, 240, 38},
  {242, 241, 38},
  {241, 243, 38},
  {240, 245, 37},
  {240, 246, 35},
  {239, 248, 33}
};

/* ========================================================================== */
/* Constants                                                                   */
/* ========================================================================== */
#define MAX_E8_ROOTS 240
#define MAX_F4_ROOTS  48
#define MAX_PRIME_FILES 50
#define SQRT2 1.4142135623730951
#define F4_THRESHOLD 0.7

/* ========================================================================== */
/* E8 root system (240 roots in R^8)                                          */
/* ========================================================================== */
static double e8_roots[MAX_E8_ROOTS][8];
static int n_e8_roots = 0;

static void generate_e8_roots(void)
{
    n_e8_roots = 0;
    /* Type I: 112 roots — ±e_i ± e_j, i<j */
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            int signs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
            for (int s = 0; s < 4; s++) {
                memset(e8_roots[n_e8_roots], 0, 8 * sizeof(double));
                e8_roots[n_e8_roots][i] = signs[s][0];
                e8_roots[n_e8_roots][j] = signs[s][1];
                n_e8_roots++;
            }
        }
    }
    /* Type II: 128 roots — (±½)^8 with even number of minus signs */
    for (int mask = 0; mask < 256; mask++) {
        int neg_count = 0;
        double root[8];
        for (int i = 0; i < 8; i++) {
            if ((mask >> i) & 1) {
                root[i] = 0.5;
                /* bit=1 → sign=+1 */
            } else {
                root[i] = -0.5;
                neg_count++;
            }
        }
        if (neg_count % 2 == 0) {
            memcpy(e8_roots[n_e8_roots], root, 8 * sizeof(double));
            n_e8_roots++;
        }
    }
}

/* ========================================================================== */
/* F4 root system (48 roots in R^4)                                           */
/* ========================================================================== */
static double f4_roots[MAX_F4_ROOTS][4];
static int f4_is_long[MAX_F4_ROOTS];
static double f4_characters[MAX_F4_ROOTS];
static double f4_norms[MAX_F4_ROOTS];
static int n_f4_roots = 0;

static void generate_f4_roots(void)
{
    n_f4_roots = 0;
    /* Long roots (24): ±e_i ± e_j, i<j */
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            int signs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
            for (int s = 0; s < 4; s++) {
                memset(f4_roots[n_f4_roots], 0, 4 * sizeof(double));
                f4_roots[n_f4_roots][i] = signs[s][0];
                f4_roots[n_f4_roots][j] = signs[s][1];
                f4_is_long[n_f4_roots] = 1;
                n_f4_roots++;
            }
        }
    }
    /* Short roots Type A: ±e_i (8 roots) */
    for (int i = 0; i < 4; i++) {
        for (int s = -1; s <= 1; s += 2) {
            memset(f4_roots[n_f4_roots], 0, 4 * sizeof(double));
            f4_roots[n_f4_roots][i] = s;
            f4_is_long[n_f4_roots] = 0;
            n_f4_roots++;
        }
    }
    /* Short roots Type B: (±½)^4 with even # of minus (8 roots) */
    for (int mask = 0; mask < 16; mask++) {
        int neg = 0;
        double root[4];
        for (int i = 0; i < 4; i++) {
            if ((mask >> i) & 1) root[i] = 0.5;
            else { root[i] = -0.5; neg++; }
        }
        if (neg % 2 == 0) {
            memcpy(f4_roots[n_f4_roots], root, 4 * sizeof(double));
            f4_is_long[n_f4_roots] = 0;
            n_f4_roots++;
        }
    }
    /* Short roots Type C: (±½)^4 with odd # of minus (8 roots) */
    for (int mask = 0; mask < 16; mask++) {
        int neg = 0;
        double root[4];
        for (int i = 0; i < 4; i++) {
            if ((mask >> i) & 1) root[i] = 0.5;
            else { root[i] = -0.5; neg++; }
        }
        if (neg % 2 == 1) {
            memcpy(f4_roots[n_f4_roots], root, 4 * sizeof(double));
            f4_is_long[n_f4_roots] = 0;
            n_f4_roots++;
        }
    }

    /* Compute norms and characters */
    for (int i = 0; i < n_f4_roots; i++) {
        double sum_sq = 0, weyl_h = 0;
        for (int d = 0; d < 4; d++) {
            sum_sq += f4_roots[i][d] * f4_roots[i][d];
            weyl_h += fabs(f4_roots[i][d]);
        }
        f4_norms[i] = sqrt(sum_sq);
        f4_characters[i] = (f4_is_long[i] ? 2.0 : 1.0) * (1.0 + 0.1 * weyl_h);
    }
}

/* ========================================================================== */
/* E8 → F4 mapping                                                            */
/* ========================================================================== */
static int e8_to_f4_map[MAX_E8_ROOTS];    /* F4 index for each E8 root, or -1 */
static double e8_to_f4_sim[MAX_E8_ROOTS]; /* cosine similarity */

static void build_e8_f4_mapping(void)
{
    /* Precompute normalized F4 roots */
    double f4_norm_arr[MAX_F4_ROOTS][4];
    for (int i = 0; i < n_f4_roots; i++) {
        double n = f4_norms[i];
        if (n < 1e-10) n = 1e-10;
        for (int d = 0; d < 4; d++)
            f4_norm_arr[i][d] = f4_roots[i][d] / n;
    }

    for (int ei = 0; ei < n_e8_roots; ei++) {
        /* Project E8 root to first 4 coords */
        double proj[4];
        memcpy(proj, e8_roots[ei], 4 * sizeof(double));
        double pn = 0;
        for (int d = 0; d < 4; d++) pn += proj[d] * proj[d];
        pn = sqrt(pn);

        if (pn < 0.01) {
            /* Degenerate: use last 4 coords */
            memcpy(proj, &e8_roots[ei][4], 4 * sizeof(double));
            pn = 0;
            for (int d = 0; d < 4; d++) pn += proj[d] * proj[d];
            pn = sqrt(pn);
        }
        if (pn < 0.01) {
            e8_to_f4_map[ei] = -1;
            e8_to_f4_sim[ei] = 0;
            continue;
        }

        /* Normalize projection */
        double proj_n[4];
        for (int d = 0; d < 4; d++) proj_n[d] = proj[d] / pn;

        /* Find nearest F4 root by |cosine similarity| */
        int best_f4 = 0;
        double best_sim = 0;
        for (int fi = 0; fi < n_f4_roots; fi++) {
            double dot = 0;
            for (int d = 0; d < 4; d++)
                dot += proj_n[d] * f4_norm_arr[fi][d];
            double absim = fabs(dot);
            if (absim > best_sim) {
                best_sim = absim;
                best_f4 = fi;
            }
        }
        e8_to_f4_map[ei] = best_f4;
        e8_to_f4_sim[ei] = best_sim;
    }
}

/* ========================================================================== */
/* E8 root assignment: normalized gap → E8 root index                         */
/* ========================================================================== */
static inline int assign_e8_root(double normalized_gap)
{
    double target_norm = sqrt(fmax(normalized_gap, 0.01));
    double phase = fmod(target_norm / SQRT2, 1.0);
    if (phase < 0) phase += 1.0;
    int idx = (int)(phase * 240) % 240;
    return idx;
}

/* ========================================================================== */
/* Jordan trace: sum of F4 root's 4 coordinates                               */
/* (proj_matrix @ root → 3-vector, then sum)                                  */
/* ========================================================================== */
static inline double jordan_trace(int f4_idx)
{
    /* proj = [root[0], root[1], root[2]+root[3]], sum = root[0]+root[1]+root[2]+root[3] */
    return f4_roots[f4_idx][0] + f4_roots[f4_idx][1]
         + f4_roots[f4_idx][2] + f4_roots[f4_idx][3];
}

/* ========================================================================== */
/* Ulam spiral coordinates                                                     */
/* ========================================================================== */
static void ulam_coord(long long p, int *x, int *y)
{
    if (p <= 0) { *x = 0; *y = 0; return; }
    int k = (int)ceil((sqrt((double)p) - 1.0) / 2.0);
    int t = 2 * k + 1;
    long long m = (long long)t * t;
    t -= 1;

    if (p >= m - t) {
        *x = k - (int)(m - p);
        *y = -k;
    } else if (p >= m - 2*t) {
        *x = -k;
        *y = -k + (int)(m - t - p);
    } else if (p >= m - 3*t) {
        *x = -k + (int)(m - 2*t - p);
        *y = k;
    } else {
        *x = k;
        *y = k - (int)(m - 3*t - p);
    }
}

/* ========================================================================== */
/* Load primes from primes1.txt–primes50.txt                                  */
/* ========================================================================== */
static long long *load_primes(const char *base_dir, int max_primes, int *out_count)
{
    long long *primes = malloc((size_t)max_primes * sizeof(long long));
    if (!primes) { fprintf(stderr, "malloc failed\n"); exit(1); }
    int count = 0;

    for (int fi = 1; fi <= MAX_PRIME_FILES && count < max_primes; fi++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/primes%d.txt", base_dir, fi);
        FILE *fp = fopen(path, "r");
        if (!fp) break;

        /* Skip header line */
        char line[4096];
        if (fgets(line, sizeof(line), fp)) {
            /* Check if it's actually a header */
            int is_header = 0;
            for (int i = 0; line[i]; i++) {
                if (isalpha((unsigned char)line[i])) { is_header = 1; break; }
            }
            if (!is_header) {
                /* Not a header, parse it */
                char *tok = strtok(line, " \t\n\r");
                while (tok && count < max_primes) {
                    long long v = atoll(tok);
                    if (v > 1) primes[count++] = v;
                    tok = strtok(NULL, " \t\n\r");
                }
            }
        }

        while (fgets(line, sizeof(line), fp) && count < max_primes) {
            char *tok = strtok(line, " \t\n\r");
            while (tok && count < max_primes) {
                long long v = atoll(tok);
                if (v > 1) primes[count++] = v;
                tok = strtok(NULL, " \t\n\r");
            }
        }
        fclose(fp);
    }
    *out_count = count;
    return primes;
}

/* ========================================================================== */
/* Min-heap for top-K selection                                                */
/* ========================================================================== */
typedef struct { double score; int index; } ScoreEntry;

static void heap_push(ScoreEntry *heap, int *size, int capacity, ScoreEntry entry)
{
    if (*size < capacity) {
        heap[*size] = entry;
        (*size)++;
        /* Sift up */
        int i = *size - 1;
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (heap[parent].score > heap[i].score) {
                ScoreEntry tmp = heap[parent];
                heap[parent] = heap[i];
                heap[i] = tmp;
                i = parent;
            } else break;
        }
    } else if (entry.score > heap[0].score) {
        heap[0] = entry;
        /* Sift down */
        int i = 0;
        for (;;) {
            int left = 2*i + 1, right = 2*i + 2, smallest = i;
            if (left < capacity && heap[left].score < heap[smallest].score)
                smallest = left;
            if (right < capacity && heap[right].score < heap[smallest].score)
                smallest = right;
            if (smallest == i) break;
            ScoreEntry tmp = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = tmp;
            i = smallest;
        }
    }
}

/* ========================================================================== */
/* Draw a filled circle with edge                                              */
/* ========================================================================== */
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
                    /* Edge pixel */
                    buf[off] = er; buf[off+1] = eg; buf[off+2] = eb;
                } else {
                    /* Fill pixel */
                    buf[off] = fr; buf[off+1] = fg; buf[off+2] = fb;
                }
            }
        }
    }
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */
int main(int argc, char **argv)
{
    /* Defaults */
    int max_primes = 2000000;
    int n_vertices = 38;
    int img_size = 6000;
    const char *output_file = "f4_crystalline_grid.ppm";
    const char *prime_dir = "/home/john/mynotes/HodgedeRham";

    /* Parse command line */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--max-primes") == 0 && i+1 < argc)
            max_primes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-vertices") == 0 && i+1 < argc)
            n_vertices = atoi(argv[++i]);
        else if (strcmp(argv[i], "--size") == 0 && i+1 < argc)
            img_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "--prime-dir") == 0 && i+1 < argc)
            prime_dir = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --max-primes N   Number of primes to load (default 2000000)\n");
            printf("  --n-vertices N   Number of crystalline vertices (default 38)\n");
            printf("  --size N         Image size in pixels (default 6000)\n");
            printf("  --output FILE    Output PPM file (default f4_crystalline_grid.ppm)\n");
            printf("  --prime-dir DIR  Directory containing primes*.txt files\n");
            return 0;
        }
    }

    printf("F4 Crystalline Grid Generator\n");
    printf("  max_primes: %d\n", max_primes);
    printf("  n_vertices: %d\n", n_vertices);
    printf("  image size: %d x %d\n", img_size, img_size);
    printf("  output: %s\n", output_file);

    /* Step 1: Generate root systems */
    printf("\nGenerating E8 roots...\n");
    generate_e8_roots();
    printf("  %d E8 roots\n", n_e8_roots);

    printf("Generating F4 roots...\n");
    generate_f4_roots();
    printf("  %d F4 roots\n", n_f4_roots);

    printf("Building E8->F4 mapping...\n");
    build_e8_f4_mapping();
    {
        int f4_mapped = 0;
        for (int i = 0; i < n_e8_roots; i++)
            if (e8_to_f4_map[i] >= 0 && e8_to_f4_sim[i] >= F4_THRESHOLD)
                f4_mapped++;
        printf("  %d E8 roots pass F4 threshold (%.1f)\n", f4_mapped, F4_THRESHOLD);
    }

    /* Step 2: Load primes */
    printf("\nLoading primes from %s...\n", prime_dir);
    int n_primes = 0;
    long long *primes = load_primes(prime_dir, max_primes, &n_primes);
    printf("  Loaded %d primes\n", n_primes);
    if (n_primes < 2) {
        fprintf(stderr, "Not enough primes loaded\n");
        free(primes);
        return 1;
    }
    printf("  Range: %lld to %lld\n", primes[0], primes[n_primes-1]);

    int n_gaps = n_primes - 1;

    /* Step 3: Compute gaps, normalized gaps, E8 assignments, F4 filter */
    printf("\nComputing gaps and E8 assignments...\n");
    double *gaps = malloc((size_t)n_gaps * sizeof(double));
    double *norm_gaps = malloc((size_t)n_gaps * sizeof(double));
    int *e8_idx = malloc((size_t)n_gaps * sizeof(int));
    int *f4_idx = malloc((size_t)n_gaps * sizeof(int));    /* -1 if not F4 */
    double *jtrace = malloc((size_t)n_gaps * sizeof(double)); /* NAN if not F4 */
    int *is_f4 = malloc((size_t)n_gaps * sizeof(int));

    if (!gaps || !norm_gaps || !e8_idx || !f4_idx || !jtrace || !is_f4) {
        fprintf(stderr, "malloc failed\n"); return 1;
    }

    int f4_count = 0;
    #pragma omp parallel for reduction(+:f4_count) schedule(static)
    for (int i = 0; i < n_gaps; i++) {
        gaps[i] = (double)(primes[i+1] - primes[i]);
        double lp = log((double)primes[i]);
        if (lp < 1.0) lp = 1.0;
        norm_gaps[i] = gaps[i] / lp;

        e8_idx[i] = assign_e8_root(norm_gaps[i]);

        int ei = e8_idx[i];
        if (e8_to_f4_map[ei] >= 0 && e8_to_f4_sim[ei] >= F4_THRESHOLD) {
            is_f4[i] = 1;
            f4_idx[i] = e8_to_f4_map[ei];
            jtrace[i] = jordan_trace(f4_idx[i]);
            f4_count++;
        } else {
            is_f4[i] = 0;
            f4_idx[i] = -1;
            jtrace[i] = 0.0 / 0.0; /* NaN */
        }
    }
    double f4_fraction = (double)f4_count / n_gaps;
    printf("  F4 fraction: %.4f (%d / %d gaps)\n", f4_fraction, f4_count, n_gaps);

    /* Step 4: F4-EFT spectrum (48 complex components) */
    printf("\nComputing F4-EFT spectrum...\n");
    double spec_re[MAX_F4_ROOTS] = {0};
    double spec_im[MAX_F4_ROOTS] = {0};

    /* Serial accumulation (complex addition isn't easy to parallelize with atomics) */
    for (int n = 0; n < n_gaps; n++) {
        if (!is_f4[n]) continue;
        int fi = f4_idx[n];
        double fluct = norm_gaps[n] - 1.0;
        double chi = f4_characters[fi];
        double phase = 2.0 * M_PI * f4_norms[fi] / SQRT2 * (double)n / (double)n_gaps;
        spec_re[fi] += fluct * chi * cos(phase);
        spec_im[fi] += fluct * chi * sin(phase);
    }

    double power[MAX_F4_ROOTS];
    for (int i = 0; i < n_f4_roots; i++)
        power[i] = spec_re[i]*spec_re[i] + spec_im[i]*spec_im[i];

    /* Step 5: Crystalline vertex scoring */
    printf("Scoring crystalline vertices...\n");
    double *scores = calloc((size_t)n_gaps, sizeof(double));
    if (!scores) { fprintf(stderr, "malloc failed\n"); return 1; }

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < n_gaps; n++) {
        if (!is_f4[n]) continue;
        int fi = f4_idx[n];
        scores[n] = power[fi];
        /* Boost idempotent-type roots (|J| ≈ 1) */
        double jt = jtrace[n];
        if (fabs(fabs(jt) - 1.0) < 0.2)
            scores[n] *= 2.0;
    }

    /* Select top n_vertices using min-heap */
    ScoreEntry *heap = malloc((size_t)n_vertices * sizeof(ScoreEntry));
    int heap_size = 0;
    for (int n = 0; n < n_gaps; n++) {
        if (scores[n] > 0) {
            ScoreEntry e = { scores[n], n };
            heap_push(heap, &heap_size, n_vertices, e);
        }
    }

    /* Extract vertex indices */
    int *vertex_gap_idx = malloc((size_t)n_vertices * sizeof(int));
    int actual_vertices = heap_size;
    for (int i = 0; i < heap_size; i++)
        vertex_gap_idx[i] = heap[i].index;
    free(heap);

    printf("  Selected %d crystalline vertices\n", actual_vertices);

    /* Step 6: Compute Ulam spiral coordinates */
    printf("\nComputing Ulam spiral coordinates...\n");
    int *ux = malloc((size_t)n_primes * sizeof(int));
    int *uy = malloc((size_t)n_primes * sizeof(int));
    if (!ux || !uy) { fprintf(stderr, "malloc failed\n"); return 1; }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_primes; i++)
        ulam_coord(primes[i], &ux[i], &uy[i]);

    /* Find coordinate range */
    int min_x = ux[0], max_x = ux[0], min_y = uy[0], max_y = uy[0];
    for (int i = 1; i < n_primes; i++) {
        if (ux[i] < min_x) min_x = ux[i];
        if (ux[i] > max_x) max_x = ux[i];
        if (uy[i] < min_y) min_y = uy[i];
        if (uy[i] > max_y) max_y = uy[i];
    }
    printf("  Coordinate range: x=[%d,%d] y=[%d,%d]\n", min_x, max_x, min_y, max_y);

    /* Step 7: Render PPM */
    printf("\nRendering %dx%d image...\n", img_size, img_size);
    size_t buf_size = (size_t)img_size * img_size * 3;
    unsigned char *img = calloc(buf_size, 1); /* black background */
    if (!img) { fprintf(stderr, "malloc failed for image buffer\n"); return 1; }

    /* Compute scaling: map coordinate range to image, with some padding */
    int coord_range_x = max_x - min_x + 1;
    int coord_range_y = max_y - min_y + 1;
    int coord_range = coord_range_x > coord_range_y ? coord_range_x : coord_range_y;
    double scale = (double)(img_size - 20) / (double)coord_range;
    int cx_off = img_size / 2;
    int cy_off = img_size / 2;
    int mid_x = (min_x + max_x) / 2;
    int mid_y = (min_y + max_y) / 2;

    /* Plot all F4-mapped gaps colored by Jordan trace */
    printf("  Plotting F4 primes...\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_gaps; i++) {
        if (!is_f4[i]) continue;
        /* Gap i corresponds to prime[i+1] (the gap between prime[i] and prime[i+1]) */
        /* Use prime[i+1]'s coordinates (matching Python: coords[1:]) */
        int pidx = i + 1;
        int px = (int)((ux[pidx] - mid_x) * scale) + cx_off;
        int py = (int)((uy[pidx] - mid_y) * scale) + cy_off;
        if (px < 0 || px >= img_size || py < 0 || py >= img_size) continue;

        /* Map Jordan trace [-2, +2] → [0, 255] for plasma colormap */
        double jt = jtrace[i];
        if (jt != jt) continue; /* NaN check */
        double t = (jt + 2.0) / 4.0; /* [0, 1] */
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        int ci = (int)(t * 255.0);
        if (ci > 255) ci = 255;

        int off = (py * img_size + px) * 3;
        /* Simple alpha blend: just overwrite (s=1 in matplotlib, alpha=0.7) */
        unsigned char r = plasma_lut[ci][0];
        unsigned char g = plasma_lut[ci][1];
        unsigned char b = plasma_lut[ci][2];
        /* Alpha blend with existing pixel (0.7 opacity) */
        unsigned char old_r = img[off], old_g = img[off+1], old_b = img[off+2];
        img[off]   = (unsigned char)(0.7 * r + 0.3 * old_r);
        img[off+1] = (unsigned char)(0.7 * g + 0.3 * old_g);
        img[off+2] = (unsigned char)(0.7 * b + 0.3 * old_b);
    }

    /* Overlay crystalline vertices as white circles with yellow edges */
    printf("  Drawing %d crystalline vertices...\n", actual_vertices);
    int vertex_radius = 3;
    if (img_size > 4000) vertex_radius = 4;
    if (img_size > 8000) vertex_radius = 6;

    for (int v = 0; v < actual_vertices; v++) {
        int gi = vertex_gap_idx[v];
        int pidx = gi + 1;
        if (pidx >= n_primes) continue;
        int px = (int)((ux[pidx] - mid_x) * scale) + cx_off;
        int py = (int)((uy[pidx] - mid_y) * scale) + cy_off;
        draw_circle(img, img_size, img_size, px, py, vertex_radius,
                    255, 255, 255,   /* white fill */
                    255, 255, 0);    /* yellow edge */
    }

    /* Write PPM file */
    printf("  Writing %s...\n", output_file);
    FILE *fp = fopen(output_file, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", output_file); return 1; }
    fprintf(fp, "P6\n%d %d\n255\n", img_size, img_size);
    fwrite(img, 1, buf_size, fp);
    fclose(fp);
    printf("  Done! (%zu bytes)\n", buf_size + 20);

    /* Print summary */
    printf("\n========================================\n");
    printf("F4 Crystalline Grid Summary\n");
    printf("========================================\n");
    printf("  Primes:              %d\n", n_primes);
    printf("  Gaps:                %d\n", n_gaps);
    printf("  F4 primes:           %d (%.1f%% of gaps)\n", f4_count, 100.0*f4_fraction);
    printf("  Crystalline vertices: %d\n", actual_vertices);
    printf("  Image:               %s (%dx%d)\n", output_file, img_size, img_size);
    printf("========================================\n");
    printf("To convert: convert %s grid.png\n", output_file);

    /* Cleanup */
    free(primes); free(gaps); free(norm_gaps);
    free(e8_idx); free(f4_idx); free(jtrace); free(is_f4);
    free(scores); free(vertex_gap_idx);
    free(ux); free(uy); free(img);

    return 0;
}
