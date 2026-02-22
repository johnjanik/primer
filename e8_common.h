/*
 * e8_common.h — Shared E8/F4 routines, prime loading, Ulam spiral, colormap
 *
 * Used by: e8_slope_viz.c, e8_decoder.c, e8_f4_analysis.c
 */

#ifndef E8_COMMON_H
#define E8_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <errno.h>
#include <ctype.h>

/* ================================================================
 * Timing macros
 * ================================================================ */

static double g_timer_start;
#define tic() (g_timer_start = omp_get_wtime())
#define toc() (omp_get_wtime() - g_timer_start)

/* ================================================================
 * E8 Root System: 240 roots in R^8
 * ================================================================ */

#define E8_NUM_ROOTS 240
#define E8_DIM 8

typedef struct {
    double roots[E8_NUM_ROOTS][E8_DIM];
    double slopes[E8_NUM_ROOTS];   /* projection slope = sum(coords[4:8]) / sum(coords[0:4]) */
    double min_norm;                /* sqrt(2) */
} E8Lattice;

static void e8_generate_roots(E8Lattice *e8)
{
    int idx = 0;

    /* Type I: 112 roots — ±e_i ± e_j for i < j */
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    memset(e8->roots[idx], 0, sizeof(double) * E8_DIM);
                    e8->roots[idx][i] = s1;
                    e8->roots[idx][j] = s2;
                    idx++;
                }
            }
        }
    }

    /* Type II: 128 roots — (±1/2)^8 with even number of minus signs */
    for (int mask = 0; mask < 256; mask++) {
        int neg_count = 0;
        double root[8];
        for (int i = 0; i < 8; i++) {
            if ((mask >> i) & 1) {
                root[i] = 0.5;
                /* bit=1 → +1/2 */
            } else {
                root[i] = -0.5;
                neg_count++;
            }
        }
        if (neg_count % 2 == 0) {
            memcpy(e8->roots[idx], root, sizeof(double) * E8_DIM);
            idx++;
        }
    }

    if (idx != E8_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d E8 roots, expected %d\n", idx, E8_NUM_ROOTS);
        exit(1);
    }

    e8->min_norm = sqrt(2.0);
}

static void e8_compute_slopes(E8Lattice *e8)
{
    for (int r = 0; r < E8_NUM_ROOTS; r++) {
        double x = e8->roots[r][0] + e8->roots[r][1] + e8->roots[r][2] + e8->roots[r][3];
        double y = e8->roots[r][4] + e8->roots[r][5] + e8->roots[r][6] + e8->roots[r][7];
        if (fabs(x) > 0.01)
            e8->slopes[r] = y / x;
        else
            e8->slopes[r] = (y > 0.0 ? 1.0 : y < 0.0 ? -1.0 : 0.0) * 10.0;
    }
}

static void e8_init(E8Lattice *e8)
{
    e8_generate_roots(e8);
    e8_compute_slopes(e8);
}

/* Match Python: phase = (sqrt(max(g, 0.01)) / sqrt(2)) % 1.0; idx = (int)(phase * 240) % 240 */
static inline int e8_assign_root(const E8Lattice *e8, double normalized_gap)
{
    double target_norm = sqrt(fmax(normalized_gap, 0.01));
    double phase = fmod(target_norm / e8->min_norm, 1.0);
    if (phase < 0.0) phase += 1.0;
    int idx = (int)(phase * E8_NUM_ROOTS) % E8_NUM_ROOTS;
    if (idx < 0) idx += E8_NUM_ROOTS;
    return idx;
}

/* ================================================================
 * Ulam Spiral Coordinates
 * ================================================================ */

static inline void ulam_coord(int64_t p, int32_t *x, int32_t *y)
{
    if (p <= 0) { *x = 0; *y = 0; return; }
    int32_t k = (int32_t)ceil((sqrt((double)p) - 1.0) / 2.0);
    int32_t t = 2 * k + 1;
    int64_t m = (int64_t)t * t;
    t -= 1;  /* now t = 2k */

    if (p >= m - t)       { *x = k - (int32_t)(m - p); *y = -k; }
    else if (p >= m - 2*t){ *x = -k; *y = -k + (int32_t)(m - t - p); }
    else if (p >= m - 3*t){ *x = -k + (int32_t)(m - 2*(int64_t)t - p); *y = k; }
    else                  { *x = k; *y = k - (int32_t)(m - 3*(int64_t)t - p); }
}

/* ================================================================
 * Coolwarm Colormap (256 entries, generated from matplotlib)
 * ================================================================ */

typedef struct { uint8_t r, g, b; } RGB;

static const RGB coolwarm_lut[256] = {
    {58,76,192},{59,77,193},{60,79,195},{62,81,196},{63,83,198},{64,84,199},
    {65,86,201},{66,88,202},{67,90,204},{69,91,205},{70,93,207},{71,95,208},
    {72,96,209},{73,98,211},{75,100,212},{76,102,214},{77,103,215},{78,105,216},
    {80,107,218},{81,108,219},{82,110,220},{83,112,221},{85,113,222},{86,115,224},
    {87,117,225},{88,118,226},{90,120,227},{91,121,228},{92,123,229},{93,125,230},
    {95,126,231},{96,128,232},{97,130,234},{99,131,234},{100,133,235},{101,134,236},
    {103,136,237},{104,137,238},{105,139,239},{107,141,240},{108,142,241},{109,144,241},
    {111,145,242},{112,147,243},{113,148,244},{115,149,244},{116,151,245},{117,152,246},
    {119,154,246},{120,155,247},{122,157,248},{123,158,248},{124,160,249},{126,161,249},
    {127,162,250},{128,164,250},{130,165,251},{131,166,251},{133,168,251},{134,169,252},
    {135,170,252},{137,172,252},{138,173,253},{139,174,253},{141,175,253},{142,177,253},
    {144,178,254},{145,179,254},{146,180,254},{148,181,254},{149,183,254},{151,184,254},
    {152,185,254},{153,186,254},{155,187,254},{156,188,254},{157,189,254},{159,190,254},
    {160,191,254},{162,192,254},{163,193,254},{164,194,254},{166,195,253},{167,196,253},
    {168,197,253},{170,198,253},{171,199,252},{172,200,252},{174,201,252},{175,202,251},
    {176,203,251},{178,203,251},{179,204,250},{180,205,250},{182,206,249},{183,207,249},
    {184,207,248},{185,208,248},{187,209,247},{188,209,246},{189,210,246},{190,211,245},
    {192,211,245},{193,212,244},{194,212,243},{195,213,242},{197,213,242},{198,214,241},
    {199,214,240},{200,215,239},{201,215,238},{202,216,238},{204,216,237},{205,217,236},
    {206,217,235},{207,217,234},{208,218,233},{209,218,232},{210,218,231},{211,219,230},
    {213,219,229},{214,219,228},{215,219,226},{216,219,225},{217,220,224},{218,220,223},
    {219,220,222},{220,220,221},{221,220,219},{222,219,218},{223,219,217},{224,218,215},
    {225,218,214},{226,217,212},{227,217,211},{228,216,209},{229,216,208},{230,215,207},
    {231,214,205},{231,214,204},{232,213,202},{233,212,201},{234,211,199},{235,211,198},
    {236,210,196},{236,209,195},{237,208,193},{237,207,192},{238,207,190},{239,206,188},
    {239,205,187},{240,204,185},{241,203,184},{241,202,182},{242,201,181},{242,200,179},
    {242,199,178},{243,198,176},{243,197,175},{244,196,173},{244,195,171},{244,194,170},
    {245,193,168},{245,192,167},{245,191,165},{246,189,164},{246,188,162},{246,187,160},
    {246,186,159},{246,185,157},{246,183,156},{246,182,154},{247,181,152},{247,179,151},
    {247,178,149},{247,177,148},{247,176,146},{247,174,145},{247,173,143},{246,171,141},
    {246,170,140},{246,169,138},{246,167,137},{246,166,135},{246,164,134},{246,163,132},
    {245,161,130},{245,160,129},{245,158,127},{244,157,126},{244,155,124},{244,154,123},
    {243,152,121},{243,150,120},{243,149,118},{242,147,117},{242,145,115},{241,144,114},
    {241,142,112},{240,141,111},{240,139,109},{239,137,108},{238,135,106},{238,134,105},
    {237,132,103},{236,130,102},{236,128,100},{235,127,99},{234,125,97},{234,123,96},
    {233,121,94},{232,119,93},{231,117,92},{230,116,90},{230,114,89},{229,112,87},
    {228,110,86},{227,108,84},{226,106,83},{225,104,82},{224,102,80},{223,100,79},
    {222,98,78},{221,96,76},{220,94,75},{219,92,74},{218,90,72},{217,88,71},
    {216,86,70},{215,84,68},{214,82,67},{212,79,66},{211,77,64},{210,75,63},
    {209,73,62},{207,70,61},{206,68,60},{205,66,58},{204,63,57},{202,61,56},
    {201,59,55},{200,56,53},{198,53,52},{197,50,51},{196,48,50},{194,45,49},
    {193,42,48},{191,40,46},{190,35,45},{188,31,44},{187,26,43},{185,22,42},
    {184,17,41},{182,13,40},{181,8,39},{179,3,38},
};

/* Map a slope in [-3, +3] to a coolwarm RGB color */
static inline RGB slope_to_color(double slope)
{
    double clamped = fmax(-3.0, fmin(3.0, slope));
    double t = (clamped + 3.0) / 6.0;   /* [0, 1] */
    int idx = (int)(t * 255.0);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    return coolwarm_lut[idx];
}

/* Plasma colormap (256 entries, generated from matplotlib) */
static const RGB plasma_lut[256] = {
    {12,7,134},{16,7,135},{19,6,137},{21,6,138},{24,6,139},{27,6,140},
    {29,6,141},{31,5,142},{33,5,143},{35,5,144},{37,5,145},{39,5,146},
    {41,5,147},{43,5,148},{45,4,148},{47,4,149},{49,4,150},{51,4,151},
    {52,4,152},{54,4,152},{56,4,153},{58,4,154},{59,3,154},{61,3,155},
    {63,3,156},{64,3,156},{66,3,157},{68,3,158},{69,3,158},{71,2,159},
    {73,2,159},{74,2,160},{76,2,161},{78,2,161},{79,2,162},{81,1,162},
    {82,1,163},{84,1,163},{86,1,163},{87,1,164},{89,1,164},{90,0,165},
    {92,0,165},{94,0,165},{95,0,166},{97,0,166},{98,0,166},{100,0,167},
    {101,0,167},{103,0,167},{104,0,167},{106,0,167},{108,0,168},{109,0,168},
    {111,0,168},{112,0,168},{114,0,168},{115,0,168},{117,0,168},{118,1,168},
    {120,1,168},{121,1,168},{123,2,168},{124,2,167},{126,3,167},{127,3,167},
    {129,4,167},{130,4,167},{132,5,166},{133,6,166},{134,7,166},{136,7,165},
    {137,8,165},{139,9,164},{140,10,164},{142,12,164},{143,13,163},{144,14,163},
    {146,15,162},{147,16,161},{149,17,161},{150,18,160},{151,19,160},{153,20,159},
    {154,21,158},{155,23,158},{157,24,157},{158,25,156},{159,26,155},{160,27,155},
    {162,28,154},{163,29,153},{164,30,152},{165,31,151},{167,33,151},{168,34,150},
    {169,35,149},{170,36,148},{172,37,147},{173,38,146},{174,39,145},{175,40,144},
    {176,42,143},{177,43,143},{178,44,142},{180,45,141},{181,46,140},{182,47,139},
    {183,48,138},{184,50,137},{185,51,136},{186,52,135},{187,53,134},{188,54,133},
    {189,55,132},{190,56,131},{191,57,130},{192,59,129},{193,60,128},{194,61,128},
    {195,62,127},{196,63,126},{197,64,125},{198,65,124},{199,66,123},{200,68,122},
    {201,69,121},{202,70,120},{203,71,119},{204,72,118},{205,73,117},{206,74,117},
    {207,75,116},{208,77,115},{209,78,114},{209,79,113},{210,80,112},{211,81,111},
    {212,82,110},{213,83,109},{214,85,109},{215,86,108},{215,87,107},{216,88,106},
    {217,89,105},{218,90,104},{219,91,103},{220,93,102},{220,94,102},{221,95,101},
    {222,96,100},{223,97,99},{223,98,98},{224,100,97},{225,101,96},{226,102,96},
    {227,103,95},{227,104,94},{228,106,93},{229,107,92},{229,108,91},{230,109,90},
    {231,110,90},{232,112,89},{232,113,88},{233,114,87},{234,115,86},{234,116,85},
    {235,118,84},{236,119,84},{236,120,83},{237,121,82},{237,123,81},{238,124,80},
    {239,125,79},{239,126,78},{240,128,77},{240,129,77},{241,130,76},{242,132,75},
    {242,133,74},{243,134,73},{243,135,72},{244,137,71},{244,138,71},{245,139,70},
    {245,141,69},{246,142,68},{246,143,67},{246,145,66},{247,146,65},{247,147,65},
    {248,149,64},{248,150,63},{248,152,62},{249,153,61},{249,154,60},{250,156,59},
    {250,157,58},{250,159,58},{250,160,57},{251,162,56},{251,163,55},{251,164,54},
    {252,166,53},{252,167,53},{252,169,52},{252,170,51},{252,172,50},{252,173,49},
    {253,175,49},{253,176,48},{253,178,47},{253,179,46},{253,181,45},{253,182,45},
    {253,184,44},{253,185,43},{253,187,43},{253,188,42},{253,190,41},{253,192,41},
    {253,193,40},{253,195,40},{253,196,39},{253,198,38},{252,199,38},{252,201,38},
    {252,203,37},{252,204,37},{252,206,37},{251,208,36},{251,209,36},{251,211,36},
    {250,213,36},{250,214,36},{250,216,36},{249,217,36},{249,219,36},{248,221,36},
    {248,223,36},{247,224,36},{247,226,37},{246,228,37},{246,229,37},{245,231,38},
    {245,233,38},{244,234,38},{243,236,38},{243,238,38},{242,240,38},{242,241,38},
    {241,243,38},{240,245,37},{240,246,35},{239,248,33},
};

/* Map Jordan trace in [-2, +2] to plasma RGB */
static inline RGB jordan_to_color(double jtrace)
{
    double clamped = fmax(-2.0, fmin(2.0, jtrace));
    double t = (clamped + 2.0) / 4.0;   /* [0, 1] */
    int idx = (int)(t * 255.0);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    return plasma_lut[idx];
}

/* ================================================================
 * F4 Root System: 48 roots in R^4 (sublattice of E8)
 * ================================================================ */

#define F4_NUM_ROOTS 48

typedef struct {
    double roots[F4_NUM_ROOTS][4];       /* 48 F4 roots in R^4 */
    double norms[F4_NUM_ROOTS];          /* norm of each root */
    double characters[F4_NUM_ROOTS];     /* F4 character weights */
    double jordan_traces[F4_NUM_ROOTS];  /* precomputed Jordan traces */
    int    is_long[F4_NUM_ROOTS];        /* 1 = long root (norm sqrt(2)) */
    int    e8_to_f4[E8_NUM_ROOTS];       /* E8→F4 map (-1 = not mapped) */
    int    e8_is_f4[E8_NUM_ROOTS];       /* 1 = this E8 root maps to F4 with quality >= 0.7 */
} F4Lattice;

static void f4_generate_roots(F4Lattice *f4)
{
    int idx = 0;

    /* 24 long roots (norm sqrt(2)): ±e_i ± e_j for i < j */
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    memset(f4->roots[idx], 0, sizeof(double) * 4);
                    f4->roots[idx][i] = s1;
                    f4->roots[idx][j] = s2;
                    idx++;
                }
            }
        }
    }

    /* 8 short roots: ±e_i */
    for (int i = 0; i < 4; i++) {
        for (int s = -1; s <= 1; s += 2) {
            memset(f4->roots[idx], 0, sizeof(double) * 4);
            f4->roots[idx][i] = s;
            idx++;
        }
    }

    /* 8 short roots: (±1/2)^4 with even number of minus signs */
    for (int mask = 0; mask < 16; mask++) {
        int neg_count = 0;
        double root[4];
        for (int i = 0; i < 4; i++) {
            if ((mask >> i) & 1) root[i] = 0.5;
            else { root[i] = -0.5; neg_count++; }
        }
        if (neg_count % 2 == 0) {
            memcpy(f4->roots[idx], root, sizeof(double) * 4);
            idx++;
        }
    }

    /* 8 short roots: (±1/2)^4 with odd number of minus signs */
    for (int mask = 0; mask < 16; mask++) {
        int neg_count = 0;
        double root[4];
        for (int i = 0; i < 4; i++) {
            if ((mask >> i) & 1) root[i] = 0.5;
            else { root[i] = -0.5; neg_count++; }
        }
        if (neg_count % 2 == 1) {
            memcpy(f4->roots[idx], root, sizeof(double) * 4);
            idx++;
        }
    }

    if (idx != F4_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d F4 roots, expected %d\n", idx, F4_NUM_ROOTS);
        exit(1);
    }
}

static void f4_compute_properties(F4Lattice *f4)
{
    for (int i = 0; i < F4_NUM_ROOTS; i++) {
        /* Norm */
        double n2 = 0;
        for (int d = 0; d < 4; d++) n2 += f4->roots[i][d] * f4->roots[i][d];
        f4->norms[i] = sqrt(n2);

        /* Long vs short */
        f4->is_long[i] = (f4->norms[i] > 1.2) ? 1 : 0;

        /* Character: long=2.0, short=1.0, modulated by Weyl height */
        double weyl_height = 0;
        for (int d = 0; d < 4; d++) weyl_height += fabs(f4->roots[i][d]);
        f4->characters[i] = (f4->is_long[i] ? 2.0 : 1.0) * (1.0 + 0.1 * weyl_height);

        /* Jordan trace: proj_matrix @ root, then sum
         * proj_matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,1]]
         * diag = [root[0], root[1], root[2]+root[3]]
         * trace = root[0] + root[1] + root[2] + root[3] + root[3]
         *       = root[0] + root[1] + root[2] + 2*root[3]
         * Wait, re-reading: diag = proj @ root = [root[0], root[1], root[2]+root[3]]
         * trace = sum(diag) = root[0] + root[1] + root[2] + root[3] */
        f4->jordan_traces[i] = f4->roots[i][0] + f4->roots[i][1]
                              + f4->roots[i][2] + f4->roots[i][3];
    }
}

/* Build E8→F4 mapping via cosine similarity of first-4-coord projection */
static void f4_build_e8_mapping(F4Lattice *f4, const E8Lattice *e8)
{
    /* Precompute normalized F4 roots */
    double f4_norm_roots[F4_NUM_ROOTS][4];
    for (int i = 0; i < F4_NUM_ROOTS; i++) {
        double n = f4->norms[i];
        if (n < 0.01) n = 1.0;
        for (int d = 0; d < 4; d++) f4_norm_roots[i][d] = f4->roots[i][d] / n;
    }

    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        /* Project E8 root to first 4 coords */
        double proj[4];
        memcpy(proj, e8->roots[ei], sizeof(double) * 4);
        double proj_norm = 0;
        for (int d = 0; d < 4; d++) proj_norm += proj[d] * proj[d];
        proj_norm = sqrt(proj_norm);

        /* If degenerate, try last 4 coords */
        if (proj_norm < 0.01) {
            memcpy(proj, &e8->roots[ei][4], sizeof(double) * 4);
            proj_norm = 0;
            for (int d = 0; d < 4; d++) proj_norm += proj[d] * proj[d];
            proj_norm = sqrt(proj_norm);
        }

        if (proj_norm < 0.01) {
            f4->e8_to_f4[ei] = -1;
            f4->e8_is_f4[ei] = 0;
            continue;
        }

        /* Normalize */
        double proj_n[4];
        for (int d = 0; d < 4; d++) proj_n[d] = proj[d] / proj_norm;

        /* Find nearest F4 root by |cosine similarity| */
        int best_f4 = 0;
        double best_sim = -1.0;
        for (int fi = 0; fi < F4_NUM_ROOTS; fi++) {
            double dot = 0;
            for (int d = 0; d < 4; d++) dot += f4_norm_roots[fi][d] * proj_n[d];
            double absdot = fabs(dot);
            if (absdot > best_sim) { best_sim = absdot; best_f4 = fi; }
        }

        f4->e8_to_f4[ei] = best_f4;

        /* Compute projection quality for is_f4 check.
         * IMPORTANT: Python is_f4_root() always uses e8_root[:4], NO fallback.
         * So we must check quality using the original first-4 projection. */
        double orig_proj[4];
        memcpy(orig_proj, e8->roots[ei], sizeof(double) * 4);
        double orig_norm = 0;
        for (int d = 0; d < 4; d++) orig_norm += orig_proj[d] * orig_proj[d];
        orig_norm = sqrt(orig_norm);

        double f4_n = f4->norms[best_f4];
        if (orig_norm < 0.01 || f4_n < 0.01) {
            f4->e8_is_f4[ei] = 0;
        } else {
            double quality_dot = 0;
            for (int d = 0; d < 4; d++)
                quality_dot += orig_proj[d] * f4->roots[best_f4][d];
            double quality = fabs(quality_dot) / (orig_norm * f4_n);
            f4->e8_is_f4[ei] = (quality >= 0.7) ? 1 : 0;
        }
    }
}

static void f4_init(F4Lattice *f4, const E8Lattice *e8)
{
    f4_generate_roots(f4);
    f4_compute_properties(f4);
    f4_build_e8_mapping(f4, e8);
}

/* ================================================================
 * E7 Root System: 126 roots in R^8 (sublattice of E8)
 *
 * Constraint: v[0] == v[1]  (perpendicular to e_0 - e_1)
 * Type I (62):  ±e_i±e_j with i,j>=2 → C(6,2)×4=60; plus e_0+e_1
 *               and -(e_0+e_1) with same sign → 2
 * Type II (64): (±½)^8 with even neg AND v[0]==v[1] → 32+32
 * ================================================================ */

#define E7_NUM_ROOTS 126

typedef struct {
    double roots[E7_NUM_ROOTS][E8_DIM];   /* E7 roots live in R^8 */
    double norms[E7_NUM_ROOTS];
    double characters[E7_NUM_ROOTS];
    double jordan_traces[E7_NUM_ROOTS];   /* sum of 8 coords */
    int    is_long[E7_NUM_ROOTS];
    int    e8_to_e7[E8_NUM_ROOTS];        /* E8→E7 map (-1 = not mapped) */
    int    e8_is_e7[E8_NUM_ROOTS];        /* 1 = this E8 root IS an E7 root */
} E7Lattice;

static void e7_generate_roots(E7Lattice *e7, const E8Lattice *e8)
{
    int idx = 0;

    /* Filter E8 roots by v[0] == v[1] */
    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        if (fabs(e8->roots[i][0] - e8->roots[i][1]) < 1e-10) {
            memcpy(e7->roots[idx], e8->roots[i], sizeof(double) * E8_DIM);
            idx++;
        }
    }

    if (idx != E7_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d E7 roots, expected %d\n", idx, E7_NUM_ROOTS);
        exit(1);
    }
}

static void e7_compute_properties(E7Lattice *e7)
{
    for (int i = 0; i < E7_NUM_ROOTS; i++) {
        /* Norm */
        double n2 = 0;
        for (int d = 0; d < E8_DIM; d++) n2 += e7->roots[i][d] * e7->roots[i][d];
        e7->norms[i] = sqrt(n2);

        /* Long vs short (all E7 roots in E8 have norm sqrt(2), but classify anyway) */
        e7->is_long[i] = (e7->norms[i] > 1.2) ? 1 : 0;

        /* Character: long=2.0, short=1.0, modulated by Weyl height */
        double weyl_height = 0;
        for (int d = 0; d < E8_DIM; d++) weyl_height += fabs(e7->roots[i][d]);
        e7->characters[i] = (e7->is_long[i] ? 2.0 : 1.0) * (1.0 + 0.1 * weyl_height);

        /* Jordan trace: sum of all 8 coords */
        double trace = 0;
        for (int d = 0; d < E8_DIM; d++) trace += e7->roots[i][d];
        e7->jordan_traces[i] = trace;
    }
}

/* Build E8→E7 mapping: direct membership check (E7 roots ARE E8 roots) */
static void e7_build_e8_mapping(E7Lattice *e7, const E8Lattice *e8)
{
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        e7->e8_to_e7[ei] = -1;
        e7->e8_is_e7[ei] = 0;

        /* Check if this E8 root satisfies v[0]==v[1] */
        if (fabs(e8->roots[ei][0] - e8->roots[ei][1]) < 1e-10) {
            /* Find matching E7 root */
            for (int ri = 0; ri < E7_NUM_ROOTS; ri++) {
                int match = 1;
                for (int d = 0; d < E8_DIM; d++) {
                    if (fabs(e8->roots[ei][d] - e7->roots[ri][d]) > 1e-10) {
                        match = 0;
                        break;
                    }
                }
                if (match) {
                    e7->e8_to_e7[ei] = ri;
                    e7->e8_is_e7[ei] = 1;
                    break;
                }
            }
        } else {
            /* Not an E7 root; find nearest by cosine similarity in R^8 */
            double e8_norm = 0;
            for (int d = 0; d < E8_DIM; d++)
                e8_norm += e8->roots[ei][d] * e8->roots[ei][d];
            e8_norm = sqrt(e8_norm);
            if (e8_norm < 0.01) continue;

            int best = 0;
            double best_sim = -1.0;
            for (int ri = 0; ri < E7_NUM_ROOTS; ri++) {
                double dot = 0;
                for (int d = 0; d < E8_DIM; d++)
                    dot += e8->roots[ei][d] * e7->roots[ri][d];
                double sim = fabs(dot) / (e8_norm * e7->norms[ri]);
                if (sim > best_sim) { best_sim = sim; best = ri; }
            }
            e7->e8_to_e7[ei] = best;
            e7->e8_is_e7[ei] = (best_sim >= 0.7) ? 1 : 0;
        }
    }
}

static void e7_init(E7Lattice *e7, const E8Lattice *e8)
{
    e7_generate_roots(e7, e8);
    e7_compute_properties(e7);
    e7_build_e8_mapping(e7, e8);
}

/* ================================================================
 * E6 Root System: 72 roots in R^8 (sublattice of E8)
 *
 * Constraint: v[0] == v[1] == v[2]
 * Type I (40):  ±e_i±e_j with i,j>=3 → C(5,2)×4=40
 * Type II (32): (±½)^8 with even neg AND v[0]==v[1]==v[2] → 16+16
 * ================================================================ */

#define E6_NUM_ROOTS 72

typedef struct {
    double roots[E6_NUM_ROOTS][E8_DIM];   /* E6 roots live in R^8 */
    double norms[E6_NUM_ROOTS];
    double characters[E6_NUM_ROOTS];
    double jordan_traces[E6_NUM_ROOTS];   /* sum of 8 coords */
    int    is_long[E6_NUM_ROOTS];
    int    e8_to_e6[E8_NUM_ROOTS];
    int    e8_is_e6[E8_NUM_ROOTS];
} E6Lattice;

static void e6_generate_roots(E6Lattice *e6, const E8Lattice *e8)
{
    int idx = 0;

    /* Filter E8 roots by v[0] == v[1] == v[2] */
    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        if (fabs(e8->roots[i][0] - e8->roots[i][1]) < 1e-10 &&
            fabs(e8->roots[i][1] - e8->roots[i][2]) < 1e-10) {
            memcpy(e6->roots[idx], e8->roots[i], sizeof(double) * E8_DIM);
            idx++;
        }
    }

    if (idx != E6_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d E6 roots, expected %d\n", idx, E6_NUM_ROOTS);
        exit(1);
    }
}

static void e6_compute_properties(E6Lattice *e6)
{
    for (int i = 0; i < E6_NUM_ROOTS; i++) {
        double n2 = 0;
        for (int d = 0; d < E8_DIM; d++) n2 += e6->roots[i][d] * e6->roots[i][d];
        e6->norms[i] = sqrt(n2);

        e6->is_long[i] = (e6->norms[i] > 1.2) ? 1 : 0;

        double weyl_height = 0;
        for (int d = 0; d < E8_DIM; d++) weyl_height += fabs(e6->roots[i][d]);
        e6->characters[i] = (e6->is_long[i] ? 2.0 : 1.0) * (1.0 + 0.1 * weyl_height);

        double trace = 0;
        for (int d = 0; d < E8_DIM; d++) trace += e6->roots[i][d];
        e6->jordan_traces[i] = trace;
    }
}

static void e6_build_e8_mapping(E6Lattice *e6, const E8Lattice *e8)
{
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        e6->e8_to_e6[ei] = -1;
        e6->e8_is_e6[ei] = 0;

        if (fabs(e8->roots[ei][0] - e8->roots[ei][1]) < 1e-10 &&
            fabs(e8->roots[ei][1] - e8->roots[ei][2]) < 1e-10) {
            for (int ri = 0; ri < E6_NUM_ROOTS; ri++) {
                int match = 1;
                for (int d = 0; d < E8_DIM; d++) {
                    if (fabs(e8->roots[ei][d] - e6->roots[ri][d]) > 1e-10) {
                        match = 0; break;
                    }
                }
                if (match) {
                    e6->e8_to_e6[ei] = ri;
                    e6->e8_is_e6[ei] = 1;
                    break;
                }
            }
        } else {
            double e8_norm = 0;
            for (int d = 0; d < E8_DIM; d++)
                e8_norm += e8->roots[ei][d] * e8->roots[ei][d];
            e8_norm = sqrt(e8_norm);
            if (e8_norm < 0.01) continue;

            int best = 0;
            double best_sim = -1.0;
            for (int ri = 0; ri < E6_NUM_ROOTS; ri++) {
                double dot = 0;
                for (int d = 0; d < E8_DIM; d++)
                    dot += e8->roots[ei][d] * e6->roots[ri][d];
                double sim = fabs(dot) / (e8_norm * e6->norms[ri]);
                if (sim > best_sim) { best_sim = sim; best = ri; }
            }
            e6->e8_to_e6[ei] = best;
            e6->e8_is_e6[ei] = (best_sim >= 0.7) ? 1 : 0;
        }
    }
}

static void e6_init(E6Lattice *e6, const E8Lattice *e8)
{
    e6_generate_roots(e6, e8);
    e6_compute_properties(e6);
    e6_build_e8_mapping(e6, e8);
}

/* ================================================================
 * G2 Root System: 12 roots in R^2
 *
 * Short (6, norm 1): angles 0°, 60°, 120°, 180°, 240°, 300°
 * Long (6, norm √3): angles 30°, 90°, 150°, 210°, 270°, 330°
 *
 * E8→G2 mapping: project E8 to first 2 coords, cosine sim in R^2
 * ================================================================ */

#define G2_NUM_ROOTS 12
#define G2_DIM 2

typedef struct {
    double roots[G2_NUM_ROOTS][G2_DIM];
    double norms[G2_NUM_ROOTS];
    double characters[G2_NUM_ROOTS];
    double jordan_traces[G2_NUM_ROOTS];   /* sum of 2 coords */
    int    is_long[G2_NUM_ROOTS];
    int    e8_to_g2[E8_NUM_ROOTS];
    int    e8_is_g2[E8_NUM_ROOTS];
} G2Lattice;

static void g2_generate_roots(G2Lattice *g2)
{
    int idx = 0;

    /* Short roots (6, norm 1): angles 0°, 60°, 120°, 180°, 240°, 300° */
    for (int k = 0; k < 6; k++) {
        double angle = k * M_PI / 3.0;
        g2->roots[idx][0] = cos(angle);
        g2->roots[idx][1] = sin(angle);
        g2->is_long[idx] = 0;
        idx++;
    }

    /* Long roots (6, norm √3): angles 30°, 90°, 150°, 210°, 270°, 330° */
    for (int k = 0; k < 6; k++) {
        double angle = (k * 60.0 + 30.0) * M_PI / 180.0;
        g2->roots[idx][0] = sqrt(3.0) * cos(angle);
        g2->roots[idx][1] = sqrt(3.0) * sin(angle);
        g2->is_long[idx] = 1;
        idx++;
    }

    if (idx != G2_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d G2 roots, expected %d\n", idx, G2_NUM_ROOTS);
        exit(1);
    }
}

static void g2_compute_properties(G2Lattice *g2)
{
    for (int i = 0; i < G2_NUM_ROOTS; i++) {
        double n2 = g2->roots[i][0] * g2->roots[i][0]
                   + g2->roots[i][1] * g2->roots[i][1];
        g2->norms[i] = sqrt(n2);

        double weyl_height = fabs(g2->roots[i][0]) + fabs(g2->roots[i][1]);
        g2->characters[i] = (g2->is_long[i] ? 2.0 : 1.0) * (1.0 + 0.1 * weyl_height);

        g2->jordan_traces[i] = g2->roots[i][0] + g2->roots[i][1];
    }
}

/* E8→G2 mapping: project E8 to first 2 coords, cosine sim in R^2 */
static void g2_build_e8_mapping(G2Lattice *g2, const E8Lattice *e8)
{
    /* Precompute normalized G2 roots */
    double g2_norm_roots[G2_NUM_ROOTS][G2_DIM];
    for (int i = 0; i < G2_NUM_ROOTS; i++) {
        double n = g2->norms[i];
        if (n < 0.01) n = 1.0;
        for (int d = 0; d < G2_DIM; d++) g2_norm_roots[i][d] = g2->roots[i][d] / n;
    }

    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        /* Project E8 root to first 2 coords */
        double proj[G2_DIM];
        proj[0] = e8->roots[ei][0];
        proj[1] = e8->roots[ei][1];
        double proj_norm = sqrt(proj[0] * proj[0] + proj[1] * proj[1]);

        if (proj_norm < 0.01) {
            g2->e8_to_g2[ei] = -1;
            g2->e8_is_g2[ei] = 0;
            continue;
        }

        double proj_n[G2_DIM];
        proj_n[0] = proj[0] / proj_norm;
        proj_n[1] = proj[1] / proj_norm;

        int best = 0;
        double best_sim = -1.0;
        for (int gi = 0; gi < G2_NUM_ROOTS; gi++) {
            double dot = proj_n[0] * g2_norm_roots[gi][0]
                       + proj_n[1] * g2_norm_roots[gi][1];
            double absdot = fabs(dot);
            if (absdot > best_sim) { best_sim = absdot; best = gi; }
        }

        g2->e8_to_g2[ei] = best;
        g2->e8_is_g2[ei] = (best_sim >= 0.7) ? 1 : 0;
    }
}

static void g2_init(G2Lattice *g2, const E8Lattice *e8)
{
    g2_generate_roots(g2);
    g2_compute_properties(g2);
    g2_build_e8_mapping(g2, e8);
}

/* ================================================================
 * S(16) Half-Spinor Lattice: 128 roots in R^8 (Type II E8 roots)
 *
 * The 128 half-spinor weights of Spin(16): (±½)^8 with even # of
 * minus signs.  These are exactly the Type II E8 roots, forming
 * the vertices of the 8-dimensional demihypercube.
 *
 * 240 = 112 (D8 vector: ±e_i ± e_j) + 128 (half-spinor: (±½)^8)
 * ================================================================ */

#define S16_NUM_ROOTS 128

typedef struct {
    double roots[S16_NUM_ROOTS][E8_DIM];   /* S16 roots in R^8 */
    double norms[S16_NUM_ROOTS];
    double characters[S16_NUM_ROOTS];
    double jordan_traces[S16_NUM_ROOTS];   /* trace-8: sum of 8 coords */
    int    sign_parity[S16_NUM_ROOTS];     /* # of negative coords (always even) */
    int    e8_to_s16[E8_NUM_ROOTS];        /* E8 root idx → S16 root idx (-1 = none) */
    int    e8_is_s16[E8_NUM_ROOTS];        /* 1 if E8 root is exact S16 member */
} S16Lattice;

static void s16_generate_roots(S16Lattice *s16)
{
    int idx = 0;
    /* Enumerate all 2^8 = 256 sign patterns; keep those with even # of minus */
    for (int mask = 0; mask < 256; mask++) {
        int neg_count = 0;
        double v[E8_DIM];
        for (int d = 0; d < 8; d++) {
            if ((mask >> d) & 1) {
                v[d] = 0.5;
            } else {
                v[d] = -0.5;
                neg_count++;
            }
        }
        if (neg_count % 2 != 0) continue;
        for (int d = 0; d < 8; d++)
            s16->roots[idx][d] = v[d];
        s16->sign_parity[idx] = neg_count;
        idx++;
    }
    if (idx != S16_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d S16 roots, expected %d\n", idx, S16_NUM_ROOTS);
        exit(1);
    }
}

static void s16_compute_properties(S16Lattice *s16)
{
    for (int i = 0; i < S16_NUM_ROOTS; i++) {
        double n2 = 0, trace = 0;
        for (int d = 0; d < E8_DIM; d++) {
            n2 += s16->roots[i][d] * s16->roots[i][d];
            trace += s16->roots[i][d];
        }
        s16->norms[i] = sqrt(n2);
        s16->jordan_traces[i] = trace;
        /* Character: modulated by Weyl height (always 4.0 for half-int) */
        double weyl_h = 0;
        for (int d = 0; d < E8_DIM; d++)
            weyl_h += fabs(s16->roots[i][d]);
        s16->characters[i] = 2.0 * (1.0 + 0.1 * weyl_h);
    }
}

static void s16_build_e8_mapping(S16Lattice *s16, const E8Lattice *e8)
{
    /* Initialize to unmapped */
    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        s16->e8_to_s16[i] = -1;
        s16->e8_is_s16[i] = 0;
    }

    /* Precompute normalized S16 roots */
    double s16_norm_roots[S16_NUM_ROOTS][E8_DIM];
    for (int i = 0; i < S16_NUM_ROOTS; i++) {
        double n = s16->norms[i];
        if (n < 1e-10) n = 1.0;
        for (int d = 0; d < E8_DIM; d++)
            s16_norm_roots[i][d] = s16->roots[i][d] / n;
    }

    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        /* Check if this E8 root is a Type II root (all coords ±½) */
        int is_half_int = 1;
        for (int d = 0; d < E8_DIM; d++) {
            if (fabs(fabs(e8->roots[ei][d]) - 0.5) > 1e-10) {
                is_half_int = 0;
                break;
            }
        }

        if (is_half_int) {
            /* Find exact match among S16 roots */
            for (int si = 0; si < S16_NUM_ROOTS; si++) {
                int match = 1;
                for (int d = 0; d < E8_DIM; d++) {
                    if (fabs(e8->roots[ei][d] - s16->roots[si][d]) > 1e-10) {
                        match = 0; break;
                    }
                }
                if (match) {
                    s16->e8_to_s16[ei] = si;
                    s16->e8_is_s16[ei] = 1;
                    break;
                }
            }
            /* If exact match found with even parity, e8_is_s16 is set.
             * If half-int but ODD parity → it's the conjugate S⁻(16),
             * not in our S⁺(16).  Map via cosine sim below. */
            if (s16->e8_to_s16[ei] >= 0)
                continue;
        }

        /* Type I root (or unmatched half-int): nearest by cosine similarity */
        double e8n = 0;
        for (int d = 0; d < E8_DIM; d++)
            e8n += e8->roots[ei][d] * e8->roots[ei][d];
        e8n = sqrt(e8n);
        if (e8n < 0.01) continue;

        double best_abs_sim = -1;
        int best_si = 0;
        for (int si = 0; si < S16_NUM_ROOTS; si++) {
            double dot = 0;
            for (int d = 0; d < E8_DIM; d++)
                dot += (e8->roots[ei][d] / e8n) * s16_norm_roots[si][d];
            double absim = fabs(dot);
            if (absim > best_abs_sim) {
                best_abs_sim = absim;
                best_si = si;
            }
        }
        s16->e8_to_s16[ei] = best_si;
        /* e8_is_s16 stays 0 for approximate mappings */
    }
}

static void s16_init(S16Lattice *s16, const E8Lattice *e8)
{
    s16_generate_roots(s16);
    s16_compute_properties(s16);
    s16_build_e8_mapping(s16, e8);
}

/* ================================================================
 * D8 / SO(16) Root System: 112 roots (Type I of E8)
 *
 * The vector roots ±e_i ± e_j (i < j) in R^8.
 * These form the D8 root system, the root system of SO(16).
 * ================================================================ */

#define D8_NUM_ROOTS 112

typedef struct {
    double roots[D8_NUM_ROOTS][E8_DIM];
    double norms[D8_NUM_ROOTS];
    double characters[D8_NUM_ROOTS];
    double jordan_traces[D8_NUM_ROOTS];   /* sum of 8 coords */
    int    e8_to_d8[E8_NUM_ROOTS];        /* E8 root idx → D8 root idx (-1 = none) */
    int    e8_is_d8[E8_NUM_ROOTS];        /* 1 if E8 root is exact D8 member */
} D8Lattice;

static void d8_generate_roots(D8Lattice *d8)
{
    int idx = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            for (int s1 = -1; s1 <= 1; s1 += 2) {
                for (int s2 = -1; s2 <= 1; s2 += 2) {
                    memset(d8->roots[idx], 0, sizeof(double) * E8_DIM);
                    d8->roots[idx][i] = s1;
                    d8->roots[idx][j] = s2;
                    idx++;
                }
            }
        }
    }
    if (idx != D8_NUM_ROOTS) {
        fprintf(stderr, "BUG: generated %d D8 roots, expected %d\n", idx, D8_NUM_ROOTS);
        exit(1);
    }
}

static void d8_compute_properties(D8Lattice *d8)
{
    for (int i = 0; i < D8_NUM_ROOTS; i++) {
        double n2 = 0, trace = 0;
        for (int d = 0; d < E8_DIM; d++) {
            n2 += d8->roots[i][d] * d8->roots[i][d];
            trace += d8->roots[i][d];
        }
        d8->norms[i] = sqrt(n2);
        d8->jordan_traces[i] = trace;
        double weyl_h = 0;
        for (int d = 0; d < E8_DIM; d++)
            weyl_h += fabs(d8->roots[i][d]);
        d8->characters[i] = 2.0 * (1.0 + 0.1 * weyl_h);
    }
}

static void d8_build_e8_mapping(D8Lattice *d8, const E8Lattice *e8)
{
    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        d8->e8_to_d8[i] = -1;
        d8->e8_is_d8[i] = 0;
    }

    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        /* Check if Type I: exactly 2 nonzero coords, each ±1 */
        int nonzero = 0;
        int is_integer = 1;
        for (int d = 0; d < E8_DIM; d++) {
            double v = fabs(e8->roots[ei][d]);
            if (v > 1e-10) {
                nonzero++;
                if (fabs(v - 1.0) > 1e-10) is_integer = 0;
            }
        }
        if (nonzero == 2 && is_integer) {
            /* Exact match: find the D8 root */
            for (int di = 0; di < D8_NUM_ROOTS; di++) {
                int match = 1;
                for (int d = 0; d < E8_DIM; d++) {
                    if (fabs(e8->roots[ei][d] - d8->roots[di][d]) > 1e-10) {
                        match = 0; break;
                    }
                }
                if (match) {
                    d8->e8_to_d8[ei] = di;
                    d8->e8_is_d8[ei] = 1;
                    break;
                }
            }
        }
        /* Type II roots (half-integer) are NOT in D8 — leave unmapped */
    }
}

static void d8_init(D8Lattice *d8, const E8Lattice *e8)
{
    d8_generate_roots(d8);
    d8_compute_properties(d8);
    d8_build_e8_mapping(d8, e8);
}

/* ================================================================
 * Configurable color scale helper
 *
 * Maps a trace value in [jmin, jmax] to plasma_lut RGB.
 * ================================================================ */

static inline RGB jordan_to_color_range(double jtrace, double jmin, double jmax)
{
    double range = jmax - jmin;
    if (range < 1e-10) range = 1.0;
    double t = (jtrace - jmin) / range;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    int idx = (int)(t * 255.0);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    return plasma_lut[idx];
}

/* ================================================================
 * Prime File Loader
 *
 * Reads primes1.txt through primes50.txt from a directory.
 * Format: header line, then 10 space-separated ints per line.
 * Returns sorted unique primes > 1.
 * ================================================================ */

static int cmp_int64(const void *a, const void *b)
{
    int64_t va = *(const int64_t *)a;
    int64_t vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
}

/*
 * load_primes: reads prime files, returns sorted unique primes > 1.
 * *out_count is set to the number of primes loaded (capped at max_primes).
 * Caller must free the returned array.
 */
static int64_t *load_primes(const char *base_dir, int64_t max_primes, int64_t *out_count)
{
    /* Allocate generously — 50 files × 2M = 100M primes max */
    int64_t cap = (max_primes < 110000000) ? max_primes + 1000000 : 110000000;
    int64_t *primes = (int64_t *)malloc(cap * sizeof(int64_t));
    if (!primes) { fprintf(stderr, "malloc failed for primes array\n"); exit(1); }
    int64_t count = 0;

    for (int file_idx = 1; file_idx <= 50; file_idx++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/primes%d.txt", base_dir, file_idx);
        FILE *fp = fopen(path, "r");
        if (!fp) break;

        char line[4096];
        while (fgets(line, sizeof(line), fp)) {
            /* Parse all integers from the line */
            char *p = line;
            while (*p) {
                /* Skip non-digits */
                while (*p && !isdigit((unsigned char)*p)) p++;
                if (!*p) break;
                /* Parse number */
                int64_t val = 0;
                while (isdigit((unsigned char)*p)) {
                    val = val * 10 + (*p - '0');
                    p++;
                }
                if (val > 1 && count < cap) {
                    primes[count++] = val;
                }
            }
        }
        fclose(fp);

        if (count >= max_primes) break;
    }

    /* Sort */
    qsort(primes, count, sizeof(int64_t), cmp_int64);

    /* Deduplicate */
    int64_t uniq = 0;
    for (int64_t i = 0; i < count; i++) {
        if (i == 0 || primes[i] != primes[i - 1]) {
            primes[uniq++] = primes[i];
        }
    }
    count = uniq;

    /* Cap at max_primes */
    if (count > max_primes) count = max_primes;

    *out_count = count;
    return primes;
}

/* ================================================================
 * Segmented Sieve of Eratosthenes
 *
 * Generates primes on-the-fly, no file dependency.
 * Returns sorted array of first max_primes primes.
 * Uses segmented sieve for cache-friendly performance.
 * ================================================================ */

/* Upper bound for the Nth prime (Rosser-Schoenfeld) */
static int64_t prime_upper_bound(int64_t n)
{
    if (n < 6) return 13;
    double ln_n = log((double)n);
    double ln_ln_n = log(ln_n);
    /* p_n < n * (ln(n) + ln(ln(n))) for n >= 6 */
    return (int64_t)(n * (ln_n + ln_ln_n)) + 1000;
}

/*
 * sieve_primes: generate exactly max_primes primes via segmented sieve.
 * *out_count is set to the number produced (== max_primes).
 * Caller must free the returned array.
 */
static int64_t *sieve_primes(int64_t max_primes, int64_t *out_count)
{
    int64_t limit = prime_upper_bound(max_primes);

    /* Phase 1: simple sieve up to sqrt(limit) for base primes */
    int64_t sqrt_limit = (int64_t)sqrt((double)limit) + 1;
    uint8_t *is_composite_small = (uint8_t *)calloc(sqrt_limit + 1, 1);
    if (!is_composite_small) { fprintf(stderr, "sieve: malloc failed\n"); exit(1); }

    int64_t base_cap = (int64_t)(sqrt_limit / (log((double)sqrt_limit) - 1.1)) + 100;
    int64_t *base_primes = (int64_t *)malloc(base_cap * sizeof(int64_t));
    int64_t n_base = 0;

    for (int64_t i = 2; i <= sqrt_limit; i++) {
        if (!is_composite_small[i]) {
            base_primes[n_base++] = i;
            for (int64_t j = i * i; j <= sqrt_limit; j += i)
                is_composite_small[j] = 1;
        }
    }
    free(is_composite_small);

    /* Phase 2: segmented sieve */
    int64_t *primes = (int64_t *)malloc(max_primes * sizeof(int64_t));
    if (!primes) { fprintf(stderr, "sieve: malloc failed for primes\n"); exit(1); }
    int64_t count = 0;

    #define SEG_SIZE (1 << 18)  /* 256K — fits in L2 cache */
    uint8_t *seg = (uint8_t *)malloc(SEG_SIZE);

    for (int64_t lo = 2; lo <= limit && count < max_primes; lo += SEG_SIZE) {
        int64_t hi = lo + SEG_SIZE - 1;
        if (hi > limit) hi = limit;

        memset(seg, 0, SEG_SIZE);

        /* Mark composites in this segment */
        for (int64_t b = 0; b < n_base; b++) {
            int64_t p = base_primes[b];
            /* First multiple of p >= lo */
            int64_t start = ((lo + p - 1) / p) * p;
            if (start < p * p) start = p * p;
            if (start > hi) continue;
            for (int64_t j = start; j <= hi; j += p)
                seg[j - lo] = 1;
        }

        /* Collect primes from this segment */
        for (int64_t i = 0; i <= hi - lo && count < max_primes; i++) {
            if (!seg[i]) {
                primes[count++] = lo + i;
            }
        }
    }
    #undef SEG_SIZE

    free(seg);
    free(base_primes);

    *out_count = count;
    return primes;
}

/* ================================================================
 * Utility: format large numbers with commas
 * ================================================================ */

static char *fmt_comma(int64_t n, char *buf, int bufsz)
{
    char raw[32];
    snprintf(raw, sizeof(raw), "%ld", n);
    int len = (int)strlen(raw);
    int commas = (len - 1) / 3;
    int total = len + commas;
    if (total >= bufsz) { snprintf(buf, bufsz, "%ld", n); return buf; }
    buf[total] = '\0';
    int src = len - 1, dst = total - 1, grp = 0;
    while (src >= 0) {
        buf[dst--] = raw[src--];
        grp++;
        if (grp == 3 && src >= 0) { buf[dst--] = ','; grp = 0; }
    }
    return buf;
}

#endif /* E8_COMMON_H */
