/*
 * apsk_decoder.c — Mersenne-Sync Decoder (MSD)
 *
 * Treats prime numbers as a broadcast medium: Mersenne primes act as
 * synchronization pulses, and the 248 primes following each Mersenne
 * prime form a "listening window" whose E8 phase shifts are decoded
 * into 8-bit bytes. A triality check (3 sub-windows of 82/83/83
 * primes) validates the signal.
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o apsk_decoder apsk_decoder.c -lm
 *
 * Usage:
 *   ./apsk_decoder --max-primes 10000000000 [--output FILE] [--null-samples N]
 */

#include "e8_common.h"

#include <getopt.h>
#include <unistd.h>
#include <time.h>

/* ================================================================
 * Mersenne primes fitting in uint64_t (exponents 2..61)
 * ================================================================ */

static const uint64_t MERSENNE_PRIMES[] = {
    3ULL,                       /* 2^2 - 1 */
    7ULL,                       /* 2^3 - 1 */
    31ULL,                      /* 2^5 - 1 */
    127ULL,                     /* 2^7 - 1 */
    8191ULL,                    /* 2^13 - 1 */
    131071ULL,                  /* 2^17 - 1 */
    524287ULL,                  /* 2^19 - 1 */
    2147483647ULL,              /* 2^31 - 1 */
    2305843009213693951ULL      /* 2^61 - 1 */
};
static const int MERSENNE_EXPONENTS[] = { 2, 3, 5, 7, 13, 17, 19, 31, 61 };
#define N_MERSENNE 9
#define WINDOW_SIZE 248

/* ================================================================
 * APSK Result Structures
 * ================================================================ */

typedef struct {
    uint8_t byte_sinusoidal;
    uint8_t byte_rootproj;
    double  cartan_sin[8];
    double  cartan_root[8];
    double  vmag_sin;
    double  vmag_root;
    /* Triality: 3 sub-windows */
    uint8_t tri_sin[3];
    uint8_t tri_root[3];
    int     triality_sin;       /* 0=noise, 1=partial, 2=transcendental */
    int     triality_root;
    /* Statistics */
    double  mean_gap;
    double  std_gap;
    int     unique_roots;
    /* Window info */
    int64_t sync_prime;
    int64_t sync_index;
    int     mersenne_exp;
    int64_t window_start;
    int64_t window_end;
} APSKResult;

typedef struct {
    double mean_vmag_sin;
    double std_vmag_sin;
    double mean_vmag_root;
    double std_vmag_root;
    double triality_rate_sin;
    double triality_rate_root;
} NullStats;

/* ================================================================
 * Binary search for a value in sorted int64_t array
 * Returns index or -1 if not found
 * ================================================================ */

static int64_t find_prime_index(const int64_t *primes, int64_t n, uint64_t target)
{
    int64_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if ((uint64_t)primes[mid] == target) return mid;
        if ((uint64_t)primes[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* ================================================================
 * Decode a sub-window of primes into an 8-bit byte
 * Both sinusoidal and root-projection methods
 * ================================================================ */

static void decode_subwindow(const int64_t *primes, int64_t start_idx,
                             int n_primes, const E8Lattice *e8,
                             double cartan_sin[8], double cartan_root[8],
                             uint8_t *byte_sin, uint8_t *byte_root,
                             double *mean_gap_out, double *std_gap_out,
                             int *unique_roots_out)
{
    memset(cartan_sin, 0, sizeof(double) * 8);
    memset(cartan_root, 0, sizeof(double) * 8);

    int n_gaps = n_primes - 1;
    if (n_gaps <= 0) {
        *byte_sin = 0;
        *byte_root = 0;
        if (mean_gap_out) *mean_gap_out = 0;
        if (std_gap_out)  *std_gap_out = 0;
        if (unique_roots_out) *unique_roots_out = 0;
        return;
    }

    double *ngaps = (double *)malloc(n_gaps * sizeof(double));
    int *root_indices = (int *)malloc(n_gaps * sizeof(int));
    int root_hit[E8_NUM_ROOTS];
    memset(root_hit, 0, sizeof(root_hit));

    double sum_gap = 0, sum_gap2 = 0;

    for (int i = 0; i < n_gaps; i++) {
        int64_t p = primes[start_idx + i];
        int64_t gap = primes[start_idx + i + 1] - p;
        double log_p = log((double)p);
        double g_norm = (double)gap / log_p;

        ngaps[i] = g_norm;
        sum_gap += g_norm;
        sum_gap2 += g_norm * g_norm;

        /* E8 root assignment */
        int idx = e8_assign_root(e8, g_norm);
        root_indices[i] = idx;
        root_hit[idx] = 1;

        /* Phase for sinusoidal accumulation */
        double phase = fmod(sqrt(fmax(g_norm, 0.01)) / e8->min_norm, 1.0);
        if (phase < 0.0) phase += 1.0;

        /* Method A: sinusoidal Cartan accumulation */
        for (int d = 0; d < 8; d++)
            cartan_sin[d] += sin(2.0 * M_PI * phase * (d + 1));

        /* Method B: direct root projection */
        for (int d = 0; d < 8; d++)
            cartan_root[d] += e8->roots[idx][d];
    }

    /* Threshold to bits */
    *byte_sin = 0;
    *byte_root = 0;
    for (int d = 0; d < 8; d++) {
        if (cartan_sin[d] > 0.0)  *byte_sin  |= (1 << (7 - d));
        if (cartan_root[d] > 0.0) *byte_root |= (1 << (7 - d));
    }

    /* Statistics */
    if (mean_gap_out) *mean_gap_out = sum_gap / n_gaps;
    if (std_gap_out) {
        double mean = sum_gap / n_gaps;
        double var = sum_gap2 / n_gaps - mean * mean;
        *std_gap_out = (var > 0) ? sqrt(var) : 0;
    }
    if (unique_roots_out) {
        int uniq = 0;
        for (int r = 0; r < E8_NUM_ROOTS; r++)
            if (root_hit[r]) uniq++;
        *unique_roots_out = uniq;
    }

    free(ngaps);
    free(root_indices);
}

/* ================================================================
 * Decode a full 248-prime APSK window with triality check
 * ================================================================ */

static void decode_window(const int64_t *primes, int64_t start_idx,
                          int64_t n_primes, const E8Lattice *e8,
                          APSKResult *result)
{
    int64_t avail = n_primes - start_idx;
    int winsz = (avail >= WINDOW_SIZE) ? WINDOW_SIZE : (int)avail;

    /* Full window decode */
    decode_subwindow(primes, start_idx, winsz, e8,
                     result->cartan_sin, result->cartan_root,
                     &result->byte_sinusoidal, &result->byte_rootproj,
                     &result->mean_gap, &result->std_gap,
                     &result->unique_roots);

    /* Cartan vector magnitudes */
    result->vmag_sin = 0;
    result->vmag_root = 0;
    for (int d = 0; d < 8; d++) {
        result->vmag_sin  += result->cartan_sin[d] * result->cartan_sin[d];
        result->vmag_root += result->cartan_root[d] * result->cartan_root[d];
    }
    result->vmag_sin  = sqrt(result->vmag_sin);
    result->vmag_root = sqrt(result->vmag_root);

    /* Window range */
    result->window_start = primes[start_idx];
    result->window_end = primes[start_idx + winsz - 1];

    /* Triality: split into 3 sub-windows (82, 83, 83) */
    int sub_sizes[3] = { 82, 83, 83 };
    int64_t sub_start = start_idx;

    for (int s = 0; s < 3; s++) {
        int sub_n = sub_sizes[s];
        if (sub_start + sub_n > n_primes) sub_n = (int)(n_primes - sub_start);
        if (sub_n <= 0) {
            result->tri_sin[s] = 0;
            result->tri_root[s] = 0;
            sub_start += sub_sizes[s];
            continue;
        }

        double cs[8], cr[8];
        decode_subwindow(primes, sub_start, sub_n, e8,
                         cs, cr, &result->tri_sin[s], &result->tri_root[s],
                         NULL, NULL, NULL);
        sub_start += sub_n;
    }

    /* Classify triality — sinusoidal */
    if (result->tri_sin[0] == result->tri_sin[1] &&
        result->tri_sin[1] == result->tri_sin[2])
        result->triality_sin = 2;  /* transcendental */
    else if (result->tri_sin[0] == result->tri_sin[1] ||
             result->tri_sin[1] == result->tri_sin[2] ||
             result->tri_sin[0] == result->tri_sin[2])
        result->triality_sin = 1;  /* partial */
    else
        result->triality_sin = 0;  /* noise */

    /* Classify triality — root projection */
    if (result->tri_root[0] == result->tri_root[1] &&
        result->tri_root[1] == result->tri_root[2])
        result->triality_root = 2;
    else if (result->tri_root[0] == result->tri_root[1] ||
             result->tri_root[1] == result->tri_root[2] ||
             result->tri_root[0] == result->tri_root[2])
        result->triality_root = 1;
    else
        result->triality_root = 0;
}

/* ================================================================
 * Null distribution: random 248-prime windows
 * ================================================================ */

static NullStats compute_null(const int64_t *primes, int64_t n_primes,
                              const E8Lattice *e8, int n_samples)
{
    NullStats stats;
    memset(&stats, 0, sizeof(stats));

    if (n_primes < WINDOW_SIZE + 100 || n_samples <= 0) {
        return stats;
    }

    double *vmag_sin  = (double *)malloc(n_samples * sizeof(double));
    double *vmag_root = (double *)malloc(n_samples * sizeof(double));
    int tri_count_sin = 0, tri_count_root = 0;

    /* Seed from time + pid for reproducibility-ish */
    unsigned int seed = (unsigned int)(time(NULL) ^ getpid());

    for (int s = 0; s < n_samples; s++) {
        /* Pick a random starting index */
        int64_t max_start = n_primes - WINDOW_SIZE;
        int64_t start = (int64_t)((double)rand_r(&seed) / RAND_MAX * max_start);
        if (start < 0) start = 0;
        if (start > max_start) start = max_start;

        APSKResult result;
        memset(&result, 0, sizeof(result));
        decode_window(primes, start, n_primes, e8, &result);

        vmag_sin[s]  = result.vmag_sin;
        vmag_root[s] = result.vmag_root;
        if (result.triality_sin == 2)  tri_count_sin++;
        if (result.triality_root == 2) tri_count_root++;
    }

    /* Compute mean and std */
    double sum_s = 0, sum_s2 = 0, sum_r = 0, sum_r2 = 0;
    for (int i = 0; i < n_samples; i++) {
        sum_s  += vmag_sin[i];
        sum_s2 += vmag_sin[i] * vmag_sin[i];
        sum_r  += vmag_root[i];
        sum_r2 += vmag_root[i] * vmag_root[i];
    }

    stats.mean_vmag_sin  = sum_s / n_samples;
    stats.mean_vmag_root = sum_r / n_samples;

    double var_s = sum_s2 / n_samples - stats.mean_vmag_sin * stats.mean_vmag_sin;
    double var_r = sum_r2 / n_samples - stats.mean_vmag_root * stats.mean_vmag_root;
    stats.std_vmag_sin  = (var_s > 0) ? sqrt(var_s) : 0;
    stats.std_vmag_root = (var_r > 0) ? sqrt(var_r) : 0;

    stats.triality_rate_sin  = (double)tri_count_sin / n_samples;
    stats.triality_rate_root = (double)tri_count_root / n_samples;

    free(vmag_sin);
    free(vmag_root);

    return stats;
}

/* ================================================================
 * Report generation
 * ================================================================ */

static const char *triality_label(int t)
{
    switch (t) {
    case 2:  return "TRANSCENDENTAL";
    case 1:  return "PARTIAL";
    default: return "NOISE";
    }
}

static char printable_char(uint8_t b)
{
    return (b >= 32 && b < 127) ? (char)b : '.';
}

static void generate_report(FILE *fp, APSKResult *results, int n_results,
                            NullStats *null_stats, int64_t n_primes,
                            int null_samples, double elapsed)
{
    char buf[64], buf2[64], buf3[64];

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  MERSENNE-SYNC DECODER (APSK)\n");
    fprintf(fp, "================================================================\n");
    fprintf(fp, "Primes scanned: %s\n", fmt_comma(n_primes, buf, sizeof(buf)));
    fprintf(fp, "Mersenne sync points found: %d\n", n_results);
    fprintf(fp, "Elapsed time: %.2f s\n\n", elapsed);

    for (int i = 0; i < n_results; i++) {
        APSKResult *r = &results[i];

        fmt_comma(r->sync_prime, buf, sizeof(buf));
        fmt_comma(r->sync_index, buf2, sizeof(buf2));
        fprintf(fp, "--- SYNC #%d: M_%d = %s (index %s) ---\n",
                i + 1, r->mersenne_exp, buf, buf2);

        fmt_comma(r->sync_index + 1, buf, sizeof(buf));
        fmt_comma(r->window_start, buf2, sizeof(buf2));
        fmt_comma(r->window_end, buf3, sizeof(buf3));
        fprintf(fp, "  Window: primes[%s..+%d], range [%s, %s]\n",
                buf, WINDOW_SIZE, buf2, buf3);

        /* Method A */
        fprintf(fp, "  Method A (sinusoidal):  ");
        for (int d = 0; d < 8; d++) fprintf(fp, "%d", (r->byte_sinusoidal >> (7 - d)) & 1);
        fprintf(fp, " | 0x%02X | '%c'\n", r->byte_sinusoidal, printable_char(r->byte_sinusoidal));

        /* Method B */
        fprintf(fp, "  Method B (root proj):   ");
        for (int d = 0; d < 8; d++) fprintf(fp, "%d", (r->byte_rootproj >> (7 - d)) & 1);
        fprintf(fp, " | 0x%02X | '%c'\n", r->byte_rootproj, printable_char(r->byte_rootproj));

        /* Triality */
        fprintf(fp, "  Triality:\n");
        int sub_sizes[3] = { 82, 83, 83 };
        for (int s = 0; s < 3; s++) {
            fprintf(fp, "    Sub-window %d (%d):  ", s + 1, sub_sizes[s]);
            for (int d = 0; d < 8; d++) fprintf(fp, "%d", (r->tri_sin[s] >> (7 - d)) & 1);
            fprintf(fp, "  0x%02X", r->tri_sin[s]);
            fprintf(fp, "  |  ");
            for (int d = 0; d < 8; d++) fprintf(fp, "%d", (r->tri_root[s] >> (7 - d)) & 1);
            fprintf(fp, "  0x%02X\n", r->tri_root[s]);
        }

        fprintf(fp, "  Triality status (sin):  %s%s\n",
                triality_label(r->triality_sin),
                r->triality_sin == 2 ? " [!]" : "");
        fprintf(fp, "  Triality status (root): %s%s\n",
                triality_label(r->triality_root),
                r->triality_root == 2 ? " [!]" : "");

        /* Statistics */
        double z_sin  = (null_stats->std_vmag_sin > 0)
            ? (r->vmag_sin - null_stats->mean_vmag_sin) / null_stats->std_vmag_sin : 0;
        double z_root = (null_stats->std_vmag_root > 0)
            ? (r->vmag_root - null_stats->mean_vmag_root) / null_stats->std_vmag_root : 0;

        fprintf(fp, "  |V| sin=%.2f (null: %.2f +/- %.2f, z=%.1f)  "
                "root=%.2f (null: %.2f +/- %.2f, z=%.1f)\n",
                r->vmag_sin, null_stats->mean_vmag_sin, null_stats->std_vmag_sin, z_sin,
                r->vmag_root, null_stats->mean_vmag_root, null_stats->std_vmag_root, z_root);

        fprintf(fp, "  Gap stats: mean=%.4f, std=%.4f  |  Unique E8 roots: %d/240\n",
                r->mean_gap, r->std_gap, r->unique_roots);
        fprintf(fp, "\n");
    }

    /* Summary */
    fprintf(fp, "================================================================\n");
    fprintf(fp, "  STATISTICAL SUMMARY\n");
    fprintf(fp, "================================================================\n");
    fprintf(fp, "Null distribution (%d random windows):\n", null_samples);
    fprintf(fp, "  |V| sinusoidal:  mean=%.3f, sigma=%.3f\n",
            null_stats->mean_vmag_sin, null_stats->std_vmag_sin);
    fprintf(fp, "  |V| root proj:   mean=%.3f, sigma=%.3f\n",
            null_stats->mean_vmag_root, null_stats->std_vmag_root);
    fprintf(fp, "  Triality match rate (sin):  %.1f%% (random expectation: ~0.4%%)\n",
            null_stats->triality_rate_sin * 100.0);
    fprintf(fp, "  Triality match rate (root): %.1f%% (random expectation: ~0.4%%)\n",
            null_stats->triality_rate_root * 100.0);

    /* Decoded message */
    fprintf(fp, "\nDecoded message (Method A): ");
    for (int i = 0; i < n_results; i++) {
        if (i > 0) fprintf(fp, " ");
        fprintf(fp, "%c", printable_char(results[i].byte_sinusoidal));
    }
    fprintf(fp, "\nDecoded message (Method B): ");
    for (int i = 0; i < n_results; i++) {
        if (i > 0) fprintf(fp, " ");
        fprintf(fp, "%c", printable_char(results[i].byte_rootproj));
    }
    fprintf(fp, "\n\nHex (A): ");
    for (int i = 0; i < n_results; i++)
        fprintf(fp, "%02X ", results[i].byte_sinusoidal);
    fprintf(fp, "\nHex (B): ");
    for (int i = 0; i < n_results; i++)
        fprintf(fp, "%02X ", results[i].byte_rootproj);
    fprintf(fp, "\n");
}

/* ================================================================
 * Main
 * ================================================================ */

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --max-primes N     Number of primes to sieve [default: 10000000000]\n");
    fprintf(stderr, "  --output FILE      Output report file [default: spiral_outputs/apsk_report.txt]\n");
    fprintf(stderr, "  --null-samples N   Random windows for null distribution [default: 1000]\n");
    fprintf(stderr, "  --help             Show this help\n");
}

int main(int argc, char **argv)
{
    int64_t max_primes = 10000000000LL;
    const char *output_path = "spiral_outputs/apsk_report.txt";
    int null_samples = 1000;

    static struct option long_options[] = {
        {"max-primes",   required_argument, 0, 'p'},
        {"output",       required_argument, 0, 'o'},
        {"null-samples", required_argument, 0, 'n'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:o:n:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'p': max_primes = atoll(optarg); break;
        case 'o': output_path = optarg; break;
        case 'n': null_samples = atoi(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    char buf[64];

    printf("================================================================\n");
    printf("  MERSENNE-SYNC DECODER (APSK)\n");
    printf("================================================================\n");
    printf("Max primes: %s\n", fmt_comma(max_primes, buf, sizeof(buf)));
    printf("Output: %s\n", output_path);
    printf("Null samples: %d\n\n", null_samples);

    /* Initialize E8 */
    printf("Initializing E8 lattice... ");
    fflush(stdout);
    E8Lattice e8;
    e8_init(&e8);
    printf("done (240 roots)\n");

    /* Sieve primes */
    printf("Sieving primes... ");
    fflush(stdout);
    tic();
    int64_t n_primes = 0;
    int64_t *primes = sieve_primes(max_primes, &n_primes);
    double sieve_time = toc();
    printf("done: %s primes in %.2f s\n",
           fmt_comma(n_primes, buf, sizeof(buf)), sieve_time);
    printf("Largest prime: %s\n\n",
           fmt_comma(primes[n_primes - 1], buf, sizeof(buf)));

    /* Find Mersenne sync points */
    printf("Scanning for Mersenne sync points...\n");
    tic();

    char buf2[64], buf3[64];
    APSKResult results[N_MERSENNE];
    int n_found = 0;

    for (int m = 0; m < N_MERSENNE; m++) {
        int64_t idx = find_prime_index(primes, n_primes, MERSENNE_PRIMES[m]);
        if (idx < 0) {
            printf("  M_%d = %s: NOT in sieve range\n",
                   MERSENNE_EXPONENTS[m],
                   fmt_comma((int64_t)MERSENNE_PRIMES[m], buf, sizeof(buf)));
            continue;
        }

        /* Check if we have enough primes after the sync point */
        if (idx + 1 + WINDOW_SIZE > n_primes) {
            fmt_comma((int64_t)MERSENNE_PRIMES[m], buf, sizeof(buf));
            fmt_comma(idx, buf2, sizeof(buf2));
            fmt_comma(n_primes - idx - 1, buf3, sizeof(buf3));
            printf("  M_%d = %s (index %s): insufficient primes for window (%s remain)\n",
                   MERSENNE_EXPONENTS[m], buf, buf2, buf3);
            continue;
        }

        fmt_comma((int64_t)MERSENNE_PRIMES[m], buf, sizeof(buf));
        fmt_comma(idx, buf2, sizeof(buf2));
        printf("  M_%d = %s (index %s): decoding window...",
               MERSENNE_EXPONENTS[m], buf, buf2);
        fflush(stdout);

        APSKResult *r = &results[n_found];
        memset(r, 0, sizeof(*r));
        r->sync_prime = (int64_t)MERSENNE_PRIMES[m];
        r->sync_index = idx;
        r->mersenne_exp = MERSENNE_EXPONENTS[m];

        decode_window(primes, idx + 1, n_primes, &e8, r);

        printf(" 0x%02X '%c' / 0x%02X '%c'  [%s]\n",
               r->byte_sinusoidal, printable_char(r->byte_sinusoidal),
               r->byte_rootproj, printable_char(r->byte_rootproj),
               triality_label(r->triality_sin));

        n_found++;
    }

    double scan_time = toc();
    printf("\nFound %d sync points in %.4f s\n\n", n_found, scan_time);

    /* Null distribution */
    printf("Computing null distribution (%d random windows)... ", null_samples);
    fflush(stdout);
    tic();
    NullStats null_stats = compute_null(primes, n_primes, &e8, null_samples);
    double null_time = toc();
    printf("done in %.2f s\n", null_time);
    printf("  Null |V| sin:  mean=%.3f, sigma=%.3f\n",
           null_stats.mean_vmag_sin, null_stats.std_vmag_sin);
    printf("  Null |V| root: mean=%.3f, sigma=%.3f\n",
           null_stats.mean_vmag_root, null_stats.std_vmag_root);
    printf("  Null triality (sin):  %.1f%%\n", null_stats.triality_rate_sin * 100.0);
    printf("  Null triality (root): %.1f%%\n\n", null_stats.triality_rate_root * 100.0);

    /* Generate report */
    printf("Writing report to %s... ", output_path);
    fflush(stdout);
    FILE *fp = fopen(output_path, "w");
    if (!fp) {
        fprintf(stderr, "\nERROR: cannot open %s: %s\n", output_path, strerror(errno));
        free(primes);
        return 1;
    }

    double total_elapsed = sieve_time + scan_time + null_time;
    generate_report(fp, results, n_found, &null_stats, n_primes,
                    null_samples, total_elapsed);
    fclose(fp);
    printf("done\n");

    /* Print decoded messages to stdout */
    printf("\n================================================================\n");
    printf("  DECODED MESSAGES\n");
    printf("================================================================\n");
    printf("Method A (sinusoidal): ");
    for (int i = 0; i < n_found; i++)
        printf("%c", printable_char(results[i].byte_sinusoidal));
    printf("\nMethod B (root proj):  ");
    for (int i = 0; i < n_found; i++)
        printf("%c", printable_char(results[i].byte_rootproj));
    printf("\n\n");

    /* Count triality matches */
    int tri_trans_sin = 0, tri_trans_root = 0;
    for (int i = 0; i < n_found; i++) {
        if (results[i].triality_sin == 2) tri_trans_sin++;
        if (results[i].triality_root == 2) tri_trans_root++;
    }
    printf("Transcendental signals: %d/%d (sin), %d/%d (root)\n",
           tri_trans_sin, n_found, tri_trans_root, n_found);

    free(primes);
    return 0;
}
