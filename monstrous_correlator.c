/*
 * monstrous_correlator.c — MC Pass-8 Moonshine Decoder
 *
 * Isolates transcendental triplets — groups of 3 consecutive prime gaps
 * whose E8 root vectors are highly coherent — and correlates their
 * spectral density with j-function (Monster group moonshine) coefficients.
 *
 * Key metric: Salem-Jordan coherence = ‖r₁+r₂+r₃‖² / (‖r₁‖²+‖r₂‖²+‖r₃‖²)
 *   Random expectation ~ 1/3 (8D isotropic)
 *   Empirical mean at 1B ~ 1.055
 *   Perfect sync (r₁=r₂=r₃): 3.0
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o monstrous_correlator monstrous_correlator.c -lm
 *
 * Usage:
 *   ./monstrous_correlator --max-primes 10000000000 [--checkpoint mc_state.ebd] [--resume]
 */

#include "e8_common.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <time.h>

/* ================================================================
 * Constants
 * ================================================================ */

#define MC_MAGIC         0x4D43503855ULL   /* "MCP8U" */
#define MC_VERSION       1
#define BATCH_CAPACITY   (1 << 20)         /* 1M primes per processing batch */
#define SIEVE_SEGMENT    (1 << 19)         /* 512K per sieve segment */
#define DEFAULT_CKPT_INT (100000000LL)     /* checkpoint every 100M primes */
#define MAX_EVENTS       256               /* stored transcendental events */
#define MAX_BYTES        256               /* stored exceptional bytes */
#define COHERENCE_BINS   20
#define NUM_TIERS        4
#define J_COEFFS_N       15
#define DUAL_COXETER_H   30.0              /* E8 dual Coxeter number h* */
#define NULL_SIEVE_SIZE  1000000LL         /* 1M primes for null distribution */

/* Tier thresholds (Salem-Jordan coherence) */
#define TIER3_THRESHOLD  2.5               /* TRANSCENDENTAL */
#define TIER2_THRESHOLD  2.0               /* RESONANT */
#define TIER1_THRESHOLD  1.5               /* HARMONIC */

/* j-function coefficients (OEIS A000521, first 15 terms of q-expansion) */
static const double J_COEFFS[J_COEFFS_N] = {
    1.0, 196884.0, 21493760.0, 864299970.0, 20245856256.0,
    333202640600.0, 4252023300096.0, 44656994071935.0,
    401490886656000.0, 3176440229784420.0, 22567393309593600.0,
    146211911499519294.0, 874313719685775360.0,
    4872010111798142520.0, 25497827389410525184.0
};

/* ================================================================
 * Transcendental Event Record
 * ================================================================ */

typedef struct {
    int64_t primes[4];         /* the 4 primes forming 3 consecutive gaps */
    double  coherence;         /* Salem-Jordan coherence */
    uint8_t byte;              /* exceptional byte (8-bit sign pattern) */
    uint8_t _pad[7];
    int64_t triplet_index;     /* which triplet number this was */
} MoonstoneEvent;

/* ================================================================
 * Checkpoint State (mmap'd)
 * ================================================================ */

typedef struct {
    /* Header */
    uint64_t magic;
    uint32_t version;
    uint32_t _pad0;

    /* Progress */
    int64_t primes_processed;
    int64_t max_primes;
    int64_t gaps_analyzed;
    int64_t triplets_analyzed;
    int64_t last_prime;

    /* Partial triplet state */
    int     tri_phase;          /* 0, 1, or 2 — which gap of current triplet */
    int     _pad1;
    double  tri_acc[3][E8_DIM]; /* accumulated root vectors per phase */
    int     tri_root_idx[3];    /* root indices per phase */
    int     _pad2;
    int64_t tri_primes[4];      /* primes forming current partial triplet */

    /* Coherence statistics */
    double  sum_coherence;
    double  sum_coherence2;
    int64_t tier_count[NUM_TIERS];       /* tier 0..3 */
    int64_t coherence_hist[COHERENCE_BINS];

    /* Online Pearson correlation (log j-function vs coherence) */
    double  pearson_sum_x;
    double  pearson_sum_y;
    double  pearson_sum_x2;
    double  pearson_sum_y2;
    double  pearson_sum_xy;
    int64_t pearson_n;

    /* Exceptional byte statistics */
    int64_t byte_freq[256];
    int64_t n_transcendental;
    int64_t stored_events;
    int64_t stored_bytes;

    /* Autocorrelation */
    int64_t last_transcendental_triplet;  /* triplet index of last tier-3 */
    double  sum_spacing;
    double  sum_spacing2;
    int64_t n_spacings;

    /* Null distribution */
    double  null_mean_coherence;
    double  null_std_coherence;
    double  null_transcendental_rate;
    int     null_computed;
    int     null_samples;

    /* Timing */
    double  elapsed_seconds;
    double  elapsed_at_resume;

    char    _pad4[48];  /* future expansion */
} MCCheckpointState;

/* File layout: [State][Events[256]][Bytes[256]] + page padding */
#define CKPT_EVENTS_OFF   ((int64_t)sizeof(MCCheckpointState))
#define CKPT_BYTES_OFF    (CKPT_EVENTS_OFF + (int64_t)(MAX_EVENTS * sizeof(MoonstoneEvent)))
#define CKPT_FILE_SIZE    ((CKPT_BYTES_OFF + MAX_BYTES + 4095) & ~4095)

typedef struct {
    int                fd;
    void              *map;
    MCCheckpointState *state;
    MoonstoneEvent    *events;
    uint8_t           *bytes;
    char               path[512];
} MCCheckpointFile;

/* ================================================================
 * Checkpoint I/O (mmap'd, MAP_SHARED + msync)
 * ================================================================ */

static int mc_ckpt_open(MCCheckpointFile *ckpt, const char *path, int resume)
{
    strncpy(ckpt->path, path, sizeof(ckpt->path) - 1);
    ckpt->path[sizeof(ckpt->path) - 1] = '\0';

    int flags = O_RDWR;
    if (!resume) flags |= O_CREAT | O_TRUNC;

    ckpt->fd = open(path, flags, 0644);
    if (ckpt->fd < 0 && resume) {
        fprintf(stderr, "Cannot open checkpoint %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (ckpt->fd < 0) {
        ckpt->fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (ckpt->fd < 0) {
            fprintf(stderr, "Cannot create checkpoint %s: %s\n", path, strerror(errno));
            return -1;
        }
    }

    if (ftruncate(ckpt->fd, CKPT_FILE_SIZE) < 0) {
        fprintf(stderr, "ftruncate: %s\n", strerror(errno));
        close(ckpt->fd);
        return -1;
    }

    ckpt->map = mmap(NULL, CKPT_FILE_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, ckpt->fd, 0);
    if (ckpt->map == MAP_FAILED) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        close(ckpt->fd);
        return -1;
    }

    ckpt->state  = (MCCheckpointState *)ckpt->map;
    ckpt->events = (MoonstoneEvent *)((uint8_t *)ckpt->map + CKPT_EVENTS_OFF);
    ckpt->bytes  = (uint8_t *)ckpt->map + CKPT_BYTES_OFF;

    if (resume) {
        if (ckpt->state->magic != MC_MAGIC || ckpt->state->version != MC_VERSION) {
            fprintf(stderr, "Invalid checkpoint (bad magic/version)\n");
            munmap(ckpt->map, CKPT_FILE_SIZE);
            close(ckpt->fd);
            return -1;
        }
    } else {
        memset(ckpt->map, 0, CKPT_FILE_SIZE);
        ckpt->state->magic   = MC_MAGIC;
        ckpt->state->version = MC_VERSION;
        ckpt->state->last_transcendental_triplet = -1;
    }
    return 0;
}

static void mc_ckpt_sync(MCCheckpointFile *ckpt)
{
    msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
}

static void mc_ckpt_close(MCCheckpointFile *ckpt)
{
    if (ckpt->map && ckpt->map != MAP_FAILED) {
        msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
        munmap(ckpt->map, CKPT_FILE_SIZE);
    }
    if (ckpt->fd >= 0) close(ckpt->fd);
}

/* ================================================================
 * Streaming Segmented Sieve
 *
 * Generates primes on-the-fly in 1M batches via 512K sieve segments.
 * OpenMP parallel composite marking.
 * ================================================================ */

typedef struct {
    int64_t *base_primes;
    int64_t  n_base;
    int64_t  current_lo;
    int64_t  limit;
    uint8_t *segment;
    int64_t *batch;
    int64_t  batch_count;
    int64_t  total_produced;
    int64_t  max_primes;
} StreamingSieve;

static void ss_init(StreamingSieve *sv, int64_t max_primes, int64_t resume_after)
{
    sv->max_primes = max_primes;
    sv->total_produced = 0;
    sv->batch_count = 0;
    sv->limit = prime_upper_bound(max_primes);
    sv->limit += sv->limit / 20 + 100000;  /* Rosser-Schoenfeld safety margin */

    int64_t sqrt_limit = (int64_t)sqrt((double)sv->limit) + 1;
    uint8_t *is_comp = (uint8_t *)calloc(sqrt_limit + 1, 1);
    if (!is_comp) { fprintf(stderr, "ss_init: calloc failed\n"); exit(1); }

    int64_t base_cap = (int64_t)(sqrt_limit / (log((double)(sqrt_limit + 2)) - 1.1)) + 1000;
    sv->base_primes = (int64_t *)malloc(base_cap * sizeof(int64_t));
    sv->n_base = 0;

    for (int64_t i = 2; i <= sqrt_limit; i++) {
        if (!is_comp[i]) {
            sv->base_primes[sv->n_base++] = i;
            for (int64_t j = i * i; j <= sqrt_limit; j += i)
                is_comp[j] = 1;
        }
    }
    free(is_comp);

    sv->segment = (uint8_t *)malloc(SIEVE_SEGMENT);
    sv->batch   = (int64_t *)malloc(BATCH_CAPACITY * sizeof(int64_t));
    sv->current_lo = (resume_after > 1) ? resume_after + 1 : 2;
}

static int64_t ss_next_batch(StreamingSieve *sv)
{
    sv->batch_count = 0;

    while (sv->batch_count < BATCH_CAPACITY &&
           sv->current_lo <= sv->limit &&
           sv->total_produced < sv->max_primes)
    {
        int64_t lo = sv->current_lo;
        int64_t hi = lo + SIEVE_SEGMENT - 1;
        if (hi > sv->limit) hi = sv->limit;
        int64_t seg_len = hi - lo + 1;

        memset(sv->segment, 0, (size_t)seg_len);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int64_t b = 0; b < sv->n_base; b++) {
            int64_t p = sv->base_primes[b];
            int64_t start = ((lo + p - 1) / p) * p;
            if (start < p * p) start = p * p;
            if (start > hi) continue;
            for (int64_t j = start; j <= hi; j += p)
                sv->segment[j - lo] = 1;
        }

        for (int64_t i = 0; i < seg_len &&
                 sv->batch_count < BATCH_CAPACITY &&
                 sv->total_produced < sv->max_primes; i++) {
            if (!sv->segment[i]) {
                sv->batch[sv->batch_count++] = lo + i;
                sv->total_produced++;
            }
        }

        sv->current_lo = hi + 1;
    }
    return sv->batch_count;
}

static void ss_destroy(StreamingSieve *sv)
{
    free(sv->base_primes);
    free(sv->segment);
    free(sv->batch);
}

/* ================================================================
 * Signal handler
 * ================================================================ */

static volatile sig_atomic_t g_shutdown = 0;

static void handle_signal(int sig)
{
    (void)sig;
    g_shutdown = 1;
}

/* ================================================================
 * Null Distribution
 *
 * Sieve ~1M primes separately, sample random triplets, compute
 * coherence for each.  Establishes baseline for z-scores.
 * ================================================================ */

typedef struct {
    double mean_coherence;
    double std_coherence;
    double transcendental_rate;
    int    n_samples;
} MCNullStats;

static MCNullStats compute_null_distribution(const E8Lattice *e8, int n_samples)
{
    MCNullStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.n_samples = n_samples;

    int64_t n_null = 0;
    int64_t *null_primes = sieve_primes(NULL_SIEVE_SIZE, &n_null);

    if (n_null < 10) {
        free(null_primes);
        return stats;
    }

    int64_t max_start = n_null - 4;
    if (max_start < 1) {
        free(null_primes);
        return stats;
    }

    double *coh_arr = (double *)malloc(n_samples * sizeof(double));
    int n_trans = 0;
    unsigned int seed = (unsigned int)(time(NULL) ^ getpid());

    for (int s = 0; s < n_samples; s++) {
        int64_t start = (int64_t)((double)rand_r(&seed) / RAND_MAX * max_start);
        if (start < 0) start = 0;
        if (start > max_start) start = max_start;

        /* 3 consecutive gaps from 4 consecutive primes */
        double sum_vec[E8_DIM];
        memset(sum_vec, 0, sizeof(sum_vec));

        for (int g = 0; g < 3; g++) {
            int64_t p = null_primes[start + g];
            int64_t gap = null_primes[start + g + 1] - p;
            double log_p = log((double)p);
            if (log_p < 1.0) log_p = 1.0;
            double norm_gap = (double)gap / log_p;
            int idx = e8_assign_root(e8, norm_gap);
            for (int d = 0; d < E8_DIM; d++)
                sum_vec[d] += e8->roots[idx][d];
        }

        double dot = 0;
        for (int d = 0; d < E8_DIM; d++)
            dot += sum_vec[d] * sum_vec[d];
        double coh = dot / 6.0;

        coh_arr[s] = coh;
        if (coh > TIER3_THRESHOLD) n_trans++;
    }

    double sum_c = 0, sum_c2 = 0;
    for (int i = 0; i < n_samples; i++) {
        sum_c  += coh_arr[i];
        sum_c2 += coh_arr[i] * coh_arr[i];
    }
    stats.mean_coherence = sum_c / n_samples;
    double var = sum_c2 / n_samples - stats.mean_coherence * stats.mean_coherence;
    stats.std_coherence = (var > 0) ? sqrt(var) : 0;
    stats.transcendental_rate = (double)n_trans / n_samples;

    free(coh_arr);
    free(null_primes);
    return stats;
}

/* ================================================================
 * Batch processing: gaps → E8 roots → triplet coherence
 * ================================================================ */

static void process_batch(const int64_t *batch, int64_t count,
                          int64_t *prev_prime_ptr,
                          const E8Lattice *e8,
                          MCCheckpointFile *ckpt)
{
    MCCheckpointState *st = ckpt->state;
    int64_t prev = *prev_prime_ptr;

    int64_t start_idx = 0;
    if (prev <= 0) {
        /* First prime ever — set as base, no gap yet */
        prev = batch[0];
        st->tri_primes[0] = prev;
        start_idx = 1;
    }

    for (int64_t i = start_idx; i < count; i++) {
        int64_t p = batch[i];
        int64_t gap = p - prev;
        double log_p = log((double)prev);
        if (log_p < 1.0) log_p = 1.0;
        double norm_gap = (double)gap / log_p;

        st->gaps_analyzed++;

        /* E8 root assignment */
        int root_idx = e8_assign_root(e8, norm_gap);

        /* Store root vector in triplet accumulator */
        for (int d = 0; d < E8_DIM; d++)
            st->tri_acc[st->tri_phase][d] = e8->roots[root_idx][d];
        st->tri_root_idx[st->tri_phase] = root_idx;

        /* Track primes forming the triplet:
         * tri_primes[0] = first prime, [1..3] = subsequent primes */
        if (st->tri_phase == 0)
            st->tri_primes[0] = prev;
        st->tri_primes[st->tri_phase + 1] = p;

        st->tri_phase++;

        /* Triplet complete? */
        if (st->tri_phase >= 3) {
            st->triplets_analyzed++;

            /* Sum vector: r₁ + r₂ + r₃ */
            double sum_vec[E8_DIM];
            double norm2_sum = 0;
            for (int d = 0; d < E8_DIM; d++) {
                sum_vec[d] = st->tri_acc[0][d]
                           + st->tri_acc[1][d]
                           + st->tri_acc[2][d];
                norm2_sum += sum_vec[d] * sum_vec[d];
            }

            /* Coherence = ‖sum‖² / (‖r₁‖² + ‖r₂‖² + ‖r₃‖²)
             * All E8 roots have norm² = 2, so denominator = 6.0 */
            double coh = norm2_sum / 6.0;

            /* Accumulate coherence statistics */
            st->sum_coherence  += coh;
            st->sum_coherence2 += coh * coh;

            /* Histogram: bin = (int)(coh / 0.15) clamped to [0,19] */
            int bin = (int)(coh / 0.15);
            if (bin < 0) bin = 0;
            if (bin >= COHERENCE_BINS) bin = COHERENCE_BINS - 1;
            st->coherence_hist[bin]++;

            /* Tier classification */
            int tier;
            if (coh > TIER3_THRESHOLD)      tier = 3;
            else if (coh > TIER2_THRESHOLD) tier = 2;
            else if (coh > TIER1_THRESHOLD) tier = 1;
            else                            tier = 0;
            st->tier_count[tier]++;

            /* Tier 3: TRANSCENDENTAL — full analysis */
            if (tier == 3) {
                st->n_transcendental++;

                /* Exceptional byte: bit d = 1 if sum[d] > 0 */
                uint8_t ebyte = 0;
                for (int d = 0; d < E8_DIM; d++) {
                    if (sum_vec[d] > 0.0)
                        ebyte |= (uint8_t)(1 << (7 - d));
                }
                st->byte_freq[ebyte]++;

                /* Exceptional magnitude (Killing form, dual Coxeter h*=30) */
                /* M_E = sqrt(coh * 6.0) / 30.0 = ‖sum‖ / h* */
                /* (used for display, not stored separately) */

                /* j-correlation: log-Pearson with J_COEFFS[n % 15] */
                int j_idx = (int)((st->n_transcendental - 1) % J_COEFFS_N);
                double x = log(J_COEFFS[j_idx]);
                double y = coh;
                st->pearson_sum_x  += x;
                st->pearson_sum_y  += y;
                st->pearson_sum_x2 += x * x;
                st->pearson_sum_y2 += y * y;
                st->pearson_sum_xy += x * y;
                st->pearson_n++;

                /* Record event (first 256 stored) */
                if (st->stored_events < MAX_EVENTS) {
                    MoonstoneEvent *ev = &ckpt->events[st->stored_events];
                    ev->primes[0] = st->tri_primes[0];
                    ev->primes[1] = st->tri_primes[1];
                    ev->primes[2] = st->tri_primes[2];
                    ev->primes[3] = st->tri_primes[3];
                    ev->coherence = coh;
                    ev->byte = ebyte;
                    ev->triplet_index = st->triplets_analyzed;
                    st->stored_events++;
                }

                /* Store exceptional byte */
                if (st->stored_bytes < MAX_BYTES) {
                    ckpt->bytes[st->stored_bytes] = ebyte;
                    st->stored_bytes++;
                }

                /* Autocorrelation: spacing from last transcendental */
                if (st->last_transcendental_triplet >= 0) {
                    int64_t spacing = st->triplets_analyzed
                                    - st->last_transcendental_triplet;
                    st->sum_spacing  += (double)spacing;
                    st->sum_spacing2 += (double)spacing * (double)spacing;
                    st->n_spacings++;
                }
                st->last_transcendental_triplet = st->triplets_analyzed;
            }

            /* Reset triplet accumulator */
            st->tri_phase = 0;
            memset(st->tri_acc, 0, sizeof(st->tri_acc));
        }

        prev = p;
    }

    *prev_prime_ptr = prev;
}

/* ================================================================
 * Report Generation
 * ================================================================ */

static const char *tier_name(int tier)
{
    switch (tier) {
    case 3:  return "TRANSCENDENTAL";
    case 2:  return "RESONANT";
    case 1:  return "HARMONIC";
    default: return "NOISE";
    }
}

static void generate_report(FILE *fp, MCCheckpointFile *ckpt)
{
    MCCheckpointState *st = ckpt->state;
    char b1[64], b2[64];

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  MONSTROUS CORRELATOR (MC Pass-8 Moonshine)\n");
    fprintf(fp, "================================================================\n");
    fprintf(fp, "Primes processed:    %s / %s\n",
            fmt_comma(st->primes_processed, b1, sizeof(b1)),
            fmt_comma(st->max_primes, b2, sizeof(b2)));
    fprintf(fp, "Gaps analyzed:       %s\n",
            fmt_comma(st->gaps_analyzed, b1, sizeof(b1)));
    fprintf(fp, "Triplets analyzed:   %s\n",
            fmt_comma(st->triplets_analyzed, b1, sizeof(b1)));
    fprintf(fp, "Last prime:          %s\n",
            fmt_comma(st->last_prime, b1, sizeof(b1)));
    fprintf(fp, "Elapsed:             %.2f s (%.2f hrs)\n\n",
            st->elapsed_seconds, st->elapsed_seconds / 3600.0);

    /* --- COHERENCE SPECTRUM --- */
    fprintf(fp, "--- COHERENCE SPECTRUM ---\n");
    if (st->triplets_analyzed > 0) {
        double mean_c = st->sum_coherence / st->triplets_analyzed;
        double var_c  = st->sum_coherence2 / st->triplets_analyzed - mean_c * mean_c;
        double std_c  = (var_c > 0) ? sqrt(var_c) : 0;
        fprintf(fp, "  Mean coherence:    %.6f (random ~ 0.333)\n", mean_c);
        fprintf(fp, "  Std coherence:     %.6f\n", std_c);
    }
    fprintf(fp, "  Tier counts:\n");
    for (int t = 3; t >= 0; t--) {
        double pct = st->triplets_analyzed > 0
            ? 100.0 * st->tier_count[t] / st->triplets_analyzed : 0;
        fprintf(fp, "    Tier %d %-16s: %12s (%7.3f%%)\n",
                t, tier_name(t),
                fmt_comma(st->tier_count[t], b1, sizeof(b1)), pct);
    }
    fprintf(fp, "\n");

    /* --- COHERENCE HISTOGRAM --- */
    fprintf(fp, "--- COHERENCE HISTOGRAM ---\n");
    int64_t max_hist = 0;
    for (int i = 0; i < COHERENCE_BINS; i++)
        if (st->coherence_hist[i] > max_hist) max_hist = st->coherence_hist[i];

    for (int i = 0; i < COHERENCE_BINS; i++) {
        double lo = i * 0.15;
        double hi = (i + 1) * 0.15;
        int bar = max_hist > 0
            ? (int)(40.0 * st->coherence_hist[i] / max_hist) : 0;
        fprintf(fp, "  [%4.2f,%4.2f) ", lo, hi);
        for (int j = 0; j < 40; j++)
            fputc(j < bar ? '#' : ' ', fp);
        fprintf(fp, " %12s\n",
                fmt_comma(st->coherence_hist[i], b1, sizeof(b1)));
    }
    fprintf(fp, "\n");

    /* --- TRANSCENDENTAL TRIPLET ANALYSIS --- */
    fprintf(fp, "--- TRANSCENDENTAL TRIPLET ANALYSIS ---\n");
    fprintf(fp, "  Count: %s\n",
            fmt_comma(st->n_transcendental, b1, sizeof(b1)));

    if (st->n_transcendental > 0) {
        /* Byte entropy */
        double H = 0;
        for (int i = 0; i < 256; i++) {
            if (st->byte_freq[i] > 0) {
                double p = (double)st->byte_freq[i] / st->n_transcendental;
                H -= p * log2(p);
            }
        }
        fprintf(fp, "  Byte entropy:  %.4f bits (max = 8.0)\n", H);

        /* Top 10 exceptional bytes by frequency */
        fprintf(fp, "  Top 10 exceptional bytes:\n");
        int sorted[256];
        for (int i = 0; i < 256; i++) sorted[i] = i;
        for (int i = 0; i < 10 && i < 256; i++) {
            int best = i;
            for (int j = i + 1; j < 256; j++) {
                if (st->byte_freq[sorted[j]] > st->byte_freq[sorted[best]])
                    best = j;
            }
            if (best != i) {
                int tmp = sorted[i];
                sorted[i] = sorted[best];
                sorted[best] = tmp;
            }
        }
        for (int i = 0; i < 10; i++) {
            int idx = sorted[i];
            if (st->byte_freq[idx] == 0) break;
            fprintf(fp, "    0x%02X: %12s (%5.2f%%)\n", idx,
                    fmt_comma(st->byte_freq[idx], b1, sizeof(b1)),
                    100.0 * st->byte_freq[idx] / st->n_transcendental);
        }
    }
    fprintf(fp, "\n");

    /* --- MONSTROUS CORRELATION (Γ) --- */
    fprintf(fp, "--- MONSTROUS CORRELATION (\xCE\x93) ---\n");
    if (st->pearson_n > 1) {
        double n = (double)st->pearson_n;
        double num   = n * st->pearson_sum_xy
                     - st->pearson_sum_x * st->pearson_sum_y;
        double den_x = n * st->pearson_sum_x2
                     - st->pearson_sum_x * st->pearson_sum_x;
        double den_y = n * st->pearson_sum_y2
                     - st->pearson_sum_y * st->pearson_sum_y;
        double denom = sqrt(fabs(den_x) * fabs(den_y));
        double gamma = (denom > 1e-15) ? num / denom : 0;
        fprintf(fp, "  Pearson \xCE\x93:  %.6f\n", gamma);
        fprintf(fp, "  Status:      %s\n",
                fabs(gamma) > 0.05
                    ? "VERIFIED (nonzero correlation)"
                    : "DECOHERENCE (no significant correlation)");
        fprintf(fp, "  Samples:     %s\n",
                fmt_comma(st->pearson_n, b1, sizeof(b1)));
    } else {
        fprintf(fp, "  (insufficient data)\n");
    }
    fprintf(fp, "\n");

    /* --- j-FUNCTION COEFFICIENTS --- */
    fprintf(fp, "--- j-FUNCTION COEFFICIENTS (A000521) ---\n");
    for (int i = 0; i < J_COEFFS_N; i++)
        fprintf(fp, "  c_%d = %.0f\n", i, J_COEFFS[i]);
    fprintf(fp, "\n");

    /* --- NULL DISTRIBUTION --- */
    fprintf(fp, "--- NULL DISTRIBUTION ---\n");
    if (st->null_computed) {
        fprintf(fp, "  Samples:             %d\n", st->null_samples);
        fprintf(fp, "  Random mean coh:     %.6f\n", st->null_mean_coherence);
        fprintf(fp, "  Random std coh:      %.6f\n", st->null_std_coherence);
        fprintf(fp, "  Random Tier-3 rate:  %.6f\n", st->null_transcendental_rate);

        if (st->triplets_analyzed > 0 && st->null_std_coherence > 0) {
            double obs_mean = st->sum_coherence / st->triplets_analyzed;
            double z = (obs_mean - st->null_mean_coherence) / st->null_std_coherence;
            fprintf(fp, "  Observed mean z:     %.2f\n", z);
        }
    } else {
        fprintf(fp, "  (not computed)\n");
    }
    fprintf(fp, "\n");

    /* --- AUTOCORRELATION --- */
    fprintf(fp, "--- AUTOCORRELATION ---\n");
    if (st->n_spacings > 0) {
        double mean_sp = st->sum_spacing / st->n_spacings;
        double var_sp  = st->sum_spacing2 / st->n_spacings - mean_sp * mean_sp;
        double std_sp  = (var_sp > 0) ? sqrt(var_sp) : 0;
        fprintf(fp, "  Mean spacing:  %.2f triplets\n", mean_sp);
        fprintf(fp, "  Std spacing:   %.2f triplets\n", std_sp);
        fprintf(fp, "  Pairs:         %s\n",
                fmt_comma(st->n_spacings, b1, sizeof(b1)));
    } else {
        fprintf(fp, "  (insufficient data)\n");
    }
    fprintf(fp, "\n");

    /* --- MOONSHINE DENSITY (first 10 events) --- */
    fprintf(fp, "--- MOONSHINE DENSITY (first 10 events) ---\n");
    int n_show = (int)st->stored_events;
    if (n_show > 10) n_show = 10;
    for (int i = 0; i < n_show; i++) {
        MoonstoneEvent *ev = &ckpt->events[i];
        char p0[32], p1[32], p2[32], p3[32], ti[32];
        fmt_comma(ev->primes[0], p0, sizeof(p0));
        fmt_comma(ev->primes[1], p1, sizeof(p1));
        fmt_comma(ev->primes[2], p2, sizeof(p2));
        fmt_comma(ev->primes[3], p3, sizeof(p3));
        fmt_comma(ev->triplet_index, ti, sizeof(ti));
        double m_e = sqrt(ev->coherence * 6.0) / DUAL_COXETER_H;
        fprintf(fp, "  #%-3d triplet %-12s  coh=%.4f  M_E=%.4f  byte=0x%02X"
                    "  primes=[%s, %s, %s, %s]\n",
                i + 1, ti, ev->coherence, m_e, ev->byte,
                p0, p1, p2, p3);
    }
    fprintf(fp, "\n");

    /* --- EXCEPTIONAL BYTE SEQUENCE --- */
    fprintf(fp, "--- EXCEPTIONAL BYTE SEQUENCE (first %d bytes) ---\n",
            (int)st->stored_bytes);
    if (st->stored_bytes > 0) {
        /* Hex dump, 16 bytes per line */
        for (int64_t off = 0; off < st->stored_bytes; off += 16) {
            fprintf(fp, "  %04x: ", (int)off);
            for (int j = 0; j < 16; j++) {
                if (off + j < st->stored_bytes)
                    fprintf(fp, "%02X ", ckpt->bytes[off + j]);
                else
                    fprintf(fp, "   ");
            }
            fprintf(fp, " |");
            for (int j = 0; j < 16; j++) {
                if (off + j < st->stored_bytes) {
                    uint8_t c = ckpt->bytes[off + j];
                    fputc((c >= 32 && c < 127) ? c : '.', fp);
                } else {
                    fputc(' ', fp);
                }
            }
            fprintf(fp, "|\n");
        }
    }
    fprintf(fp, "\n");

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  END OF REPORT\n");
    fprintf(fp, "================================================================\n");
}

/* ================================================================
 * Main
 * ================================================================ */

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --max-primes N      Target primes [default: 10000000000]\n");
    fprintf(stderr, "  --checkpoint FILE   Checkpoint file [default: mc_state.ebd]\n");
    fprintf(stderr, "  --resume            Resume from checkpoint\n");
    fprintf(stderr, "  --output FILE       Report file [default: spiral_outputs/mc_report.txt]\n");
    fprintf(stderr, "  --null-samples N    Random triplets for null distribution [default: 10000]\n");
    fprintf(stderr, "  --interval N        Checkpoint interval [default: 100000000]\n");
    fprintf(stderr, "  --help              Show help\n");
}

int main(int argc, char **argv)
{
    int64_t max_primes       = 10000000000LL;
    const char *ckpt_path    = "mc_state.ebd";
    const char *output_path  = "spiral_outputs/mc_report.txt";
    int64_t ckpt_interval    = DEFAULT_CKPT_INT;
    int null_samples         = 10000;
    int do_resume            = 0;

    static struct option long_opts[] = {
        {"max-primes",   required_argument, 0, 'p'},
        {"checkpoint",   required_argument, 0, 'c'},
        {"resume",       no_argument,       0, 'r'},
        {"output",       required_argument, 0, 'o'},
        {"null-samples", required_argument, 0, 'n'},
        {"interval",     required_argument, 0, 'i'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:c:ro:n:i:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'p': max_primes    = atoll(optarg); break;
        case 'c': ckpt_path     = optarg; break;
        case 'r': do_resume     = 1; break;
        case 'o': output_path   = optarg; break;
        case 'n': null_samples  = atoi(optarg); break;
        case 'i': ckpt_interval = atoll(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    char b1[64], b2[64];

    printf("================================================================\n");
    printf("  MONSTROUS CORRELATOR (MC Pass-8 Moonshine)\n");
    printf("================================================================\n");
    printf("Max primes:       %s\n", fmt_comma(max_primes, b1, sizeof(b1)));
    printf("Checkpoint:       %s%s\n", ckpt_path, do_resume ? " (RESUME)" : "");
    printf("Checkpoint every: %s primes\n", fmt_comma(ckpt_interval, b1, sizeof(b1)));
    printf("Output:           %s\n", output_path);
    printf("Null samples:     %d\n", null_samples);
    printf("Threads:          %d\n\n", omp_get_max_threads());

    /* E8 lattice */
    printf("Initializing E8 lattice... ");
    fflush(stdout);
    E8Lattice e8;
    e8_init(&e8);
    printf("done (240 roots, norm = sqrt(2))\n");

    /* Checkpoint */
    printf("Opening checkpoint... ");
    fflush(stdout);
    MCCheckpointFile ckpt;
    if (mc_ckpt_open(&ckpt, ckpt_path, do_resume) < 0)
        return 1;
    ckpt.state->max_primes = max_primes;

    if (do_resume) {
        printf("resumed from %s primes, last_prime=%s\n",
               fmt_comma(ckpt.state->primes_processed, b1, sizeof(b1)),
               fmt_comma(ckpt.state->last_prime, b2, sizeof(b2)));
    } else {
        printf("fresh start\n");
    }

    /* Null distribution (skip on resume if already computed) */
    if (!do_resume || !ckpt.state->null_computed) {
        printf("Computing null distribution (%d random triplets from %s sieved primes)... ",
               null_samples,
               fmt_comma(NULL_SIEVE_SIZE, b1, sizeof(b1)));
        fflush(stdout);
        tic();
        MCNullStats nstats = compute_null_distribution(&e8, null_samples);
        double null_time = toc();

        ckpt.state->null_mean_coherence      = nstats.mean_coherence;
        ckpt.state->null_std_coherence        = nstats.std_coherence;
        ckpt.state->null_transcendental_rate  = nstats.transcendental_rate;
        ckpt.state->null_computed             = 1;
        ckpt.state->null_samples              = null_samples;

        printf("done in %.2f s\n", null_time);
        printf("  Null mean coherence: %.6f\n", nstats.mean_coherence);
        printf("  Null std coherence:  %.6f\n", nstats.std_coherence);
        printf("  Null Tier-3 rate:    %.6f\n\n", nstats.transcendental_rate);
    } else {
        printf("Null distribution loaded from checkpoint: mean=%.6f, std=%.6f\n\n",
               ckpt.state->null_mean_coherence,
               ckpt.state->null_std_coherence);
    }

    /* Streaming sieve */
    printf("Initializing sieve... ");
    fflush(stdout);
    StreamingSieve sieve;
    int64_t resume_after = do_resume ? ckpt.state->last_prime : 0;
    int64_t resume_count = do_resume ? ckpt.state->primes_processed : 0;
    ss_init(&sieve, max_primes, resume_after);
    if (do_resume) sieve.total_produced = resume_count;
    printf("done (base primes: %s, limit: %s)\n\n",
           fmt_comma(sieve.n_base, b1, sizeof(b1)),
           fmt_comma(sieve.limit, b2, sizeof(b2)));

    /* Main processing loop */
    double t_start = omp_get_wtime();
    double saved_elapsed = do_resume ? ckpt.state->elapsed_seconds : 0;
    int64_t prev_prime = do_resume ? ckpt.state->last_prime : 0;
    int64_t last_ckpt_count = resume_count;
    int64_t batch_num = 0;

    printf("Processing...\n");

    while (!g_shutdown) {
        int64_t count = ss_next_batch(&sieve);
        if (count == 0) break;

        process_batch(sieve.batch, count, &prev_prime, &e8, &ckpt);

        ckpt.state->primes_processed = sieve.total_produced;
        ckpt.state->last_prime = prev_prime;
        ckpt.state->elapsed_seconds = saved_elapsed + (omp_get_wtime() - t_start);

        batch_num++;

        /* Progress every 10 batches (~10M primes) */
        if (batch_num % 10 == 0) {
            double elapsed = omp_get_wtime() - t_start;
            double rate = (double)(sieve.total_produced - resume_count) / elapsed;
            double pct = 100.0 * sieve.total_produced / max_primes;
            char pb[64];
            fmt_comma(ckpt.state->triplets_analyzed, pb, sizeof(pb));
            printf("\r  %14s / %s (%.1f%%) | %6.1fM/s | tri %s"
                   " | T3=%ld T2=%ld T1=%ld   ",
                   fmt_comma(sieve.total_produced, b1, sizeof(b1)),
                   fmt_comma(max_primes, b2, sizeof(b2)),
                   pct, rate / 1e6, pb,
                   (long)ckpt.state->tier_count[3],
                   (long)ckpt.state->tier_count[2],
                   (long)ckpt.state->tier_count[1]);
            fflush(stdout);
        }

        /* Checkpoint */
        if (sieve.total_produced - last_ckpt_count >= ckpt_interval) {
            mc_ckpt_sync(&ckpt);
            last_ckpt_count = sieve.total_produced;
            printf("\n  [CHECKPOINT] %s primes saved\n",
                   fmt_comma(sieve.total_produced, b1, sizeof(b1)));
        }
    }

    double total_elapsed = omp_get_wtime() - t_start;
    ckpt.state->elapsed_seconds = saved_elapsed + total_elapsed;
    ckpt.state->primes_processed = sieve.total_produced;
    ckpt.state->last_prime = prev_prime;

    if (g_shutdown) {
        printf("\n\nShutdown — saving checkpoint at %s primes\n",
               fmt_comma(sieve.total_produced, b1, sizeof(b1)));
    }

    mc_ckpt_sync(&ckpt);

    printf("\n\nDone: %s primes in %.2f s (%.2f hrs)\n",
           fmt_comma(sieve.total_produced, b1, sizeof(b1)),
           total_elapsed, total_elapsed / 3600.0);

    /* Write report to file */
    printf("Writing report to %s... ", output_path);
    fflush(stdout);
    FILE *fp = fopen(output_path, "w");
    if (fp) {
        generate_report(fp, &ckpt);
        fclose(fp);
        printf("done\n\n");
    } else {
        printf("FAILED: %s\n\n", strerror(errno));
    }

    /* Also print report to stdout */
    generate_report(stdout, &ckpt);

    mc_ckpt_close(&ckpt);
    ss_destroy(&sieve);
    return 0;
}
