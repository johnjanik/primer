/*
 * srv_verify.c — Spectral Rigidity Verifier (SRV Pass-9)
 *
 * Self-contained empirical verification of three "physical constants of
 * arithmetic" predicted by the E8 Diamond framework:
 *
 *   1. Spectral Variance  Λ_J = Var(g̃)                  → predicted 1/√2 ≈ 0.707106
 *   2. Monstrous Ratio    R_M = Count(g=6)/Count(g=2)    → predicted 52/8 = 6.5
 *   3. Phase-Sync Mandala Ψ = Σ exp(2πi √(g̃)/√2)       → E8 Theta function
 *
 * where g̃ = (p_{n+1} - p_n) / ln(p_n) is the normalized prime gap.
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o srv_verify srv_verify.c -lm
 *
 * Usage:
 *   ./srv_verify --max-primes 100000000000 [--sieve|--file PATH]
 *                [--checkpoint srv_state.ebd] [--resume]
 *                [--output spiral_outputs/srv_report.txt]
 *                [--interval 100000000]
 *
 * Precision:
 *   Welford accumulators and Kahan mandala sums use 80-bit long double
 *   (18-19 significant digits) for stability at 10^11 iterations.
 *
 * No external dependencies beyond standard C + OpenMP + POSIX.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <inttypes.h>

/* ================================================================
 * Inline Utilities (no external header dependency)
 * ================================================================ */

/* Timing */
static double g_timer_start;
#define tic() (g_timer_start = omp_get_wtime())
#define toc() (omp_get_wtime() - g_timer_start)

/* Format int64 with commas: "1,234,567" */
static char *fmt_comma(int64_t n, char *buf, int bufsz)
{
    char raw[32];
    snprintf(raw, sizeof(raw), "%" PRId64, n);
    int len = (int)strlen(raw);
    int commas = (len - (raw[0] == '-' ? 2 : 1)) / 3;
    int total = len + commas;
    if (total >= bufsz) { snprintf(buf, bufsz, "%" PRId64, n); return buf; }
    buf[total] = '\0';
    int src = len - 1, dst = total - 1, grp = 0;
    while (src >= 0) {
        buf[dst--] = raw[src--];
        grp++;
        if (grp == 3 && src >= 0 && raw[src] != '-') { buf[dst--] = ','; grp = 0; }
    }
    return buf;
}

/*
 * Upper bound for the Nth prime.
 *
 * Uses Dusart 2010 (Theorem 6.9) for n >= 688,383:
 *   p_n < n * (ln(n) + ln(ln(n)) - 1 + (ln(ln(n)) - 2) / ln(n))
 *
 * Falls back to Rosser-Schoenfeld for smaller n:
 *   p_n < n * (ln(n) + ln(ln(n)))   for n >= 6
 *
 * Both are rigorous upper bounds (not estimates).
 */
static int64_t prime_upper_bound(int64_t n)
{
    if (n < 6) return 13;
    double ln_n = log((double)n);
    double ln_ln_n = log(ln_n);

    if (n >= 688383) {
        /* Dusart 2010: tight bound, <0.5% over for n > 10^7 */
        double bound = (double)n * (ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.0) / ln_n);
        return (int64_t)bound + 1000;
    }
    /* Rosser-Schoenfeld: slightly looser but safe for smaller n */
    return (int64_t)((double)n * (ln_n + ln_ln_n)) + 1000;
}

/* ================================================================
 * Constants
 * ================================================================ */

#define SRV_MAGIC         0x5352563956ULL   /* "SRV9V" */
#define SRV_VERSION       2                 /* v2: long double accumulators */
#define BATCH_CAPACITY    (1 << 20)         /* 1M primes per batch */
#define SIEVE_SEGMENT     (1 << 19)         /* 512K sieve segment */
#define DEFAULT_CKPT_INT  (100000000LL)     /* checkpoint every 100M primes */
#define GAP_HIST_BINS     64                /* gaps 0,2,4,...,126 */
#define MAX_GAP_TRACKED   126               /* 2 * (GAP_HIST_BINS - 1) */
#define MAX_SNAPSHOTS     1024              /* convergence history */

/* Predicted constants */
#define PREDICTED_VARIANCE  0.70710678118654752440  /* 1/√2 */
#define PREDICTED_RATIO     6.5                     /* dim(F4)/rank(E8) = 52/8 */

/* Long double pi (for phase computation) */
#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

/* ================================================================
 * Signal Handler
 * ================================================================ */

static volatile sig_atomic_t g_shutdown = 0;

static void handle_signal(int sig)
{
    (void)sig;
    g_shutdown = 1;
}

/* ================================================================
 * Convergence Snapshot (stored as double for compact mmap layout)
 * ================================================================ */

typedef struct {
    int64_t primes_at;       /* primes processed at this snapshot */
    double  variance;        /* Var(g̃) at this point */
    double  ratio_g6_g2;     /* Count(g=6)/Count(g=2) */
    double  mandala_x;       /* Re(Ψ) */
    double  mandala_y;       /* Im(Ψ) */
    double  mandala_norm;    /* |Ψ| / √N */
    double  mean_g;          /* mean(g̃) at this point */
    double  _pad;
} ConvergenceSnapshot;

/* ================================================================
 * Checkpoint State (mmap'd)
 *
 * Welford and Kahan accumulators use long double (80-bit extended
 * precision on x86-64, stored as 16 bytes) for stability at 10^11
 * iterations. This gives 18-19 significant digits vs 15-16 for
 * IEEE 754 double.
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
    int64_t last_prime;         /* last prime seen, for gap computation on resume */

    /* Gap counts (specific gaps of interest) */
    int64_t count_g2;           /* twin primes (gap=2) */
    int64_t count_g4;           /* cousin primes (gap=4) */
    int64_t count_g6;           /* sexy primes (gap=6) */
    int64_t count_g8;
    int64_t count_g10;
    int64_t count_g12;
    int64_t count_g14;
    int64_t count_g18;
    int64_t count_g20;
    int64_t count_g30;

    /* Full gap histogram: bin i = count of gaps equal to 2*i */
    int64_t gap_hist[GAP_HIST_BINS];
    int64_t gap_overflow;       /* gaps > MAX_GAP_TRACKED */
    int64_t max_gap_seen;       /* largest gap observed */

    /* Welford's online algorithm — long double for 10^11 stability */
    long double welford_mean;   /* running mean of g̃ */
    long double welford_M2;     /* Σ(g̃ - mean)² running sum */

    /* Phase-sync mandala — Kahan summation in long double */
    long double mandala_x;      /* Re(Ψ) = Σ cos(2π·phase) */
    long double mandala_y;      /* Im(Ψ) = Σ sin(2π·phase) */
    long double kahan_cx;       /* Kahan compensator for x */
    long double kahan_cy;       /* Kahan compensator for y */

    /* Convergence tracking */
    int32_t n_snapshots;
    int32_t _pad1;

    /* Timing */
    double elapsed_seconds;
    double elapsed_at_resume;

    char _pad_future[64];       /* future expansion */
} SRVCheckpointState;

/* File layout: [SRVCheckpointState][ConvergenceSnapshot × MAX_SNAPSHOTS] */
#define CKPT_SNAP_OFF   ((int64_t)sizeof(SRVCheckpointState))
#define CKPT_FILE_SIZE  ((CKPT_SNAP_OFF + (int64_t)(MAX_SNAPSHOTS * sizeof(ConvergenceSnapshot)) + 4095) & ~4095)

typedef struct {
    int                  fd;
    void                *map;
    SRVCheckpointState  *state;
    ConvergenceSnapshot *snaps;
    char                 path[512];
} SRVCheckpointFile;

/* ================================================================
 * Checkpoint I/O (mmap'd, MAP_SHARED + msync)
 * ================================================================ */

static int srv_ckpt_open(SRVCheckpointFile *ckpt, const char *path, int resume)
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

    ckpt->state = (SRVCheckpointState *)ckpt->map;
    ckpt->snaps = (ConvergenceSnapshot *)((uint8_t *)ckpt->map + CKPT_SNAP_OFF);

    if (resume) {
        if (ckpt->state->magic != SRV_MAGIC || ckpt->state->version != SRV_VERSION) {
            fprintf(stderr, "Invalid checkpoint (bad magic/version, expected v%d)\n",
                    SRV_VERSION);
            munmap(ckpt->map, CKPT_FILE_SIZE);
            close(ckpt->fd);
            return -1;
        }
    } else {
        memset(ckpt->map, 0, CKPT_FILE_SIZE);
        ckpt->state->magic   = SRV_MAGIC;
        ckpt->state->version = SRV_VERSION;
    }
    return 0;
}

static void srv_ckpt_sync(SRVCheckpointFile *ckpt)
{
    msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
}

static void srv_ckpt_close(SRVCheckpointFile *ckpt)
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
 *
 * Correctly handles partial segment consumption at batch boundaries
 * to prevent prime-skipping gaps.
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
    /* Partial segment tracking — prevents prime skipping at batch boundaries */
    int64_t  seg_lo;            /* lo of current segment */
    int64_t  seg_len;           /* length of current segment */
    int64_t  seg_pos;           /* scan position within segment */
    int      seg_valid;         /* 1 = segment sieved but not fully consumed */
} SRVSieve;

static void ss_init(SRVSieve *sv, int64_t max_primes, int64_t resume_after)
{
    sv->max_primes = max_primes;
    sv->total_produced = 0;
    sv->batch_count = 0;
    sv->seg_valid = 0;
    sv->seg_pos = 0;
    sv->seg_lo = 0;
    sv->seg_len = 0;
    sv->limit = prime_upper_bound(max_primes);
    /* Small safety margin on top of Dusart bound (already tight) */
    sv->limit += sv->limit / 100 + 100000;

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

static int64_t ss_next_batch(SRVSieve *sv)
{
    sv->batch_count = 0;

    while (sv->batch_count < BATCH_CAPACITY &&
           sv->total_produced < sv->max_primes)
    {
        /* If no partially consumed segment, sieve the next one */
        if (!sv->seg_valid) {
            if (sv->current_lo > sv->limit) break;

            int64_t lo = sv->current_lo;
            int64_t hi = lo + SIEVE_SEGMENT - 1;
            if (hi > sv->limit) hi = sv->limit;

            sv->seg_lo  = lo;
            sv->seg_len = hi - lo + 1;
            sv->seg_pos = 0;

            memset(sv->segment, 0, (size_t)sv->seg_len);

            #pragma omp parallel for schedule(dynamic, 64)
            for (int64_t b = 0; b < sv->n_base; b++) {
                int64_t p = sv->base_primes[b];
                int64_t start = ((lo + p - 1) / p) * p;
                if (start < p * p) start = p * p;
                if (start > hi) continue;
                for (int64_t j = start; j <= hi; j += p)
                    sv->segment[j - lo] = 1;
            }

            sv->seg_valid = 1;
        }

        /* Collect primes from current segment (resume from seg_pos) */
        while (sv->seg_pos < sv->seg_len &&
               sv->batch_count < BATCH_CAPACITY &&
               sv->total_produced < sv->max_primes)
        {
            if (!sv->segment[sv->seg_pos]) {
                sv->batch[sv->batch_count++] = sv->seg_lo + sv->seg_pos;
                sv->total_produced++;
            }
            sv->seg_pos++;
        }

        /* If segment fully consumed, advance to next */
        if (sv->seg_pos >= sv->seg_len) {
            sv->current_lo = sv->seg_lo + sv->seg_len;
            sv->seg_valid = 0;
        }
        /* Otherwise seg_valid stays 1, seg_pos remembers where we stopped */
    }
    return sv->batch_count;
}

static void ss_destroy(SRVSieve *sv)
{
    free(sv->base_primes);
    free(sv->segment);
    free(sv->batch);
}

/* ================================================================
 * Gap Processing: Welford + Kahan + Gap Counting
 *
 * All accumulations use long double (80-bit extended precision on
 * x86-64) for stability at 10^11 iterations. Welford's algorithm
 * is inherently stable, but long double provides ~3 extra decimal
 * digits of headroom for the Kahan compensated sum.
 *
 * Precision budget at N = 10^11:
 *   double  (ε ≈ 1.1e-16): Kahan error O(Nε²) ≈ 10^{-21} — fine
 *   long double (ε ≈ 5.4e-20): Kahan error O(Nε²) ≈ 10^{-29} — overkill
 *   Both are adequate; long double provides defense-in-depth.
 * ================================================================ */

static inline void process_gap(SRVCheckpointState *st, int64_t p_curr, int64_t gap)
{
    st->gaps_analyzed++;

    /* 1. Gap counting */
    switch (gap) {
        case 2:  st->count_g2++;  break;
        case 4:  st->count_g4++;  break;
        case 6:  st->count_g6++;  break;
        case 8:  st->count_g8++;  break;
        case 10: st->count_g10++; break;
        case 12: st->count_g12++; break;
        case 14: st->count_g14++; break;
        case 18: st->count_g18++; break;
        case 20: st->count_g20++; break;
        case 30: st->count_g30++; break;
    }

    /* Gap histogram (even gaps: bin = gap/2) */
    if (gap <= MAX_GAP_TRACKED && gap >= 0) {
        st->gap_hist[gap / 2]++;
    } else {
        st->gap_overflow++;
    }
    if (gap > st->max_gap_seen) st->max_gap_seen = gap;

    /* 2. Normalized gap (long double precision) */
    long double tilde_g = (long double)gap / logl((long double)p_curr);

    /* 3. Welford's online algorithm for variance (long double) */
    int64_t n = st->gaps_analyzed;   /* already incremented */
    long double delta = tilde_g - st->welford_mean;
    st->welford_mean += delta / (long double)n;
    long double delta2 = tilde_g - st->welford_mean;
    st->welford_M2 += delta * delta2;

    /* 4. Phase-sync mandala (Kahan summation in long double) */
    long double phase = fmodl(sqrtl(tilde_g) / sqrtl(2.0L), 1.0L);
    if (phase < 0.0L) phase += 1.0L;
    long double theta = 2.0L * M_PIl * phase;

    /* Kahan sum for cos */
    {
        long double y = cosl(theta) - st->kahan_cx;
        long double t = st->mandala_x + y;
        st->kahan_cx = (t - st->mandala_x) - y;
        st->mandala_x = t;
    }

    /* Kahan sum for sin */
    {
        long double y = sinl(theta) - st->kahan_cy;
        long double t = st->mandala_y + y;
        st->kahan_cy = (t - st->mandala_y) - y;
        st->mandala_y = t;
    }
}

/* ================================================================
 * Process a batch of primes from the sieve
 *
 * The first prime in a batch might need to be paired with the
 * last prime from the previous batch (stored in st->last_prime).
 * ================================================================ */

static void process_batch(SRVCheckpointState *st, const int64_t *primes, int64_t count)
{
    int64_t start = 0;

    /* If we have a carry-over prime from the previous batch, use it */
    if (st->last_prime > 0 && count > 0) {
        int64_t gap = primes[0] - st->last_prime;
        process_gap(st, st->last_prime, gap);
        st->primes_processed++;
        start = 0;  /* primes[0] is already "consumed" as p_next */
    } else if (st->last_prime == 0 && count > 0) {
        /* First prime ever — just record it, no gap to compute */
        st->primes_processed++;
        start = 1;
        st->last_prime = primes[0];
    }

    /* Process consecutive pairs within this batch */
    for (int64_t i = (start == 0 ? 1 : start); i < count; i++) {
        int64_t gap = primes[i] - primes[i - 1];
        process_gap(st, primes[i - 1], gap);
        st->primes_processed++;
    }

    /* Store the last prime for the next batch */
    if (count > 0) {
        st->last_prime = primes[count - 1];
    }
}

/* ================================================================
 * Convergence Snapshot
 * ================================================================ */

static void take_snapshot(SRVCheckpointFile *ckpt)
{
    SRVCheckpointState *st = ckpt->state;
    if (st->n_snapshots >= MAX_SNAPSHOTS) return;

    int64_t n = st->gaps_analyzed;
    if (n < 1) return;

    /* Avoid duplicate snapshot at same prime count */
    if (st->n_snapshots > 0 &&
        ckpt->snaps[st->n_snapshots - 1].primes_at == st->primes_processed)
        return;

    ConvergenceSnapshot *snap = &ckpt->snaps[st->n_snapshots];

    snap->primes_at    = st->primes_processed;
    /* Cast long double accumulators to double for snapshot storage */
    snap->variance     = (double)(st->welford_M2 / (long double)n);
    snap->ratio_g6_g2  = (st->count_g2 > 0) ? (double)st->count_g6 / (double)st->count_g2 : 0.0;
    snap->mandala_x    = (double)st->mandala_x;
    snap->mandala_y    = (double)st->mandala_y;
    snap->mandala_norm = (double)(sqrtl(st->mandala_x * st->mandala_x +
                                        st->mandala_y * st->mandala_y) /
                                  sqrtl((long double)n));
    snap->mean_g       = (double)st->welford_mean;

    st->n_snapshots++;
}

/* ================================================================
 * Report Generation
 * ================================================================ */

static void generate_report(SRVCheckpointFile *ckpt, const char *outpath,
                            double elapsed)
{
    SRVCheckpointState *st = ckpt->state;
    char b1[32], b2[32], b3[32], b4[32];

    FILE *fp = fopen(outpath, "w");
    if (!fp) { perror("Cannot open output file"); return; }

    /* Also write to stdout */
    FILE *out[2] = { fp, stdout };

    for (int f = 0; f < 2; f++) {
        FILE *o = out[f];
        fprintf(o, "================================================================\n");
        fprintf(o, "  SPECTRAL RIGIDITY VERIFIER (SRV Pass-9)\n");
        fprintf(o, "================================================================\n");
        fprintf(o, "Primes processed:    %s / %s\n",
                fmt_comma(st->primes_processed, b1, sizeof(b1)),
                fmt_comma(st->max_primes, b2, sizeof(b2)));
        fprintf(o, "Gaps analyzed:       %s\n",
                fmt_comma(st->gaps_analyzed, b3, sizeof(b3)));
        fprintf(o, "Last prime:          %s\n",
                fmt_comma(st->last_prime, b4, sizeof(b4)));
        fprintf(o, "Elapsed:             %.2f s (%.2f hrs)\n",
                elapsed, elapsed / 3600.0);
        fprintf(o, "Precision:           long double (80-bit, ~19 sig. digits)\n");
        fprintf(o, "\n");

        /* === IDENTITY 1: Spectral Variance === */
        long double n = (long double)st->gaps_analyzed;
        long double variance = (n > 0) ? st->welford_M2 / n : 0.0L;
        double var_d = (double)variance;
        double var_err = var_d - PREDICTED_VARIANCE;

        fprintf(o, "--- IDENTITY 1: SPECTRAL VARIANCE (%cJ) ---\n", 0xCE);  /* Λ */
        fprintf(o, "  Predicted:      1/sqrt(2) = %.15f\n", PREDICTED_VARIANCE);
        fprintf(o, "  Observed:       %.15Lf\n", variance);
        fprintf(o, "  Deviation:      %+.15f (%+.8f%%)\n",
                var_err, (PREDICTED_VARIANCE > 0) ? 100.0 * var_err / PREDICTED_VARIANCE : 0.0);
        fprintf(o, "  Mean(g~):       %.15Lf  (predicted: 1.0)\n", st->welford_mean);
        fprintf(o, "  Status:         %s\n",
                fabs(var_err) < 0.001 ? "VERIFIED" :
                fabs(var_err) < 0.01  ? "CONVERGING" : "DIVERGENT");
        fprintf(o, "\n");

        /* === IDENTITY 2: Monstrous Ratio === */
        double ratio = (st->count_g2 > 0) ? (double)st->count_g6 / (double)st->count_g2 : 0.0;
        double ratio_err = ratio - PREDICTED_RATIO;

        fprintf(o, "--- IDENTITY 2: MONSTROUS RATIO (R_M) ---\n");
        fprintf(o, "  Predicted:      dim(F4)/rank(E8) = 52/8 = %.1f\n", PREDICTED_RATIO);
        fprintf(o, "  Observed:       %.15f\n", ratio);
        fprintf(o, "  Count(g=2):     %s  (twin primes)\n",
                fmt_comma(st->count_g2, b1, sizeof(b1)));
        fprintf(o, "  Count(g=6):     %s  (sexy primes)\n",
                fmt_comma(st->count_g6, b2, sizeof(b2)));
        fprintf(o, "  Deviation:      %+.15f (%+.8f%%)\n",
                ratio_err, (PREDICTED_RATIO > 0) ? 100.0 * ratio_err / PREDICTED_RATIO : 0.0);
        fprintf(o, "  Status:         %s\n",
                fabs(ratio_err) < 0.01 ? "VERIFIED" :
                fabs(ratio_err) < 0.1  ? "CONVERGING" : "DIVERGENT");
        fprintf(o, "\n");

        /* === IDENTITY 3: Phase-Sync Mandala === */
        long double norm_psi = sqrtl(st->mandala_x * st->mandala_x +
                                      st->mandala_y * st->mandala_y);
        long double rw_expected = sqrtl(n);
        double norm_ratio = (double)((rw_expected > 0) ? norm_psi / rw_expected : 0.0L);

        fprintf(o, "--- IDENTITY 3: PHASE-SYNC MANDALA (Psi) ---\n");
        fprintf(o, "  Psi = Sum exp(2*pi*i * sqrt(g~)/sqrt(2))\n");
        fprintf(o, "  Re(Psi):        %+.6Lf\n", st->mandala_x);
        fprintf(o, "  Im(Psi):        %+.6Lf\n", st->mandala_y);
        fprintf(o, "  |Psi|:          %.6Lf\n", norm_psi);
        fprintf(o, "  |Psi|/sqrt(N):  %.6f  (random walk ~ 1.0)\n", norm_ratio);
        fprintf(o, "  arg(Psi):       %.6Lf rad (%.2Lf deg)\n",
                atan2l(st->mandala_y, st->mandala_x),
                atan2l(st->mandala_y, st->mandala_x) * 180.0L / M_PIl);
        fprintf(o, "  Status:         %s\n",
                norm_ratio < 0.5 ? "COHERENT (sub-random-walk)" :
                norm_ratio < 2.0 ? "DIFFUSIVE (random-walk)" : "SUPER-DIFFUSIVE");
        fprintf(o, "\n");

        /* === GAP DISTRIBUTION === */
        fprintf(o, "--- GAP DISTRIBUTION ---\n");
        int64_t total_gaps = st->gaps_analyzed;
        struct { int gap; const char *name; } gap_names[] = {
            {2,  "twin"}, {4,  "cousin"}, {6,  "sexy"},
            {8,  ""}, {10, ""}, {12, ""}, {14, ""},
            {18, ""}, {20, ""}, {30, ""},
        };
        fprintf(o, "  %-8s %-20s %-10s\n", "Gap", "Count", "Fraction");
        /* gap=1 (only 2->3) */
        if (st->gap_hist[0] > 0) {
            fprintf(o, "  %-8d %-20s %.6f%%  (2->3)\n", 1,
                    fmt_comma(st->gap_hist[0], b1, sizeof(b1)),
                    100.0 * (double)st->gap_hist[0] / (double)total_gaps);
        }
        for (int i = 0; i < (int)(sizeof(gap_names)/sizeof(gap_names[0])); i++) {
            int g = gap_names[i].gap;
            int64_t cnt = st->gap_hist[g / 2];
            if (cnt == 0) continue;
            fprintf(o, "  %-8d %-20s %.6f%%%s%s%s\n", g,
                    fmt_comma(cnt, b1, sizeof(b1)),
                    100.0 * (double)cnt / (double)total_gaps,
                    gap_names[i].name[0] ? "  (" : "",
                    gap_names[i].name,
                    gap_names[i].name[0] ? ")" : "");
        }
        if (st->gap_overflow > 0) {
            fprintf(o, "  >%-7d %-20s %.6f%%\n", MAX_GAP_TRACKED,
                    fmt_comma(st->gap_overflow, b1, sizeof(b1)),
                    100.0 * (double)st->gap_overflow / (double)total_gaps);
        }
        fprintf(o, "  Max gap seen:   %s\n",
                fmt_comma(st->max_gap_seen, b1, sizeof(b1)));
        fprintf(o, "\n");

        /* === GAP RATIOS === */
        fprintf(o, "--- GAP RATIOS ---\n");
        if (st->count_g2 > 0) {
            fprintf(o, "  g6/g2:   %.15f  (predicted: 6.5)\n",
                    (double)st->count_g6 / (double)st->count_g2);
            fprintf(o, "  g4/g2:   %.15f\n",
                    (double)st->count_g4 / (double)st->count_g2);
            fprintf(o, "  g8/g2:   %.15f\n",
                    (double)st->count_g8 / (double)st->count_g2);
            fprintf(o, "  g12/g2:  %.15f\n",
                    (double)st->count_g12 / (double)st->count_g2);
            fprintf(o, "  g30/g2:  %.15f\n",
                    (double)st->count_g30 / (double)st->count_g2);
        }
        fprintf(o, "\n");

        /* === CONVERGENCE HISTORY === */
        fprintf(o, "--- CONVERGENCE HISTORY ---\n");
        fprintf(o, "  Snapshots: %d\n", st->n_snapshots);
        if (st->n_snapshots > 0) {
            fprintf(o, "  %-20s %-16s %-16s %-16s\n",
                    "Primes", "Var(g~)", "g6/g2", "|Psi|/sqrt(N)");
            int step = (st->n_snapshots > 20) ? st->n_snapshots / 20 : 1;
            for (int i = 0; i < st->n_snapshots; i += step) {
                ConvergenceSnapshot *s = &ckpt->snaps[i];
                fprintf(o, "  %-20s %-16.10f %-16.10f %-16.8f\n",
                        fmt_comma(s->primes_at, b1, sizeof(b1)),
                        s->variance, s->ratio_g6_g2, s->mandala_norm);
            }
            /* Always show last */
            if ((st->n_snapshots - 1) % step != 0) {
                ConvergenceSnapshot *s = &ckpt->snaps[st->n_snapshots - 1];
                fprintf(o, "  %-20s %-16.10f %-16.10f %-16.8f\n",
                        fmt_comma(s->primes_at, b1, sizeof(b1)),
                        s->variance, s->ratio_g6_g2, s->mandala_norm);
            }
        }
        fprintf(o, "\n");

        /* === SUMMARY VERDICT === */
        fprintf(o, "--- SUMMARY ---\n");
        fprintf(o, "  Var(g~)     = %.15Lf  (predicted 1/sqrt(2) = %.15f, err = %+.2e)\n",
                variance, PREDICTED_VARIANCE, var_err);
        fprintf(o, "  g6/g2       = %.15f  (predicted 52/8 = %.1f,       err = %+.2e)\n",
                ratio, PREDICTED_RATIO, ratio_err);
        fprintf(o, "  |Psi|/sN    = %.15f  (random walk ~ 1.0)\n", norm_ratio);
        fprintf(o, "\n");

        fprintf(o, "================================================================\n");
        fprintf(o, "  END OF REPORT\n");
        fprintf(o, "================================================================\n");
    }

    fclose(fp);
}

/* ================================================================
 * Convergence CSV Export
 * ================================================================ */

static void export_convergence_csv(SRVCheckpointFile *ckpt, const char *base_path)
{
    SRVCheckpointState *st = ckpt->state;
    if (st->n_snapshots == 0) return;

    /* Derive CSV path from report path */
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s", base_path);
    char *dot = strrchr(csv_path, '.');
    if (dot) *dot = '\0';
    strncat(csv_path, "_convergence.csv", sizeof(csv_path) - strlen(csv_path) - 1);

    FILE *fp = fopen(csv_path, "w");
    if (!fp) return;

    fprintf(fp, "primes,variance,ratio_g6_g2,mandala_x,mandala_y,mandala_norm_per_sqrtN,mean_g\n");
    for (int i = 0; i < st->n_snapshots; i++) {
        ConvergenceSnapshot *s = &ckpt->snaps[i];
        fprintf(fp, "%" PRId64 ",%.15f,%.15f,%.6f,%.6f,%.15f,%.15f\n",
                s->primes_at, s->variance, s->ratio_g6_g2,
                s->mandala_x, s->mandala_y, s->mandala_norm, s->mean_g);
    }
    fclose(fp);
}

/* ================================================================
 * File Mode: Read primes from text file
 * ================================================================ */

static int64_t read_primes_from_file(const char *path, int64_t *batch,
                                     int64_t max_batch, FILE **fp_state,
                                     int *first_call)
{
    if (*first_call) {
        *fp_state = fopen(path, "r");
        if (!*fp_state) {
            perror("Cannot open prime file");
            return -1;
        }
        *first_call = 0;
    }
    if (!*fp_state) return 0;

    int64_t count = 0;
    int c;

    while (count < max_batch && (c = fgetc(*fp_state)) != EOF) {
        if (isdigit(c)) {
            ungetc(c, *fp_state);
            uint64_t val;
            if (fscanf(*fp_state, "%" SCNu64, &val) == 1) {
                if (val >= 2) {
                    batch[count++] = (int64_t)val;
                }
            }
        }
    }

    if (count == 0) {
        fclose(*fp_state);
        *fp_state = NULL;
    }

    return count;
}

/* ================================================================
 * Main
 * ================================================================ */

static void print_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nSpectral Rigidity Verifier — verify E8 arithmetic constants\n\n");
    printf("Options:\n");
    printf("  --max-primes N    Number of primes to process (default: 10B)\n");
    printf("  --sieve           Generate primes internally (default)\n");
    printf("  --file PATH       Read primes from text file\n");
    printf("  --checkpoint FILE Checkpoint file (default: srv_state.ebd)\n");
    printf("  --resume          Resume from checkpoint\n");
    printf("  --output FILE     Report output file (default: spiral_outputs/srv_report.txt)\n");
    printf("  --interval N      Checkpoint/progress interval (default: 100M)\n");
    printf("  --help            Show this help\n");
}

int main(int argc, char *argv[])
{
    /* Defaults */
    int64_t max_primes    = 10000000000LL;  /* 10B */
    const char *file_path = NULL;
    const char *ckpt_path = "srv_state.ebd";
    const char *out_path  = "spiral_outputs/srv_report.txt";
    int64_t interval      = DEFAULT_CKPT_INT;
    int resume            = 0;
    int use_file          = 0;

    /* CLI parsing */
    static struct option long_opts[] = {
        {"max-primes",  required_argument, 0, 'n'},
        {"sieve",       no_argument,       0, 's'},
        {"file",        required_argument, 0, 'f'},
        {"checkpoint",  required_argument, 0, 'c'},
        {"resume",      no_argument,       0, 'r'},
        {"output",      required_argument, 0, 'o'},
        {"interval",    required_argument, 0, 'i'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:sf:c:ro:i:h", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'n': max_primes = strtoll(optarg, NULL, 10); break;
            case 's': use_file = 0; break;
            case 'f': file_path = optarg; use_file = 1; break;
            case 'c': ckpt_path = optarg; break;
            case 'r': resume = 1; break;
            case 'o': out_path = optarg; break;
            case 'i': interval = strtoll(optarg, NULL, 10); break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    /* Signal handling */
    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    /* Print config */
    char b1[32], b2[32];
    printf("================================================================\n");
    printf("  SPECTRAL RIGIDITY VERIFIER (SRV Pass-9)\n");
    printf("================================================================\n");
    printf("Max primes:       %s\n", fmt_comma(max_primes, b1, sizeof(b1)));
    printf("Checkpoint:       %s\n", ckpt_path);
    printf("Checkpoint every: %s primes\n", fmt_comma(interval, b2, sizeof(b2)));
    printf("Output:           %s\n", out_path);
    printf("Mode:             %s\n", use_file ? "FILE" : "SIEVE");
    if (use_file) printf("File:             %s\n", file_path);
    printf("Precision:        long double (%zu bytes, ~%d sig. digits)\n",
           sizeof(long double), (int)(sizeof(long double) >= 16 ? 18 : 15));
    printf("Threads:          %d\n", omp_get_max_threads());
    printf("\n");

    /* Open checkpoint */
    SRVCheckpointFile ckpt;
    if (srv_ckpt_open(&ckpt, ckpt_path, resume) < 0) return 1;

    SRVCheckpointState *st = ckpt.state;

    if (resume) {
        printf("Resuming from %s primes\n",
               fmt_comma(st->primes_processed, b1, sizeof(b1)));
        st->elapsed_at_resume = st->elapsed_seconds;
    }
    st->max_primes = max_primes;

    /* Ensure output directory exists */
    {
        char dir[512];
        snprintf(dir, sizeof(dir), "%s", out_path);
        char *slash = strrchr(dir, '/');
        if (slash) {
            *slash = '\0';
            mkdir(dir, 0755);
        }
    }

    tic();

    /* ============ SIEVE MODE ============ */
    if (!use_file) {
        printf("Initializing sieve... ");
        fflush(stdout);

        SRVSieve sv;
        int64_t resume_after = resume ? st->last_prime : 0;
        ss_init(&sv, max_primes, resume_after);

        /* If resuming, skip primes we've already processed */
        if (resume && st->primes_processed > 0) {
            printf("skipping %s already-processed primes... ",
                   fmt_comma(st->primes_processed, b1, sizeof(b1)));
            fflush(stdout);
            int64_t to_skip = st->primes_processed;
            while (to_skip > 0) {
                int64_t got = ss_next_batch(&sv);
                if (got <= 0) break;
                if (got <= to_skip) {
                    to_skip -= got;
                } else {
                    /* Partial batch — process the remainder */
                    int64_t start = to_skip;
                    for (int64_t i = start; i < got; i++) {
                        if (i == start && st->last_prime > 0) {
                            int64_t gap = sv.batch[i] - st->last_prime;
                            process_gap(st, st->last_prime, gap);
                            st->primes_processed++;
                        } else if (i > start) {
                            int64_t gap = sv.batch[i] - sv.batch[i - 1];
                            process_gap(st, sv.batch[i - 1], gap);
                            st->primes_processed++;
                        }
                    }
                    if (got > start)
                        st->last_prime = sv.batch[got - 1];
                    to_skip = 0;
                }
            }
        }

        printf("done (base primes: %s, limit: %s)\n",
               fmt_comma(sv.n_base, b1, sizeof(b1)),
               fmt_comma(sv.limit, b2, sizeof(b2)));
        printf("\nProcessing...\n");

        int64_t next_checkpoint = st->primes_processed + interval;

        while (!g_shutdown && st->primes_processed < max_primes) {
            int64_t got = ss_next_batch(&sv);
            if (got <= 0) break;

            process_batch(st, sv.batch, got);

            /* Progress + checkpoint */
            if (st->primes_processed >= next_checkpoint ||
                st->primes_processed >= max_primes)
            {
                double elapsed = toc() + st->elapsed_at_resume;
                st->elapsed_seconds = elapsed;
                double rate = (elapsed > 0.0) ?
                    (double)st->primes_processed / elapsed / 1e6 : 0.0;

                double var_now = (double)(st->welford_M2 / (long double)st->gaps_analyzed);
                double ratio_now = (st->count_g2 > 0) ?
                    (double)st->count_g6 / (double)st->count_g2 : 0.0;

                printf("  %13s / %-13s (%4.1f%%) | %6.1fM/s | Var=%.10f | g6/g2=%.8f\n",
                       fmt_comma(st->primes_processed, b1, sizeof(b1)),
                       fmt_comma(max_primes, b2, sizeof(b2)),
                       100.0 * (double)st->primes_processed / (double)max_primes,
                       rate, var_now, ratio_now);
                fflush(stdout);

                take_snapshot(&ckpt);

                printf("  [CHECKPOINT] %s primes saved\n",
                       fmt_comma(st->primes_processed, b1, sizeof(b1)));
                srv_ckpt_sync(&ckpt);

                next_checkpoint = st->primes_processed + interval;
            }
        }

        ss_destroy(&sv);
    }
    /* ============ FILE MODE ============ */
    else {
        printf("Reading primes from %s...\n\n", file_path);

        int64_t *batch = (int64_t *)malloc(BATCH_CAPACITY * sizeof(int64_t));
        FILE *fp_file = NULL;
        int first_call = 1;

        int64_t next_checkpoint = st->primes_processed + interval;

        while (!g_shutdown && st->primes_processed < max_primes) {
            int64_t got = read_primes_from_file(file_path, batch,
                              BATCH_CAPACITY, &fp_file, &first_call);
            if (got <= 0) break;

            if (st->primes_processed + got > max_primes)
                got = max_primes - st->primes_processed;

            process_batch(st, batch, got);

            if (st->primes_processed >= next_checkpoint ||
                st->primes_processed >= max_primes)
            {
                double elapsed = toc() + st->elapsed_at_resume;
                st->elapsed_seconds = elapsed;
                double rate = (elapsed > 0.0) ?
                    (double)st->primes_processed / elapsed / 1e6 : 0.0;

                double var_now = (double)(st->welford_M2 / (long double)st->gaps_analyzed);
                double ratio_now = (st->count_g2 > 0) ?
                    (double)st->count_g6 / (double)st->count_g2 : 0.0;

                printf("  %13s / %-13s (%4.1f%%) | %6.1fM/s | Var=%.10f | g6/g2=%.8f\n",
                       fmt_comma(st->primes_processed, b1, sizeof(b1)),
                       fmt_comma(max_primes, b2, sizeof(b2)),
                       100.0 * (double)st->primes_processed / (double)max_primes,
                       rate, var_now, ratio_now);

                take_snapshot(&ckpt);
                srv_ckpt_sync(&ckpt);

                next_checkpoint = st->primes_processed + interval;
            }
        }

        free(batch);
        if (fp_file) fclose(fp_file);
    }

    /* Final snapshot and save */
    double final_elapsed = toc() + st->elapsed_at_resume;
    st->elapsed_seconds = final_elapsed;
    take_snapshot(&ckpt);
    srv_ckpt_sync(&ckpt);

    if (g_shutdown) {
        printf("\n  [SHUTDOWN] Interrupted — checkpoint saved.\n");
    }

    printf("\nDone. Generating report...\n");
    generate_report(&ckpt, out_path, final_elapsed);
    export_convergence_csv(&ckpt, out_path);
    printf("Report: %s\n", out_path);

    srv_ckpt_close(&ckpt);
    return 0;
}
