/*
 * monstrous_governor.c — Monstrous Governor Scan (MGS Pass-10)
 *
 * Empirical search for the Monster Group's spectral signature in prime gaps.
 *
 * Four analysis modules:
 *   1. Resonance Detector  — Goertzel DFT at f = k/196883 (k=1,2,3) + null
 *   2. Variance Windowing  — Sliding window of 196,883 gaps for local Var(g̃)
 *   3. Residual Mapping    — Hardy-Littlewood residuals ε_d(N) for d=2,4,6,...
 *   4. j-Coefficient Sync  — Pearson correlation of ε with ln(c_n)
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o monstrous_governor monstrous_governor.c -lm
 *
 * Usage:
 *   ./monstrous_governor --max-primes 100000000000
 *       [--checkpoint mgs_state.ebd] [--resume]
 *       [--output spiral_outputs/mgs_report.txt]
 *       [--interval 100000000]
 *
 * Self-contained.  No external dependencies beyond standard C + OpenMP + POSIX.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
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
 * Constants
 * ================================================================ */

#define MONSTER_DIM       196883   /* smallest non-trivial rep of M */
#define SIEVE_SEGMENT     (1 << 19)  /* 512 KB */
#define BATCH_CAPACITY    1000000
#define MAX_RESONANCES    512
#define MAX_SNAPSHOTS     200
#define N_HARMONICS       4        /* harmonics 1,2,3 of 196883 + null at 100000 */
#define N_GAP_BINS        32       /* gap counts for d=2,4,...,64 */
#define J_COEFFS_COUNT    30

#define CKPT_MAGIC        0x4D47533130ULL  /* "MGS10" */
#define CKPT_VERSION      1

/* Twin prime constant C2 = ∏_{p>2} (1 - 1/(p-1)^2) */
#define TWIN_PRIME_CONST  0.6601618158468695739278121L

#ifndef M_PIl
#define M_PIl 3.141592653589793238462643383279502884L
#endif

/* ================================================================
 * Inline Utilities
 * ================================================================ */

static double g_timer_start;
#define tic() (g_timer_start = omp_get_wtime())
#define toc() (omp_get_wtime() - g_timer_start)

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

static int64_t prime_upper_bound(int64_t n)
{
    if (n < 6) return 13;
    double ln_n = log((double)n);
    double ln_ln_n = log(ln_n);
    if (n >= 688383) {
        double bound = (double)n * (ln_n + ln_ln_n - 1.0 + (ln_ln_n - 2.0) / ln_n);
        return (int64_t)bound + 1000;
    }
    return (int64_t)((double)n * (ln_n + ln_ln_n)) + 1000;
}

/* ================================================================
 * j-Function Coefficients (OEIS A000521)
 *
 * j(q) = 1/q + 744 + Σ c_n q^n
 * c_1 = 196884, c_2 = 21493760, ...
 * ================================================================ */

static const double J_COEFFS[J_COEFFS_COUNT] = {
    196884.0,
    21493760.0,
    864299970.0,
    20245856256.0,
    333202640600.0,
    4252023300096.0,
    44656994071935.0,
    401490886656000.0,
    3176440229784420.0,
    22567393309593600.0,
    146211911499519294.0,
    874313719685775360.0,
    4872010111798142520.0,
    25497827389410525184.0,
    126142916465781843075.0,
    593121772421445689200.0,
    2662842413150775245160.0,
    11459912788444786513920.0,
    47438786801234168813250.0,
    189449976248893390028800.0,
    731811377318137519245696.0,
    2740630712513624654929920.0,
    9971041659937182693533820.0,
    35307453186561427099877376.0,
    121883284330422510433351500.0,
    410789960190307909157638144.0,
    1352597709798498253498010650.0,
    4356223736027498896502093824.0,
    13734004999004684553750085280.0,
    42409000721313489345498834944.0
};

/* Asymptotic: c_n ~ exp(4π√n) / (√2 · n^(3/4)) for n > J_COEFFS_COUNT */
static double j_coeff_approx(int n)
{
    if (n <= 0) return 744.0;
    if (n <= J_COEFFS_COUNT) return J_COEFFS[n - 1];
    double sn = sqrt((double)n);
    return exp(4.0 * M_PI * sn) / (M_SQRT2 * pow((double)n, 0.75));
}

/* ================================================================
 * Singular Series S(d)
 *
 * S(d) = 2 * C2 * ∏_{p|d, p>2} (p-1)/(p-2)    for even d
 * S(d) = 0                                       for odd d
 *
 * Returns the density coefficient; expected count = N * S(d) / ln(p_N)
 * ================================================================ */

static long double singular_series(int d)
{
    if (d <= 0 || (d & 1)) return 0.0L;
    long double s = 2.0L * TWIN_PRIME_CONST;
    /* Extract odd part of d — we only care about odd prime divisors */
    int dd = d;
    while (dd % 2 == 0) dd /= 2;
    /* Factor over odd prime divisors */
    for (int p = 3; p * p <= dd; p += 2) {
        if (dd % p == 0) {
            s *= (long double)(p - 1) / (long double)(p - 2);
            while (dd % p == 0) dd /= p;
        }
    }
    if (dd > 1) { /* remaining odd prime factor */
        s *= (long double)(dd - 1) / (long double)(dd - 2);
    }
    return s;
}

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
 * Resonance State — recorded at every k * MONSTER_DIM primes
 * ================================================================ */

typedef struct {
    int64_t k;              /* resonance index */
    int64_t prime;          /* prime at this index */
    double  global_var;     /* Var(g̃) at this point */
    double  window_var;     /* local window Var(g̃) */
    double  residual_g2;    /* ε_2(N) = count_g2/N - S(2)/ln(p) */
    double  residual_g6;    /* ε_6(N) */
    int64_t gap;            /* gap at this prime */
} ResonanceState;

/* ================================================================
 * Convergence Snapshot
 * ================================================================ */

typedef struct {
    double primes_at;
    double global_var;
    double rm;
    double window_var;
    double power_fund;     /* power at 1/196883 */
    double gamma_m;        /* current Pearson Γ_M */
} ConvergenceSnap;

/* ================================================================
 * Checkpoint State (mmap'd)
 * ================================================================ */

typedef struct {
    /* Header */
    uint64_t magic;
    uint32_t version;
    uint32_t pad0;

    /* Progress */
    int64_t primes_processed;
    int64_t max_primes;
    int64_t gaps_analyzed;
    int64_t last_prime;

    /* Global Welford variance (long double) */
    long double welford_mean;
    long double welford_M2;

    /* Gap counts: gap_count[0]=gap1, gap_count[1]=gap2, ..., gap_count[k]=gap(2k) for even */
    int64_t gap_count[N_GAP_BINS];
    int64_t gap_overflow;
    int64_t max_gap_seen;

    /* Goertzel accumulators: harmonics at 1/196883, 2/196883, 3/196883, and null 1/100000 */
    long double goertzel_cos[N_HARMONICS];
    long double goertzel_sin[N_HARMONICS];

    /* Sliding window state */
    int64_t window_count;     /* elements in buffer (≤ MONSTER_DIM) */
    int64_t window_head;      /* circular write position */
    long double window_sum;   /* sum of elements in window */
    long double window_sum2;  /* sum of squares in window */

    /* Pearson correlation (residual vs ln(j-coeff)) */
    long double pear_sx, pear_sy, pear_sxx, pear_syy, pear_sxy;
    int64_t pear_n;

    /* Counters */
    int32_t n_resonances;
    int32_t n_snapshots;
    double  elapsed_seconds;
    double  elapsed_at_resume;

    /* Padding for alignment before arrays */
    char pad1[32];

    /* Resonance states (first MAX_RESONANCES) */
    ResonanceState resonances[MAX_RESONANCES];

    /* Convergence snapshots */
    ConvergenceSnap snapshots[MAX_SNAPSHOTS];

    /* Sliding window buffer: 196883 doubles ≈ 1.55 MB */
    double window_buf[MONSTER_DIM];

} MGSState;

#define CKPT_FILE_SIZE sizeof(MGSState)

/* ================================================================
 * Checkpoint open/close (mmap pattern)
 * ================================================================ */

typedef struct { int fd; MGSState *map; } MGSCheckpointFile;

static MGSState *mgs_ckpt_open(MGSCheckpointFile *ckpt, const char *path, int resume)
{
    int flags = O_RDWR | O_CREAT;
    ckpt->fd = open(path, flags, 0644);
    if (ckpt->fd < 0) { perror("open checkpoint"); exit(1); }

    struct stat st;
    fstat(ckpt->fd, &st);

    if (resume && (size_t)st.st_size >= CKPT_FILE_SIZE) {
        ckpt->map = (MGSState *)mmap(NULL, CKPT_FILE_SIZE,
                     PROT_READ | PROT_WRITE, MAP_SHARED, ckpt->fd, 0);
        if (ckpt->map == MAP_FAILED) { perror("mmap"); exit(1); }
        if (ckpt->map->magic != CKPT_MAGIC || ckpt->map->version != CKPT_VERSION) {
            fprintf(stderr, "Checkpoint magic/version mismatch\n");
            exit(1);
        }
        return ckpt->map;
    }

    if (ftruncate(ckpt->fd, CKPT_FILE_SIZE) < 0) { perror("ftruncate"); exit(1); }
    ckpt->map = (MGSState *)mmap(NULL, CKPT_FILE_SIZE,
                 PROT_READ | PROT_WRITE, MAP_SHARED, ckpt->fd, 0);
    if (ckpt->map == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(ckpt->map, 0, CKPT_FILE_SIZE);
    ckpt->map->magic = CKPT_MAGIC;
    ckpt->map->version = CKPT_VERSION;
    return ckpt->map;
}

static void mgs_ckpt_close(MGSCheckpointFile *ckpt)
{
    if (ckpt->map && ckpt->map != MAP_FAILED) {
        msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
        munmap(ckpt->map, CKPT_FILE_SIZE);
    }
    if (ckpt->fd >= 0) close(ckpt->fd);
}

/* ================================================================
 * Streaming Segmented Sieve (identical to srv_verify.c)
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
    int64_t  seg_lo, seg_len, seg_pos;
    int      seg_valid;
} MGSSieve;

static void ss_init(MGSSieve *sv, int64_t max_primes, int64_t resume_after)
{
    sv->max_primes = max_primes;
    sv->total_produced = 0;
    sv->batch_count = 0;
    sv->seg_valid = 0;
    sv->seg_pos = 0;
    sv->seg_lo = 0;
    sv->seg_len = 0;
    sv->limit = prime_upper_bound(max_primes);
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

static int64_t ss_next_batch(MGSSieve *sv)
{
    sv->batch_count = 0;
    while (sv->batch_count < BATCH_CAPACITY &&
           sv->total_produced < sv->max_primes)
    {
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
        if (sv->seg_pos >= sv->seg_len) {
            sv->current_lo = sv->seg_lo + sv->seg_len;
            sv->seg_valid = 0;
        }
    }
    return sv->batch_count;
}

static void ss_destroy(MGSSieve *sv)
{
    free(sv->base_primes);
    free(sv->segment);
    free(sv->batch);
}

/* ================================================================
 * Goertzel frequencies
 *
 * harm[0] = 1/196883  (fundamental Monster dimension)
 * harm[1] = 2/196883  (2nd harmonic)
 * harm[2] = 3/196883  (3rd harmonic)
 * harm[3] = 1/100000  (null comparison frequency)
 * ================================================================ */

static const double GOERTZEL_FREQ[N_HARMONICS] = {
    1.0 / 196883.0,
    2.0 / 196883.0,
    3.0 / 196883.0,
    1.0 / 100000.0
};

/* ================================================================
 * Process one gap — updates all accumulators
 * ================================================================ */

static inline void process_gap(MGSState *st, int64_t p_curr, int64_t gap)
{
    st->gaps_analyzed++;
    int64_t n = st->gaps_analyzed;

    /* Gap histogram */
    if (gap > 0 && gap <= 2 * N_GAP_BINS) {
        if (gap == 1)
            st->gap_count[0]++;
        else if (gap % 2 == 0)
            st->gap_count[gap / 2]++;
        else
            st->gap_overflow++;
    } else {
        st->gap_overflow++;
    }
    if (gap > st->max_gap_seen) st->max_gap_seen = gap;

    /* Normalized gap */
    long double tilde_g = (long double)gap / logl((long double)p_curr);

    /* 1. Global Welford variance */
    long double delta = tilde_g - st->welford_mean;
    st->welford_mean += delta / (long double)n;
    long double delta2 = tilde_g - st->welford_mean;
    st->welford_M2 += delta * delta2;

    /* 2. Goertzel accumulation at each harmonic (CENTERED: subtract mean=1) */
    long double centered_g = tilde_g - 1.0L;
    for (int h = 0; h < N_HARMONICS; h++) {
        long double theta = 2.0L * M_PIl * (long double)n * (long double)GOERTZEL_FREQ[h];
        st->goertzel_cos[h] += centered_g * cosl(theta);
        st->goertzel_sin[h] += centered_g * sinl(theta);
    }

    /* 3. Sliding window variance */
    double tg_d = (double)tilde_g;
    if (st->window_count < MONSTER_DIM) {
        /* Window not yet full — just accumulate */
        st->window_buf[st->window_count] = tg_d;
        st->window_sum  += tilde_g;
        st->window_sum2 += tilde_g * tilde_g;
        st->window_count++;
        st->window_head = st->window_count % MONSTER_DIM;
    } else {
        /* Window full — slide: remove oldest, add new */
        long double old_val = (long double)st->window_buf[st->window_head];
        st->window_sum  += tilde_g - old_val;
        st->window_sum2 += tilde_g * tilde_g - old_val * old_val;
        st->window_buf[st->window_head] = tg_d;
        st->window_head = (st->window_head + 1) % MONSTER_DIM;

        /*
         * Periodic precision reset: every 100M gaps, recompute
         * window_sum and window_sum2 from the raw buffer to prevent
         * floating-point drift from incremental add/subtract.
         */
        if (n % 100000000 == 0) {
            long double s = 0.0L, s2 = 0.0L;
            for (int64_t w = 0; w < MONSTER_DIM; w++) {
                long double v = (long double)st->window_buf[w];
                s  += v;
                s2 += v * v;
            }
            st->window_sum  = s;
            st->window_sum2 = s2;
        }
    }

    /* 4. Resonance check: every MONSTER_DIM primes */
    if (n % MONSTER_DIM == 0) {
        int64_t k = n / MONSTER_DIM;

        /* Compute current metrics */
        long double var_global = (n > 1) ? st->welford_M2 / (long double)(n - 1) : 0.0L;
        long double var_window = 0.0L;
        if (st->window_count == MONSTER_DIM) {
            long double wmean = st->window_sum / (long double)MONSTER_DIM;
            var_window = st->window_sum2 / (long double)MONSTER_DIM - wmean * wmean;
        }

        /* Residuals: ε_d = count_d/N - S(d)/ln(p) */
        long double ln_p = logl((long double)p_curr);
        long double eps_g2 = (long double)st->gap_count[1] / (long double)n
                             - singular_series(2) / ln_p;
        long double eps_g6 = (long double)st->gap_count[3] / (long double)n
                             - singular_series(6) / ln_p;

        /* Store resonance state */
        if (st->n_resonances < MAX_RESONANCES) {
            ResonanceState *rs = &st->resonances[st->n_resonances];
            rs->k = k;
            rs->prime = p_curr;
            rs->global_var = (double)var_global;
            rs->window_var = (double)var_window;
            rs->residual_g2 = (double)eps_g2;
            rs->residual_g6 = (double)eps_g6;
            rs->gap = gap;
            st->n_resonances++;
        }

        /* Pearson correlation: x = ε_6(N), y = ln(c_k)
         * Both in long double to avoid scale mismatch
         * (ε ~ 10^{-3}, ln(c_k) ~ 10^2) */
        long double ln_jk = logl((long double)j_coeff_approx((int)k));
        long double x = eps_g6;
        long double y = ln_jk;
        st->pear_sx  += x;
        st->pear_sy  += y;
        st->pear_sxx += x * x;
        st->pear_syy += y * y;
        st->pear_sxy += x * y;
        st->pear_n++;
    }
}

/* ================================================================
 * Process a batch of primes
 * ================================================================ */

static void process_batch(MGSState *st, int64_t *batch, int64_t count)
{
    int64_t start = 0;
    if (st->last_prime == 0 && count > 0) {
        st->last_prime = batch[0];
        st->primes_processed = 1;
        start = 1;
    }
    for (int64_t i = start; i < count; i++) {
        int64_t gap = batch[i] - st->last_prime;
        process_gap(st, st->last_prime, gap);
        st->last_prime = batch[i];
        st->primes_processed++;
    }
}

/* ================================================================
 * Record convergence snapshot
 * ================================================================ */

static void record_snapshot(MGSState *st)
{
    if (st->n_snapshots >= MAX_SNAPSHOTS) return;
    int ns = st->n_snapshots;
    if (ns > 0 && st->snapshots[ns-1].primes_at == (double)st->primes_processed)
        return;

    int64_t n = st->gaps_analyzed;
    long double var = (n > 1) ? st->welford_M2 / (long double)(n - 1) : 0.0L;
    long double rm = (st->gap_count[1] > 0)
        ? (long double)st->gap_count[3] / (long double)st->gap_count[1] : 0.0L;
    long double wvar = 0.0L;
    if (st->window_count == MONSTER_DIM) {
        long double wm = st->window_sum / (long double)MONSTER_DIM;
        wvar = st->window_sum2 / (long double)MONSTER_DIM - wm * wm;
    }
    /* Power at fundamental frequency */
    long double power = (st->goertzel_cos[0] * st->goertzel_cos[0] +
                         st->goertzel_sin[0] * st->goertzel_sin[0]) / (long double)n;

    /* Pearson Γ_M */
    double gamma_m = 0.0;
    if (st->pear_n > 2) {
        long double N_p = (long double)st->pear_n;
        long double cov = st->pear_sxy - st->pear_sx * st->pear_sy / N_p;
        long double vx  = st->pear_sxx - st->pear_sx * st->pear_sx / N_p;
        long double vy  = st->pear_syy - st->pear_sy * st->pear_sy / N_p;
        if (vx > 0 && vy > 0)
            gamma_m = (double)(cov / sqrtl(vx * vy));
    }

    st->snapshots[ns].primes_at  = (double)st->primes_processed;
    st->snapshots[ns].global_var = (double)var;
    st->snapshots[ns].rm         = (double)rm;
    st->snapshots[ns].window_var = (double)wvar;
    st->snapshots[ns].power_fund = (double)power;
    st->snapshots[ns].gamma_m    = gamma_m;
    st->n_snapshots++;
}

/* ================================================================
 * Report Generation
 * ================================================================ */

static void generate_report(MGSState *st, const char *path, double elapsed)
{
    FILE *fp = stdout;
    FILE *ff = NULL;
    if (path) {
        ff = fopen(path, "w");
        if (!ff) { perror("fopen report"); return; }
        fp = ff;
    }

    char b1[32], b2[32];
    int64_t n = st->gaps_analyzed;
    long double var = (n > 1) ? st->welford_M2 / (long double)(n - 1) : 0.0L;

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  MONSTROUS GOVERNOR SCAN (MGS Pass-10)\n");
    fprintf(fp, "================================================================\n");
    fprintf(fp, "Primes processed:    %s / %s\n",
            fmt_comma(st->primes_processed, b1, sizeof(b1)),
            fmt_comma(st->max_primes, b2, sizeof(b2)));
    fprintf(fp, "Gaps analyzed:       %s\n", fmt_comma(n, b1, sizeof(b1)));
    fprintf(fp, "Last prime:          %s\n", fmt_comma(st->last_prime, b1, sizeof(b1)));
    fprintf(fp, "Monster dimension:   %s\n", fmt_comma(MONSTER_DIM, b1, sizeof(b1)));
    fprintf(fp, "Resonance points:    %d (every %s primes)\n",
            st->n_resonances > MAX_RESONANCES ? MAX_RESONANCES : st->n_resonances,
            fmt_comma(MONSTER_DIM, b1, sizeof(b1)));
    fprintf(fp, "Elapsed:             %.2f s (%.2f hrs)\n", elapsed, elapsed / 3600.0);
    fprintf(fp, "Precision:           long double (80-bit)\n\n");

    /* --- Global Variance --- */
    fprintf(fp, "--- GLOBAL VARIANCE ---\n");
    fprintf(fp, "  Var(g~):           %.15Lf\n", var);
    fprintf(fp, "  Mean(g~):          %.15Lf\n", st->welford_mean);
    fprintf(fp, "  Gallagher limit:   1.0\n");
    fprintf(fp, "  E8 prediction:     0.707107 (1/sqrt(2))\n\n");

    /* --- Resonance Detector (Goertzel Power Spectrum) --- */
    fprintf(fp, "--- RESONANCE DETECTOR (Power Spectrum) ---\n");
    const char *harm_names[N_HARMONICS] = {
        "1/196883 (Monster fundamental)",
        "2/196883 (2nd harmonic)",
        "3/196883 (3rd harmonic)",
        "1/100000 (null comparison)"
    };
    double expected_power = (double)var / 2.0;  /* white noise: Var/2 per freq bin */
    /* Periodogram at a single frequency follows Exp(mean=expected_power).
       For exponential distribution, std = mean.  Do NOT divide by sqrt(N)
       — that formula applies to sample means, not single DFT coefficients. */
    for (int h = 0; h < N_HARMONICS; h++) {
        long double pw = (st->goertzel_cos[h] * st->goertzel_cos[h] +
                          st->goertzel_sin[h] * st->goertzel_sin[h]) / (long double)n;
        double sigma = (expected_power > 0) ? ((double)pw - expected_power) /
                        expected_power : 0.0;
        fprintf(fp, "  f = %-35s  Power = %.6Lf  (%.2f sigma)\n",
                harm_names[h], pw, sigma);
    }
    fprintf(fp, "  White noise expectation: %.6f\n", expected_power);
    fprintf(fp, "  10-sigma threshold:      %.6f\n\n", expected_power + 10.0 * expected_power);

    /* --- Window Variance --- */
    fprintf(fp, "--- WINDOW VARIANCE (W = %s) ---\n",
            fmt_comma(MONSTER_DIM, b1, sizeof(b1)));
    if (st->window_count == MONSTER_DIM) {
        long double wm = st->window_sum / (long double)MONSTER_DIM;
        long double wvar = st->window_sum2 / (long double)MONSTER_DIM - wm * wm;
        fprintf(fp, "  Current window Var: %.15Lf\n", wvar);
        fprintf(fp, "  Current window mean: %.15Lf\n", wm);
    } else {
        fprintf(fp, "  Window not yet full (%s / %s)\n",
                fmt_comma(st->window_count, b1, sizeof(b1)),
                fmt_comma(MONSTER_DIM, b2, sizeof(b2)));
    }
    fprintf(fp, "\n");

    /* --- Hardy-Littlewood Residuals --- */
    fprintf(fp, "--- HARDY-LITTLEWOOD RESIDUALS ---\n");
    long double ln_p = logl((long double)st->last_prime);
    int gaps_to_show[] = {2, 4, 6, 8, 10, 12, 14, 18, 20, 30};
    int n_show = 10;
    fprintf(fp, "  %-6s  %-16s  %-12s  %-12s  %-12s\n",
            "Gap", "Count", "Observed", "S(d)/ln(p)", "Residual");
    for (int i = 0; i < n_show; i++) {
        int d = gaps_to_show[i];
        int idx = d / 2;
        if (idx >= N_GAP_BINS) continue;
        long double obs = (long double)st->gap_count[idx] / (long double)n;
        long double ss = singular_series(d) / ln_p;
        long double eps = obs - ss;
        fprintf(fp, "  %-6d  %-16s  %.10Lf  %.10Lf  %+.10Lf\n",
                d, fmt_comma(st->gap_count[idx], b1, sizeof(b1)),
                obs, ss, eps);
    }
    fprintf(fp, "\n");

    /* --- Monstrous Correlation Γ_M --- */
    fprintf(fp, "--- MONSTROUS CORRELATION (Gamma_M) ---\n");
    if (st->pear_n > 2) {
        long double N_p = (long double)st->pear_n;
        long double cov = st->pear_sxy - st->pear_sx * st->pear_sy / N_p;
        long double vx  = st->pear_sxx - st->pear_sx * st->pear_sx / N_p;
        long double vy  = st->pear_syy - st->pear_sy * st->pear_sy / N_p;
        double gamma = 0.0;
        if (vx > 0 && vy > 0)
            gamma = (double)(cov / sqrtl(vx * vy));
        fprintf(fp, "  Pearson Gamma_M:   %.10f\n", gamma);
        fprintf(fp, "  Samples (k):       %s\n",
                fmt_comma(st->pear_n, b1, sizeof(b1)));
        fprintf(fp, "  Prediction:        > 0.95 if Monster governs\n");
        fprintf(fp, "  Status:            %s\n\n",
                fabs(gamma) > 0.95 ? "MONSTROUS RESONANCE" :
                fabs(gamma) > 0.50 ? "MODERATE CORRELATION" :
                fabs(gamma) > 0.10 ? "WEAK CORRELATION" : "NO CORRELATION");
    } else {
        fprintf(fp, "  Insufficient resonance points (need > 2)\n\n");
    }

    /* --- Resonance States (first 20) --- */
    fprintf(fp, "--- RESONANCE STATES (first 20) ---\n");
    fprintf(fp, "  %-6s  %-20s  %-12s  %-12s  %-12s  %-12s\n",
            "k", "Prime", "Global Var", "Window Var", "eps_g2", "eps_g6");
    int to_show = st->n_resonances < 20 ? st->n_resonances : 20;
    for (int i = 0; i < to_show; i++) {
        ResonanceState *rs = &st->resonances[i];
        fprintf(fp, "  %-6" PRId64 "  %-20s  %.8f  %.8f  %+.8f  %+.8f\n",
                rs->k, fmt_comma(rs->prime, b1, sizeof(b1)),
                rs->global_var, rs->window_var,
                rs->residual_g2, rs->residual_g6);
    }
    fprintf(fp, "\n");

    /* --- Gap Distribution --- */
    fprintf(fp, "--- GAP DISTRIBUTION ---\n");
    fprintf(fp, "  %-6s  %-16s  %-10s\n", "Gap", "Count", "Fraction");
    for (int i = 0; i < N_GAP_BINS; i++) {
        if (st->gap_count[i] == 0) continue;
        int gap_val = (i == 0) ? 1 : 2 * i;
        double frac = 100.0 * (double)st->gap_count[i] / (double)n;
        fprintf(fp, "  %-6d  %-16s  %.6f%%\n",
                gap_val, fmt_comma(st->gap_count[i], b1, sizeof(b1)), frac);
    }
    if (st->gap_overflow > 0)
        fprintf(fp, "  >%-5d %-16s  %.6f%%\n",
                2 * N_GAP_BINS, fmt_comma(st->gap_overflow, b1, sizeof(b1)),
                100.0 * (double)st->gap_overflow / (double)n);
    fprintf(fp, "  Max gap seen:   %s\n", fmt_comma(st->max_gap_seen, b1, sizeof(b1)));
    fprintf(fp, "\n");

    /* --- Convergence History --- */
    fprintf(fp, "--- CONVERGENCE HISTORY ---\n");
    fprintf(fp, "  Snapshots: %d\n", st->n_snapshots);
    fprintf(fp, "  %-20s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
            "Primes", "Var(g~)", "g6/g2", "Win Var", "Power_f1", "Gamma_M");
    for (int i = 0; i < st->n_snapshots; i++) {
        ConvergenceSnap *cs = &st->snapshots[i];
        fprintf(fp, "  %-20s  %.8f  %.8f  %.8f  %.8f  %+.8f\n",
                fmt_comma((int64_t)cs->primes_at, b1, sizeof(b1)),
                cs->global_var, cs->rm, cs->window_var,
                cs->power_fund, cs->gamma_m);
    }
    fprintf(fp, "\n");

    /* --- 196883 Checksum --- */
    fprintf(fp, "--- VERIFICATION CHECKSUMS ---\n");
    fprintf(fp, "  196,883rd prime (expected 2,714,441): ");
    if (st->n_resonances > 0)
        fprintf(fp, "%s\n", fmt_comma(st->resonances[0].prime, b1, sizeof(b1)));
    else
        fprintf(fp, "not yet reached\n");
    fprintf(fp, "  Variance at 196,883 gaps:             ");
    if (st->n_resonances > 0)
        fprintf(fp, "%.10f\n", st->resonances[0].global_var);
    else
        fprintf(fp, "not yet computed\n");

    /* --- Summary --- */
    fprintf(fp, "\n--- SUMMARY ---\n");
    long double power_fund = (st->goertzel_cos[0] * st->goertzel_cos[0] +
                              st->goertzel_sin[0] * st->goertzel_sin[0]) / (long double)n;
    long double power_null = (st->goertzel_cos[3] * st->goertzel_cos[3] +
                              st->goertzel_sin[3] * st->goertzel_sin[3]) / (long double)n;
    fprintf(fp, "  Global Var(g~) = %.15Lf\n", var);
    fprintf(fp, "  Power at 1/%s = %.10Lf  (null = %.10Lf, ratio = %.4Lf)\n",
            fmt_comma(MONSTER_DIM, b1, sizeof(b1)),
            power_fund, power_null,
            (power_null > 0) ? power_fund / power_null : 0.0L);
    if (st->pear_n > 2) {
        long double N_p = (long double)st->pear_n;
        long double cov = st->pear_sxy - st->pear_sx * st->pear_sy / N_p;
        long double vx  = st->pear_sxx - st->pear_sx * st->pear_sx / N_p;
        long double vy  = st->pear_syy - st->pear_sy * st->pear_sy / N_p;
        double gamma = 0.0;
        if (vx > 0 && vy > 0)
            gamma = (double)(cov / sqrtl(vx * vy));
        fprintf(fp, "  Gamma_M = %.10f  (k = %s resonance points)\n",
                gamma, fmt_comma(st->pear_n, b1, sizeof(b1)));
    }

    fprintf(fp, "\n================================================================\n");
    fprintf(fp, "  END OF REPORT\n");
    fprintf(fp, "================================================================\n");

    if (ff) {
        fclose(ff);
        fprintf(stdout, "Report: %s\n", path);
    }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    /* Defaults */
    int64_t max_primes = 10000000000LL;    /* 10B */
    const char *ckpt_path = "mgs_state.ebd";
    const char *output_path = "spiral_outputs/mgs_report.txt";
    int64_t interval = 100000000;          /* 100M */
    int resume = 0;

    static struct option long_opts[] = {
        {"max-primes", required_argument, 0, 'n'},
        {"checkpoint", required_argument, 0, 'c'},
        {"output",     required_argument, 0, 'o'},
        {"interval",   required_argument, 0, 'i'},
        {"resume",     no_argument,       0, 'r'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:c:o:i:rh", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'n': max_primes = atoll(optarg); break;
            case 'c': ckpt_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 'i': interval = atoll(optarg); break;
            case 'r': resume = 1; break;
            case 'h':
                printf("Usage: %s [options]\n"
                       "  --max-primes N   Target prime count [10B]\n"
                       "  --checkpoint F   Checkpoint file [mgs_state.ebd]\n"
                       "  --output F       Report output [spiral_outputs/mgs_report.txt]\n"
                       "  --interval N     Checkpoint interval [100M]\n"
                       "  --resume         Resume from checkpoint\n",
                       argv[0]);
                return 0;
            default: return 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    char b1[32], b2[32];
    printf("================================================================\n");
    printf("  MONSTROUS GOVERNOR SCAN (MGS Pass-10)\n");
    printf("================================================================\n");
    printf("Max primes:       %s\n", fmt_comma(max_primes, b1, sizeof(b1)));
    printf("Monster dim:      %s\n", fmt_comma(MONSTER_DIM, b1, sizeof(b1)));
    printf("Checkpoint:       %s\n", ckpt_path);
    printf("Checkpoint every: %s primes\n", fmt_comma(interval, b1, sizeof(b1)));
    printf("Output:           %s\n", output_path);
    printf("Precision:        long double (%zu bytes)\n", sizeof(long double));
    printf("Checkpoint size:  %.2f MB\n", (double)CKPT_FILE_SIZE / (1024.0 * 1024.0));
    printf("Threads:          %d\n", omp_get_max_threads());

    if (resume) printf("\nResuming from %s\n", ckpt_path);

    /* Open checkpoint */
    MGSCheckpointFile ckpt;
    MGSState *st = mgs_ckpt_open(&ckpt, ckpt_path, resume);
    st->max_primes = max_primes;

    if (resume && st->primes_processed > 0) {
        printf("  Loaded: %s primes, %s gaps\n",
               fmt_comma(st->primes_processed, b1, sizeof(b1)),
               fmt_comma(st->gaps_analyzed, b2, sizeof(b2)));
    }

    tic();

    /* Initialize sieve */
    printf("\nInitializing sieve... ");
    fflush(stdout);

    MGSSieve sv;
    int64_t resume_after = resume ? st->last_prime : 0;
    ss_init(&sv, max_primes, resume_after);

    /* If resuming, skip already-processed primes */
    if (resume && st->primes_processed > 0) {
        printf("skipping %s primes... ", fmt_comma(st->primes_processed, b1, sizeof(b1)));
        fflush(stdout);
        int64_t to_skip = st->primes_processed;
        while (to_skip > 0) {
            int64_t got = ss_next_batch(&sv);
            if (got <= 0) break;
            if (got <= to_skip) {
                to_skip -= got;
            } else {
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
        st->elapsed_at_resume = st->elapsed_seconds;
    }

    printf("done (base primes: %s, limit: %s)\n",
           fmt_comma(sv.n_base, b1, sizeof(b1)),
           fmt_comma(sv.limit, b2, sizeof(b2)));
    printf("\nProcessing...\n");
    fflush(stdout);

    int64_t next_checkpoint = st->primes_processed + interval;

    /* Main loop */
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
            double rate = (double)st->primes_processed / elapsed;

            int64_t n_gaps = st->gaps_analyzed;
            long double var = (n_gaps > 1) ? st->welford_M2 / (long double)(n_gaps - 1) : 0.0L;
            long double rm = (st->gap_count[1] > 0)
                ? (long double)st->gap_count[3] / (long double)st->gap_count[1] : 0.0L;

            record_snapshot(st);

            printf("  %s / %s (%5.1f%%) | %5.1fM/s | Var=%.10Lf | g6/g2=%.8Lf | res=%d\n",
                   fmt_comma(st->primes_processed, b1, sizeof(b1)),
                   fmt_comma(max_primes, b2, sizeof(b2)),
                   100.0 * (double)st->primes_processed / (double)max_primes,
                   rate / 1e6, var, rm, st->n_resonances);
            fflush(stdout);

            /* Sync checkpoint */
            msync(st, CKPT_FILE_SIZE, MS_ASYNC);
            printf("  [CHECKPOINT] %s primes saved\n",
                   fmt_comma(st->primes_processed, b1, sizeof(b1)));
            fflush(stdout);

            next_checkpoint = st->primes_processed + interval;
        }
    }

    /* Final */
    double elapsed = toc() + st->elapsed_at_resume;
    st->elapsed_seconds = elapsed;
    record_snapshot(st);
    msync(st, CKPT_FILE_SIZE, MS_SYNC);

    printf("\nDone. Generating report...\n");
    generate_report(st, output_path, elapsed);
    generate_report(st, NULL, elapsed);

    ss_destroy(&sv);
    mgs_ckpt_close(&ckpt);
    return 0;
}
