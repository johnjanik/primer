/*
 * base18_decoder.c — EBD Pass-7 Lagrangian Decoder
 *
 * Streaming high-performance decoder: primes -> E8 -> base-18 symbols
 * with mmap checkpoints, OpenMP parallelism, and triality validation.
 * Designed for runs of 100+ trillion primes.
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o base18_decoder base18_decoder.c -lm
 *
 * Usage:
 *   ./base18_decoder --max-primes 10000000000 [--checkpoint FILE] [--resume]
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

#define ACTIVE_SYMBOLS   18
#define SILENT_SYMBOLS   8
#define TEXT_SAMPLE_SIZE  (1 << 20)   /* 1M symbols for head/tail samples */
#define BATCH_CAPACITY   (1 << 20)   /* 1M primes per processing batch */
#define SIEVE_SEGMENT    (1 << 19)   /* 512K per sieve segment */
#define DEFAULT_CKPT_INT (100000000LL)  /* checkpoint every 100M primes */
#define CKPT_MAGIC       0x4542443138ULL
#define CKPT_VERSION     1
#define HAMILTONIAN_LEN  14

/* ================================================================
 * Base-18 Alphabet & Mapping
 * ================================================================ */

static const char ALPHABET_18[ACTIVE_SYMBOLS] = {
    'J','O','I','R','E','H','Q','P',
    'G','U','A','Y','V','C','N','F','B','M'
};
static const char SILENT_CHARS[SILENT_SYMBOLS] = {
    'D','K','L','S','T','W','X','Z'
};
static const int SILENT_MOD26[SILENT_SYMBOLS] = {3,10,11,18,19,22,23,25};

static int8_t g_root_to_base18[E8_NUM_ROOTS];   /* -1 = silent */
static int8_t g_root_to_silent[E8_NUM_ROOTS];   /* -1 = active, 0-7 = silent idx */

static void init_base18_mapping(void)
{
    int8_t mod26_to_base18[26];
    memset(mod26_to_base18, -1, 26);

    int active_idx = 0;
    for (int m = 0; m < 26; m++) {
        int is_silent = 0;
        for (int s = 0; s < SILENT_SYMBOLS; s++)
            if (m == SILENT_MOD26[s]) { is_silent = 1; break; }
        if (!is_silent)
            mod26_to_base18[m] = (int8_t)(active_idx++ % ACTIVE_SYMBOLS);
    }

    for (int i = 0; i < E8_NUM_ROOTS; i++) {
        int m = i % 26;
        g_root_to_base18[i] = mod26_to_base18[m];
        g_root_to_silent[i] = -1;
        for (int s = 0; s < SILENT_SYMBOLS; s++)
            if (m == SILENT_MOD26[s]) { g_root_to_silent[i] = (int8_t)s; break; }
    }
}

/* ================================================================
 * Checkpoint State (mmap'd)
 * ================================================================ */

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t _pad0;

    /* Progress */
    int64_t primes_processed;
    int64_t max_primes;
    int64_t symbols_produced;
    int64_t spacers_produced;
    int64_t last_prime;

    /* Base-18 frequencies */
    int64_t symbol_freq[ACTIVE_SYMBOLS];
    int64_t spacer_freq[SILENT_SYMBOLS];

    /* E8 root histogram */
    int64_t root_freq[E8_NUM_ROOTS];

    /* Gap statistics */
    double  sum_norm_gap;
    double  sum_norm_gap2;
    int64_t n_gaps_total;

    /* Triality accumulator (current partial state) */
    int     tri_phase;
    int     tri_gap_in_phase;
    double  tri_acc[3][E8_DIM];
    int     tri_symbols[3];

    /* Triality results */
    int64_t tri_transcendental;
    int64_t tri_partial;
    int64_t tri_noise;
    int64_t tri_total;

    /* Salem-Jordan filter */
    int64_t salem_passed;
    int64_t salem_rejected;
    double  salem_sum_tension;

    /* Bigram matrix */
    int64_t bigram[ACTIVE_SYMBOLS][ACTIVE_SYMBOLS];
    int     last_active_symbol;

    /* Hamiltonian path (14-char repeat) */
    int64_t hamiltonian_matches;
    int8_t  recent_symbols[HAMILTONIAN_LEN];
    int     recent_count;

    /* Text sample bookkeeping */
    int64_t text_head_len;
    int64_t text_tail_pos;
    int64_t text_tail_count;

    /* Timing */
    double  elapsed_seconds;
    double  elapsed_at_resume;

    char    _pad1[48];
} CheckpointState;

/* File layout: [CheckpointState][text_head][text_tail] */
#define CKPT_HEAD_OFF  ((int64_t)sizeof(CheckpointState))
#define CKPT_TAIL_OFF  (CKPT_HEAD_OFF + TEXT_SAMPLE_SIZE)
#define CKPT_FILE_SIZE (CKPT_TAIL_OFF + TEXT_SAMPLE_SIZE)

typedef struct {
    int              fd;
    void            *map;
    CheckpointState *state;
    uint8_t         *text_head;
    uint8_t         *text_tail;
    char             path[512];
} CheckpointFile;

static int ckpt_open(CheckpointFile *ckpt, const char *path, int resume)
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

    ckpt->state     = (CheckpointState *)ckpt->map;
    ckpt->text_head = (uint8_t *)ckpt->map + CKPT_HEAD_OFF;
    ckpt->text_tail = (uint8_t *)ckpt->map + CKPT_TAIL_OFF;

    if (resume) {
        if (ckpt->state->magic != CKPT_MAGIC || ckpt->state->version != CKPT_VERSION) {
            fprintf(stderr, "Invalid checkpoint (bad magic/version)\n");
            munmap(ckpt->map, CKPT_FILE_SIZE);
            close(ckpt->fd);
            return -1;
        }
    } else {
        memset(ckpt->map, 0, CKPT_FILE_SIZE);
        ckpt->state->magic   = CKPT_MAGIC;
        ckpt->state->version = CKPT_VERSION;
        ckpt->state->last_active_symbol = -1;
    }
    return 0;
}

static void ckpt_sync(CheckpointFile *ckpt)
{
    msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
}

static void ckpt_close(CheckpointFile *ckpt)
{
    if (ckpt->map && ckpt->map != MAP_FAILED) {
        msync(ckpt->map, CKPT_FILE_SIZE, MS_SYNC);
        munmap(ckpt->map, CKPT_FILE_SIZE);
    }
    if (ckpt->fd >= 0) close(ckpt->fd);
}

static void ckpt_append_symbol(CheckpointFile *ckpt, uint8_t ch)
{
    CheckpointState *s = ckpt->state;
    if (s->text_head_len < TEXT_SAMPLE_SIZE)
        ckpt->text_head[s->text_head_len++] = ch;
    ckpt->text_tail[s->text_tail_pos % TEXT_SAMPLE_SIZE] = ch;
    s->text_tail_pos++;
    s->text_tail_count++;
}

/* ================================================================
 * Streaming Segmented Sieve
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

static void sieve_init(StreamingSieve *sv, int64_t max_primes, int64_t resume_after)
{
    sv->max_primes = max_primes;
    sv->total_produced = 0;
    sv->batch_count = 0;
    sv->limit = prime_upper_bound(max_primes);
    /* Safety margin for Rosser-Schoenfeld approximation */
    sv->limit += sv->limit / 20 + 100000;

    int64_t sqrt_limit = (int64_t)sqrt((double)sv->limit) + 1;
    uint8_t *is_comp = (uint8_t *)calloc(sqrt_limit + 1, 1);
    if (!is_comp) { fprintf(stderr, "sieve_init: calloc failed\n"); exit(1); }

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

static int64_t sieve_next_batch(StreamingSieve *sv)
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

        /* Mark composites — OpenMP parallel over base primes */
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

static void sieve_destroy(StreamingSieve *sv)
{
    free(sv->base_primes);
    free(sv->segment);
    free(sv->batch);
}

/* ================================================================
 * Batch processing: gaps -> E8 -> base-18 -> statistics
 * ================================================================ */

static void process_batch(const int64_t *batch, int64_t count,
                          int64_t *prev_prime_ptr,
                          const E8Lattice *e8,
                          CheckpointFile *ckpt)
{
    CheckpointState *st = ckpt->state;
    int64_t prev = *prev_prime_ptr;

    int64_t start_idx = 0;
    if (prev <= 0) {
        prev = batch[0];
        start_idx = 1;
    }

    for (int64_t i = start_idx; i < count; i++) {
        int64_t p = batch[i];
        int64_t gap = p - prev;
        double log_p = log((double)prev);
        if (log_p < 1.0) log_p = 1.0;
        double norm_gap = (double)gap / log_p;

        st->sum_norm_gap  += norm_gap;
        st->sum_norm_gap2 += norm_gap * norm_gap;
        st->n_gaps_total++;

        int root_idx = e8_assign_root(e8, norm_gap);
        st->root_freq[root_idx]++;

        int8_t b18 = g_root_to_base18[root_idx];
        int8_t sil = g_root_to_silent[root_idx];
        uint8_t out_char;

        if (b18 >= 0) {
            st->symbol_freq[b18]++;
            st->symbols_produced++;
            out_char = (uint8_t)ALPHABET_18[b18];

            /* Bigram */
            if (st->last_active_symbol >= 0)
                st->bigram[st->last_active_symbol][b18]++;
            st->last_active_symbol = b18;

            /* Hamiltonian 14-char detection */
            if (st->recent_count < HAMILTONIAN_LEN) {
                st->recent_symbols[st->recent_count++] = b18;
            } else {
                memmove(st->recent_symbols, st->recent_symbols + 1, HAMILTONIAN_LEN - 1);
                st->recent_symbols[HAMILTONIAN_LEN - 1] = b18;
            }
            if (st->recent_count == HAMILTONIAN_LEN) {
                int match = 1;
                for (int k = 0; k < 7; k++) {
                    if (st->recent_symbols[k] != st->recent_symbols[k + 7]) {
                        match = 0; break;
                    }
                }
                if (match) st->hamiltonian_matches++;
            }
        } else {
            if (sil >= 0) st->spacer_freq[sil]++;
            st->spacers_produced++;
            out_char = '.';
            st->last_active_symbol = -1;
        }

        ckpt_append_symbol(ckpt, out_char);

        /* Triality: group 3 consecutive gaps */
        for (int d = 0; d < E8_DIM; d++)
            st->tri_acc[st->tri_phase][d] += e8->roots[root_idx][d];
        st->tri_symbols[st->tri_phase] = b18;
        st->tri_gap_in_phase++;

        if (st->tri_gap_in_phase >= 1) {
            st->tri_gap_in_phase = 0;
            st->tri_phase++;

            if (st->tri_phase >= 3) {
                st->tri_total++;

                /* Salem-Jordan: coherence = |v1+v2+v3|^2 / (|v1|^2+|v2|^2+|v3|^2) */
                double sum_vec[E8_DIM];
                double norm2_parts = 0, norm2_sum = 0;
                for (int d = 0; d < E8_DIM; d++) {
                    sum_vec[d] = st->tri_acc[0][d] + st->tri_acc[1][d] + st->tri_acc[2][d];
                    norm2_sum += sum_vec[d] * sum_vec[d];
                    for (int t = 0; t < 3; t++)
                        norm2_parts += st->tri_acc[t][d] * st->tri_acc[t][d];
                }

                double tension = (norm2_parts > 0) ? norm2_sum / norm2_parts : 0;
                st->salem_sum_tension += tension;

                /* Coherence near 1/3 = random; near 1.0 = aligned */
                if (fabs(norm2_sum - 3.0) < 1e-6)
                    st->salem_passed++;
                else
                    st->salem_rejected++;

                /* Triality classification (active symbols only) */
                int s0 = st->tri_symbols[0], s1 = st->tri_symbols[1], s2 = st->tri_symbols[2];
                if (s0 >= 0 && s1 >= 0 && s2 >= 0) {
                    if (s0 == s1 && s1 == s2)
                        st->tri_transcendental++;
                    else if (s0 == s1 || s1 == s2 || s0 == s2)
                        st->tri_partial++;
                    else
                        st->tri_noise++;
                }

                st->tri_phase = 0;
                memset(st->tri_acc, 0, sizeof(st->tri_acc));
            }
        }

        prev = p;
    }

    *prev_prime_ptr = prev;
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
 * Report
 * ================================================================ */

static void generate_report(FILE *fp, CheckpointFile *ckpt)
{
    CheckpointState *st = ckpt->state;
    char b1[64], b2[64];

    fprintf(fp, "================================================================\n");
    fprintf(fp, "  BASE-18 DECODER (EBD Pass-7 Lagrangian)\n");
    fprintf(fp, "================================================================\n");
    fprintf(fp, "Primes processed:  %s / %s\n",
            fmt_comma(st->primes_processed, b1, sizeof(b1)),
            fmt_comma(st->max_primes, b2, sizeof(b2)));
    fprintf(fp, "Gaps analyzed:     %s\n", fmt_comma(st->n_gaps_total, b1, sizeof(b1)));
    fprintf(fp, "Active symbols:    %s (%.2f%%)\n",
            fmt_comma(st->symbols_produced, b1, sizeof(b1)),
            st->n_gaps_total > 0 ? 100.0 * st->symbols_produced / st->n_gaps_total : 0.0);
    fprintf(fp, "Silent spacers:    %s (%.2f%%)\n",
            fmt_comma(st->spacers_produced, b1, sizeof(b1)),
            st->n_gaps_total > 0 ? 100.0 * st->spacers_produced / st->n_gaps_total : 0.0);
    fprintf(fp, "Last prime:        %s\n", fmt_comma(st->last_prime, b1, sizeof(b1)));
    fprintf(fp, "Elapsed:           %.2f s (%.2f hrs)\n\n",
            st->elapsed_seconds, st->elapsed_seconds / 3600.0);

    /* Gap statistics */
    if (st->n_gaps_total > 0) {
        double mean_g = st->sum_norm_gap / st->n_gaps_total;
        double var_g  = st->sum_norm_gap2 / st->n_gaps_total - mean_g * mean_g;
        fprintf(fp, "--- GAP STATISTICS ---\n");
        fprintf(fp, "  Mean normalized gap: %.6f\n", mean_g);
        fprintf(fp, "  Std normalized gap:  %.6f\n\n", var_g > 0 ? sqrt(var_g) : 0.0);
    }

    /* Base-18 frequencies */
    fprintf(fp, "--- BASE-18 SYMBOL FREQUENCIES ---\n");
    int64_t max_freq = 0;
    for (int i = 0; i < ACTIVE_SYMBOLS; i++)
        if (st->symbol_freq[i] > max_freq) max_freq = st->symbol_freq[i];

    for (int i = 0; i < ACTIVE_SYMBOLS; i++) {
        int bar = max_freq > 0 ? (int)(40.0 * st->symbol_freq[i] / max_freq) : 0;
        fprintf(fp, "  %c: ", ALPHABET_18[i]);
        for (int j = 0; j < 40; j++) fputc(j < bar ? '#' : ' ', fp);
        fprintf(fp, " %12s (%5.2f%%)\n",
                fmt_comma(st->symbol_freq[i], b1, sizeof(b1)),
                st->symbols_produced > 0 ? 100.0 * st->symbol_freq[i] / st->symbols_produced : 0.0);
    }
    fprintf(fp, "\n");

    /* Silent spacers */
    fprintf(fp, "--- SILENT SPACER FREQUENCIES ---\n");
    for (int i = 0; i < SILENT_SYMBOLS; i++) {
        fprintf(fp, "  %c: %12s (%5.2f%%)\n", SILENT_CHARS[i],
                fmt_comma(st->spacer_freq[i], b1, sizeof(b1)),
                st->spacers_produced > 0 ? 100.0 * st->spacer_freq[i] / st->spacers_produced : 0.0);
    }
    fprintf(fp, "\n");

    /* Entropy */
    double H = 0;
    if (st->symbols_produced > 0) {
        for (int i = 0; i < ACTIVE_SYMBOLS; i++) {
            if (st->symbol_freq[i] > 0) {
                double p = (double)st->symbol_freq[i] / st->symbols_produced;
                H -= p * log2(p);
            }
        }
    }
    double max_H = log2(ACTIVE_SYMBOLS);
    fprintf(fp, "--- INFORMATION THEORY ---\n");
    fprintf(fp, "  Shannon entropy:    %.4f bits/symbol (max=%.4f, util=%.2f%%)\n\n",
            H, max_H, max_H > 0 ? 100.0 * H / max_H : 0.0);

    /* Triality */
    fprintf(fp, "--- TRIALITY ANALYSIS ---\n");
    fprintf(fp, "  Total triplets:     %s\n", fmt_comma(st->tri_total, b1, sizeof(b1)));
    int64_t tri_class = st->tri_transcendental + st->tri_partial + st->tri_noise;
    fprintf(fp, "  Transcendental:     %s (%.4f%%)\n",
            fmt_comma(st->tri_transcendental, b1, sizeof(b1)),
            tri_class > 0 ? 100.0 * st->tri_transcendental / tri_class : 0.0);
    fprintf(fp, "  Partial:            %s (%.4f%%)\n",
            fmt_comma(st->tri_partial, b1, sizeof(b1)),
            tri_class > 0 ? 100.0 * st->tri_partial / tri_class : 0.0);
    fprintf(fp, "  Noise:              %s (%.4f%%)\n",
            fmt_comma(st->tri_noise, b1, sizeof(b1)),
            tri_class > 0 ? 100.0 * st->tri_noise / tri_class : 0.0);
    fprintf(fp, "  Expected trans (uniform): %.4f%%\n\n",
            100.0 / (ACTIVE_SYMBOLS * ACTIVE_SYMBOLS));

    /* Salem-Jordan */
    fprintf(fp, "--- SALEM-JORDAN FILTER ---\n");
    fprintf(fp, "  Passed (|sum|^2 ~ 3.0):  %s\n", fmt_comma(st->salem_passed, b1, sizeof(b1)));
    fprintf(fp, "  Rejected:                %s\n", fmt_comma(st->salem_rejected, b1, sizeof(b1)));
    double avg_t = st->tri_total > 0 ? st->salem_sum_tension / st->tri_total : 0;
    fprintf(fp, "  Mean coherence:          %.6f (random ~ 0.333)\n\n", avg_t);

    /* Hamiltonian */
    fprintf(fp, "--- HAMILTONIAN PATH (14-char G2 repeat) ---\n");
    fprintf(fp, "  Period-7 matches:  %s\n\n", fmt_comma(st->hamiltonian_matches, b1, sizeof(b1)));

    /* Bigram matrix excerpt */
    fprintf(fp, "--- BIGRAM MATRIX (10x10 excerpt) ---\n");
    fprintf(fp, "     ");
    for (int j = 0; j < 10; j++) fprintf(fp, "%8c", ALPHABET_18[j]);
    fprintf(fp, "\n");
    for (int i = 0; i < 10; i++) {
        fprintf(fp, "  %c  ", ALPHABET_18[i]);
        for (int j = 0; j < 10; j++)
            fprintf(fp, "%8ld", (long)st->bigram[i][j]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "\n");

    /* Text head sample */
    int64_t head_show = st->text_head_len < 360 ? st->text_head_len : 360;
    fprintf(fp, "--- TEXT SAMPLE (first %ld chars, Weyl width 72) ---\n", (long)head_show);
    for (int64_t i = 0; i < head_show; i++) {
        fputc(ckpt->text_head[i], fp);
        if ((i + 1) % 72 == 0) fputc('\n', fp);
    }
    if (head_show > 0 && head_show % 72 != 0) fputc('\n', fp);
    fprintf(fp, "\n");

    /* Text tail sample */
    if (st->text_tail_count > 0) {
        int64_t tail_avail = st->text_tail_count < TEXT_SAMPLE_SIZE
            ? st->text_tail_count : TEXT_SAMPLE_SIZE;
        int64_t tail_show = tail_avail < 360 ? tail_avail : 360;
        fprintf(fp, "--- TEXT SAMPLE (last %ld chars) ---\n", (long)tail_show);
        int64_t start_p = (st->text_tail_pos - tail_show + TEXT_SAMPLE_SIZE) % TEXT_SAMPLE_SIZE;
        for (int64_t i = 0; i < tail_show; i++) {
            fputc(ckpt->text_tail[(start_p + i) % TEXT_SAMPLE_SIZE], fp);
            if ((i + 1) % 72 == 0) fputc('\n', fp);
        }
        if (tail_show > 0 && tail_show % 72 != 0) fputc('\n', fp);
        fprintf(fp, "\n");
    }

    /* E8 root usage */
    int roots_used = 0;
    for (int i = 0; i < E8_NUM_ROOTS; i++)
        if (st->root_freq[i] > 0) roots_used++;
    fprintf(fp, "--- E8 ROOT USAGE ---\n");
    fprintf(fp, "  Distinct roots hit: %d / %d\n\n", roots_used, E8_NUM_ROOTS);

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
    fprintf(stderr, "  --checkpoint FILE   Checkpoint file [default: base18_state.ebd]\n");
    fprintf(stderr, "  --resume            Resume from checkpoint\n");
    fprintf(stderr, "  --output FILE       Report file [default: spiral_outputs/base18_report.txt]\n");
    fprintf(stderr, "  --interval N        Checkpoint interval [default: 100000000]\n");
    fprintf(stderr, "  --help              Show help\n");
}

int main(int argc, char **argv)
{
    int64_t max_primes   = 10000000000LL;
    const char *ckpt_path   = "base18_state.ebd";
    const char *output_path = "spiral_outputs/base18_report.txt";
    int64_t ckpt_interval   = DEFAULT_CKPT_INT;
    int do_resume = 0;

    static struct option long_opts[] = {
        {"max-primes",  required_argument, 0, 'p'},
        {"checkpoint",  required_argument, 0, 'c'},
        {"resume",      no_argument,       0, 'r'},
        {"output",      required_argument, 0, 'o'},
        {"interval",    required_argument, 0, 'i'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "p:c:ro:i:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'p': max_primes    = atoll(optarg); break;
        case 'c': ckpt_path     = optarg; break;
        case 'r': do_resume     = 1; break;
        case 'o': output_path   = optarg; break;
        case 'i': ckpt_interval = atoll(optarg); break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    signal(SIGINT,  handle_signal);
    signal(SIGTERM, handle_signal);

    char b1[64], b2[64];

    printf("================================================================\n");
    printf("  BASE-18 DECODER (EBD Pass-7 Lagrangian)\n");
    printf("================================================================\n");
    printf("Max primes:       %s\n", fmt_comma(max_primes, b1, sizeof(b1)));
    printf("Checkpoint:       %s%s\n", ckpt_path, do_resume ? " (RESUME)" : "");
    printf("Checkpoint every: %s primes\n", fmt_comma(ckpt_interval, b1, sizeof(b1)));
    printf("Output:           %s\n", output_path);
    printf("Threads:          %d\n\n", omp_get_max_threads());

    /* E8 + mapping */
    printf("Initializing E8 lattice + base-18 mapping... ");
    fflush(stdout);
    E8Lattice e8;
    e8_init(&e8);
    init_base18_mapping();
    printf("done\n");

    /* Checkpoint */
    printf("Opening checkpoint... ");
    fflush(stdout);
    CheckpointFile ckpt;
    if (ckpt_open(&ckpt, ckpt_path, do_resume) < 0)
        return 1;
    ckpt.state->max_primes = max_primes;

    if (do_resume) {
        printf("resumed from %s primes, last_prime=%s\n",
               fmt_comma(ckpt.state->primes_processed, b1, sizeof(b1)),
               fmt_comma(ckpt.state->last_prime, b2, sizeof(b2)));
    } else {
        printf("fresh start\n");
    }

    /* Streaming sieve */
    printf("Initializing sieve... ");
    fflush(stdout);
    StreamingSieve sieve;
    int64_t resume_after = do_resume ? ckpt.state->last_prime : 0;
    int64_t resume_count = do_resume ? ckpt.state->primes_processed : 0;
    sieve_init(&sieve, max_primes, resume_after);
    if (do_resume) sieve.total_produced = resume_count;
    printf("done (base primes: %s, limit: %s)\n\n",
           fmt_comma(sieve.n_base, b1, sizeof(b1)),
           fmt_comma(sieve.limit, b2, sizeof(b2)));

    /* Main loop */
    double t_start = omp_get_wtime();
    double saved_elapsed = do_resume ? ckpt.state->elapsed_seconds : 0;
    int64_t prev_prime = do_resume ? ckpt.state->last_prime : 0;
    int64_t last_ckpt_count = resume_count;
    int64_t batch_num = 0;

    printf("Processing...\n");

    while (!g_shutdown) {
        int64_t count = sieve_next_batch(&sieve);
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
            fmt_comma(ckpt.state->symbols_produced, pb, sizeof(pb));
            printf("\r  %14s / %s (%.1f%%) | %6.1fM/s | sym %s | T/P/N %ld/%ld/%ld   ",
                   fmt_comma(sieve.total_produced, b1, sizeof(b1)),
                   fmt_comma(max_primes, b2, sizeof(b2)),
                   pct, rate / 1e6, pb,
                   (long)ckpt.state->tri_transcendental,
                   (long)ckpt.state->tri_partial,
                   (long)ckpt.state->tri_noise);
            fflush(stdout);
        }

        /* Checkpoint */
        if (sieve.total_produced - last_ckpt_count >= ckpt_interval) {
            ckpt_sync(&ckpt);
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

    ckpt_sync(&ckpt);

    printf("\n\nDone: %s primes in %.2f s (%.2f hrs)\n",
           fmt_comma(sieve.total_produced, b1, sizeof(b1)),
           total_elapsed, total_elapsed / 3600.0);

    /* Report */
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

    generate_report(stdout, &ckpt);

    ckpt_close(&ckpt);
    sieve_destroy(&sieve);
    return 0;
}
