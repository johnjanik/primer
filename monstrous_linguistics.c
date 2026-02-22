/*
 * monstrous_linguistics.c — N-gram & Cryptographic Profiling of the F4 String
 *
 * Treats the F4-root-index-mod-26 encoding of prime gaps as a linguistic corpus
 * and applies classical cryptographic analysis: IC, Kasiski, Shannon entropy,
 * N-grams, keyword matching, letter frequency profiling.
 *
 * Build: gcc -O3 -march=native -Wall -fopenmp -o monstrous_linguistics monstrous_linguistics.c -lm
 * Usage: ./monstrous_linguistics --max-primes 500000000 [--output report.txt] [--ngram-max 5]
 */

#include <unistd.h>
#include "e8_common.h"

/* ================================================================
 * Configuration
 * ================================================================ */

typedef struct {
    int64_t max_primes;
    int     ngram_max;
    char    output[512];
} MLConfig;

static MLConfig parse_ml_args(int argc, char **argv)
{
    MLConfig cfg = {
        .max_primes = 2000000,
        .ngram_max  = 5,
    };
    snprintf(cfg.output, sizeof(cfg.output),
             "spiral_outputs/monstrous_linguistics_report.txt");

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-primes") && i+1 < argc)
            cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--ngram-max") && i+1 < argc)
            cfg.ngram_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i+1 < argc)
            snprintf(cfg.output, sizeof(cfg.output), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --max-primes N    Number of primes (default 2000000)\n");
            printf("  --ngram-max N     Max N-gram length (default 5)\n");
            printf("  --output FILE     Output report path\n");
            exit(0);
        }
    }
    return cfg;
}

/* ================================================================
 * Incomplete gamma / chi-square p-value
 * ================================================================ */

/* Regularized lower incomplete gamma via series expansion:
 * P(a,x) = gamma(a,x)/Gamma(a) = e^{-x} x^a sum_{n=0}^inf x^n / Gamma(a+n+1) */
static double regularized_gamma_p(double a, double x)
{
    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    double sum = 1.0 / a;
    double term = 1.0 / a;
    for (int n = 1; n < 500; n++) {
        term *= x / (a + n);
        sum += term;
        if (fabs(term) < 1e-15 * fabs(sum)) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

/* Chi-square survival p-value: P(X > chi2) with k degrees of freedom */
static double chi2_pvalue(double chi2, int k)
{
    if (chi2 <= 0.0) return 1.0;
    double a = (double)k / 2.0;
    double x = chi2 / 2.0;
    /* For very large x, P(a,x) → 1 so p-value → 0 */
    if (x > a + 50.0 * sqrt(a)) return 0.0;
    double p = regularized_gamma_p(a, x);
    if (p != p) return 0.0;  /* NaN guard */
    return 1.0 - p;  /* upper tail */
}

/* ================================================================
 * F4 string generation
 * ================================================================ */

/* Build the E8→letter lookup table (240 entries, -1 = not F4) */
static void build_e8_to_letter(const F4Lattice *f4, int8_t lut[E8_NUM_ROOTS])
{
    for (int ei = 0; ei < E8_NUM_ROOTS; ei++) {
        if (f4->e8_is_f4[ei] && f4->e8_to_f4[ei] >= 0) {
            lut[ei] = (int8_t)(f4->e8_to_f4[ei] % 26);
        } else {
            lut[ei] = -1;
        }
    }
}

/*
 * Generate the F4 letter string.  Returns malloc'd array of chars (0-25, not ASCII).
 * *out_len is set to the length of the string.
 */
static uint8_t *generate_f4_string(const int64_t *primes, int64_t n_primes,
                                   const E8Lattice *e8, const F4Lattice *f4,
                                   int64_t *out_len)
{
    int64_t n_gaps = n_primes - 1;

    int8_t lut[E8_NUM_ROOTS];
    build_e8_to_letter(f4, lut);

    /* First pass: count how many gaps map to F4 */
    int64_t count = 0;
    #pragma omp parallel for schedule(static) reduction(+:count)
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i+1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        double ng = gap / log_p;
        double tn = sqrt(fmax(ng, 0.01));
        double phase = fmod(tn / sqrt(2.0), 1.0);
        if (phase < 0.0) phase += 1.0;
        int ei = (int)(phase * 240) % 240;
        if (ei < 0) ei += 240;
        if (lut[ei] >= 0) count++;
    }

    uint8_t *text = (uint8_t *)malloc(count + 1);
    if (!text) { fprintf(stderr, "malloc failed for F4 string\n"); exit(1); }

    /* Second pass: fill in the letters (serial to preserve order) */
    int64_t pos = 0;
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i+1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        double ng = gap / log_p;
        double tn = sqrt(fmax(ng, 0.01));
        double phase = fmod(tn / sqrt(2.0), 1.0);
        if (phase < 0.0) phase += 1.0;
        int ei = (int)(phase * 240) % 240;
        if (ei < 0) ei += 240;
        if (lut[ei] >= 0) {
            text[pos++] = (uint8_t)lut[ei];
        }
    }
    text[pos] = 0;

    *out_len = pos;
    return text;
}

/* ================================================================
 * Analysis: letter frequencies, IC, entropy
 * ================================================================ */

static void compute_letter_freq(const uint8_t *text, int64_t N, int64_t freq[26])
{
    memset(freq, 0, 26 * sizeof(int64_t));
    for (int64_t i = 0; i < N; i++)
        freq[text[i]]++;
}

static double compute_ic(const int64_t freq[26], int64_t N)
{
    double num = 0;
    for (int i = 0; i < 26; i++)
        num += (double)freq[i] * (freq[i] - 1);
    double denom = (double)N * (N - 1) / 26.0;
    return (denom > 0) ? num / denom : 0.0;
}

static double compute_entropy(const int64_t freq[26], int64_t N)
{
    double H = 0;
    for (int i = 0; i < 26; i++) {
        if (freq[i] > 0) {
            double p = (double)freq[i] / N;
            H -= p * log2(p);
        }
    }
    return H;
}

static double compute_chi2_stat(const int64_t freq[26], int64_t N)
{
    double expected = (double)N / 26.0;
    double chi2 = 0;
    for (int i = 0; i < 26; i++) {
        double d = (double)freq[i] - expected;
        chi2 += d * d / expected;
    }
    return chi2;
}

/* ================================================================
 * N-gram analysis
 *
 * For n<=5, encode n-gram as base-26 integer.
 * n=2: 676,  n=3: 17576,  n=4: 456976,  n=5: 11881376
 * ================================================================ */

#define MAX_NGRAM 5

/* Encode n letters (0-25) starting at text[pos] as a base-26 int */
static inline int ngram_key(const uint8_t *text, int64_t pos, int n)
{
    int key = 0;
    for (int k = 0; k < n; k++)
        key = key * 26 + text[pos + k];
    return key;
}

/* Decode base-26 key back to string */
static void ngram_decode(int key, int n, char *buf)
{
    for (int k = n - 1; k >= 0; k--) {
        buf[k] = 'A' + (key % 26);
        key /= 26;
    }
    buf[n] = '\0';
}

/* Power of 26 */
static int pow26(int n)
{
    int r = 1;
    for (int i = 0; i < n; i++) r *= 26;
    return r;
}

typedef struct { int key; int64_t count; } NgramEntry;

static int ngram_cmp_desc(const void *a, const void *b)
{
    const NgramEntry *ea = (const NgramEntry *)a;
    const NgramEntry *eb = (const NgramEntry *)b;
    if (eb->count != ea->count) return (eb->count > ea->count) ? 1 : -1;
    return ea->key - eb->key;
}

/*
 * Count all n-grams, return top-k.
 * Also sets *out_distinct and *out_entropy.
 */
static NgramEntry *compute_ngrams(const uint8_t *text, int64_t N, int n,
                                  int top_k,
                                  int *out_distinct, double *out_entropy)
{
    int space = pow26(n);
    int64_t *counts = (int64_t *)calloc(space, sizeof(int64_t));
    if (!counts) { fprintf(stderr, "calloc failed for %d-gram\n", n); exit(1); }

    int64_t total = N - n + 1;

    /* Count (parallel for large texts) */
    if (total > 1000000 && n <= 3) {
        /* For small n-gram spaces, use thread-local copies to avoid atomics */
        int nt = omp_get_max_threads();
        int64_t *local = (int64_t *)calloc((int64_t)nt * space, sizeof(int64_t));
        if (!local) { fprintf(stderr, "calloc failed\n"); exit(1); }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int64_t *my = local + (int64_t)tid * space;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < total; i++)
                my[ngram_key(text, i, n)]++;
        }
        /* Merge */
        for (int t = 0; t < nt; t++) {
            int64_t *src = local + (int64_t)t * space;
            for (int k = 0; k < space; k++)
                counts[k] += src[k];
        }
        free(local);
    } else {
        for (int64_t i = 0; i < total; i++)
            counts[ngram_key(text, i, n)]++;
    }

    /* Distinct count + entropy */
    int distinct = 0;
    double H = 0;
    for (int k = 0; k < space; k++) {
        if (counts[k] > 0) {
            distinct++;
            double p = (double)counts[k] / total;
            H -= p * log2(p);
        }
    }
    *out_distinct = distinct;
    *out_entropy = H;

    /* Extract top-k via partial sort (heap) */
    NgramEntry *top = (NgramEntry *)calloc(top_k, sizeof(NgramEntry));
    int heap_sz = 0;
    for (int k = 0; k < space; k++) {
        if (counts[k] == 0) continue;
        NgramEntry e = { k, counts[k] };
        if (heap_sz < top_k) {
            top[heap_sz++] = e;
            /* Bubble up (min-heap on count) */
            int i = heap_sz - 1;
            while (i > 0) {
                int parent = (i - 1) / 2;
                if (top[parent].count > top[i].count) {
                    NgramEntry tmp = top[parent]; top[parent] = top[i]; top[i] = tmp;
                    i = parent;
                } else break;
            }
        } else if (e.count > top[0].count) {
            top[0] = e;
            /* Sift down */
            int i = 0;
            for (;;) {
                int l = 2*i+1, r = 2*i+2, s = i;
                if (l < heap_sz && top[l].count < top[s].count) s = l;
                if (r < heap_sz && top[r].count < top[s].count) s = r;
                if (s == i) break;
                NgramEntry tmp = top[i]; top[i] = top[s]; top[s] = tmp;
                i = s;
            }
        }
    }
    /* Sort top-k descending by count */
    qsort(top, heap_sz, sizeof(NgramEntry), ngram_cmp_desc);

    free(counts);
    return top;
}

/* ================================================================
 * Kasiski examination
 *
 * For each trigram (26^3 = 17576), store last 4 positions.
 * Compute distances between consecutive occurrences.
 * Factor distances to find candidate periods.
 * ================================================================ */

#define KASISKI_HISTORY 4   /* positions per trigram to remember */
#define KASISKI_MAX_FACTOR 300

static void kasiski_test(const uint8_t *text, int64_t N,
                         int64_t factor_counts[KASISKI_MAX_FACTOR + 1])
{
    memset(factor_counts, 0, (KASISKI_MAX_FACTOR + 1) * sizeof(int64_t));

    int space = 26 * 26 * 26;  /* 17,576 */
    /* For each trigram, store last KASISKI_HISTORY positions */
    int32_t *hist = (int32_t *)malloc(space * KASISKI_HISTORY * sizeof(int32_t));
    int16_t *hist_len = (int16_t *)calloc(space, sizeof(int16_t));
    if (!hist || !hist_len) { fprintf(stderr, "malloc failed for Kasiski\n"); exit(1); }

    /* Collect distances */
    int64_t n_dist = 0;
    int64_t dist_cap = 10000000;
    int64_t *distances = (int64_t *)malloc(dist_cap * sizeof(int64_t));
    if (!distances) { fprintf(stderr, "malloc failed for distances\n"); exit(1); }

    int64_t total = N - 2;
    for (int64_t i = 0; i < total; i++) {
        int key = text[i] * 676 + text[i+1] * 26 + text[i+2];
        int32_t *h = hist + key * KASISKI_HISTORY;
        int hlen = hist_len[key];

        /* Compute distances to previous occurrences */
        for (int j = 0; j < hlen && n_dist < dist_cap; j++) {
            distances[n_dist++] = (int64_t)(i - h[j]);
        }

        /* Push this position into history (circular) */
        if (hlen < KASISKI_HISTORY) {
            h[hlen] = (int32_t)i;
            hist_len[key] = hlen + 1;
        } else {
            /* Shift out oldest */
            for (int j = 0; j < KASISKI_HISTORY - 1; j++)
                h[j] = h[j + 1];
            h[KASISKI_HISTORY - 1] = (int32_t)i;
        }
    }

    free(hist);
    free(hist_len);

    /* Factor each distance: for each f in 2..300, count divisible distances */
    /* Parallelize over factors */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int f = 2; f <= KASISKI_MAX_FACTOR; f++) {
        int64_t cnt = 0;
        for (int64_t d = 0; d < n_dist; d++) {
            if (distances[d] % f == 0) cnt++;
        }
        factor_counts[f] = cnt;
    }

    free(distances);
}

/* ================================================================
 * Keyword search (Boyer-Moore-Horspool)
 * ================================================================ */

static int64_t count_keyword(const uint8_t *text, int64_t N,
                             const char *keyword)
{
    int klen = (int)strlen(keyword);
    if (klen > N) return 0;

    /* Convert keyword to 0-25 */
    uint8_t kw[16];
    for (int i = 0; i < klen; i++)
        kw[i] = (uint8_t)(keyword[i] - 'A');

    int64_t count = 0;
    for (int64_t i = 0; i <= N - klen; i++) {
        int match = 1;
        for (int j = 0; j < klen; j++) {
            if (text[i + j] != kw[j]) { match = 0; break; }
        }
        if (match) count++;
    }
    return count;
}

/* ================================================================
 * Report generation
 * ================================================================ */

static void print_bar(FILE *fp, int64_t count, int64_t max_count, int width)
{
    int len = (max_count > 0) ? (int)((double)count / max_count * width) : 0;
    for (int i = 0; i < width; i++)
        fputc((i < len) ? '#' : ' ', fp);
}

/* Keywords to search */
static const char *KEYWORDS[] = {"GQQ", "HOD", "JHI", "LLL", "PRIA", "SEL", "WEY"};
#define N_KEYWORDS 7

/* Exceptional Kasiski periods */
static const struct { int period; const char *label; } EXCEPTIONAL_PERIODS[] = {
    {14,  "G2 dim + 2"},
    {26,  "F4/2 + 2"},
    {48,  "|F4 roots|"},
    {52,  "F4 dim"},
    {126, "|E7 roots|"},
    {240, "|E8 roots|"},
};
#define N_EXCEPTIONAL 6

static void generate_report(FILE *fp, const uint8_t *text, int64_t N,
                            int ngram_max)
{
    char b1[32], b2[32];

    fprintf(fp, "============================================================\n");
    fprintf(fp, "  MONSTROUS LINGUISTICS REPORT\n");
    fprintf(fp, "============================================================\n");
    fprintf(fp, "Corpus: F4 root index mod 26, %s characters\n\n",
            fmt_comma(N, b1, sizeof(b1)));

    /* ---- Letter frequencies ---- */
    int64_t freq[26];
    compute_letter_freq(text, N, freq);

    int64_t max_freq = 0;
    for (int i = 0; i < 26; i++)
        if (freq[i] > max_freq) max_freq = freq[i];

    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  LETTER FREQUENCIES\n");
    fprintf(fp, "------------------------------------------------------------\n");
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "  %c: ", 'A' + i);
        print_bar(fp, freq[i], max_freq, 40);
        fprintf(fp, " %10s (%5.2f%%)\n",
                fmt_comma(freq[i], b1, sizeof(b1)),
                100.0 * freq[i] / N);
    }

    double chi2_stat = compute_chi2_stat(freq, N);
    double chi2_p = chi2_pvalue(chi2_stat, 25);
    const char *uniform_label = (chi2_p > 0.05) ? "UNIFORM" : "NOT uniform";
    fprintf(fp, "\n  Chi-square statistic: %.2f\n", chi2_stat);
    fprintf(fp, "  Chi-square p-value:  %.6f (%s)\n\n", chi2_p, uniform_label);

    /* ---- Global statistics ---- */
    double ic = compute_ic(freq, N);
    double entropy = compute_entropy(freq, N);
    double max_ent = log2(26.0);
    double util = 100.0 * entropy / max_ent;

    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  GLOBAL STATISTICS\n");
    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  Index of Coincidence: %.6f  (random=1.000, English=1.73)\n", ic);
    fprintf(fp, "  Shannon Entropy:      %.4f bits/char  (max=%.3f, utilization=%.2f%%)\n\n",
            entropy, max_ent, util);

    /* ---- Exceptional keyword matches ---- */
    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  EXCEPTIONAL KEYWORD MATCHES\n");
    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  %-10s %10s %12s %8s\n", "Keyword", "Found", "Expected", "Ratio");
    fprintf(fp, "  ---------- ---------- ------------ --------\n");
    for (int k = 0; k < N_KEYWORDS; k++) {
        int64_t found = count_keyword(text, N, KEYWORDS[k]);
        int klen = (int)strlen(KEYWORDS[k]);
        double expected = (double)(N - klen + 1);
        for (int j = 0; j < klen; j++) expected /= 26.0;
        if (expected < 1e-6) expected = 1e-6;
        double ratio = (double)found / expected;
        fprintf(fp, "  %-10s %10s %12.1f %7.2fx\n",
                KEYWORDS[k], fmt_comma(found, b1, sizeof(b1)),
                expected, ratio);
    }
    fprintf(fp, "\n");

    /* ---- N-gram analysis ---- */
    for (int n = 2; n <= ngram_max && n <= MAX_NGRAM; n++) {
        int distinct;
        double ng_entropy;
        NgramEntry *top = compute_ngrams(text, N, n, 20, &distinct, &ng_entropy);
        int possible = pow26(n);
        int64_t total = N - n + 1;

        fprintf(fp, "------------------------------------------------------------\n");
        fprintf(fp, "  TOP %d-GRAMS\n", n);
        fprintf(fp, "------------------------------------------------------------\n");
        fprintf(fp, "  Distinct: %s / %s possible  (%.2f%% coverage)\n",
                fmt_comma(distinct, b1, sizeof(b1)),
                fmt_comma(possible, b2, sizeof(b2)),
                100.0 * distinct / possible);
        fprintf(fp, "  %d-gram entropy: %.4f bits\n\n", n, ng_entropy);

        for (int i = 0; i < 20; i++) {
            if (top[i].count == 0) break;
            char gram[8];
            ngram_decode(top[i].key, n, gram);
            fprintf(fp, "    %s: %10s  (%.3f%%)\n",
                    gram, fmt_comma(top[i].count, b1, sizeof(b1)),
                    100.0 * top[i].count / total);
        }
        fprintf(fp, "\n");
        free(top);
    }

    /* ---- Bigram matrix (10x10 excerpt) ---- */
    {
        int distinct;
        double ng_ent;
        NgramEntry *unused = compute_ngrams(text, N, 2, 1, &distinct, &ng_ent);
        free(unused);

        /* Recompute bigram matrix directly */
        int64_t mat[26][26];
        memset(mat, 0, sizeof(mat));
        for (int64_t i = 0; i < N - 1; i++)
            mat[text[i]][text[i+1]]++;

        fprintf(fp, "------------------------------------------------------------\n");
        fprintf(fp, "  BIGRAM FREQUENCY MATRIX (top-left 10x10 excerpt)\n");
        fprintf(fp, "------------------------------------------------------------\n");
        fprintf(fp, "     ");
        for (int j = 0; j < 10; j++)
            fprintf(fp, "%6c", 'A' + j);
        fprintf(fp, "\n");
        for (int i = 0; i < 10; i++) {
            fprintf(fp, "  %c  ", 'A' + i);
            for (int j = 0; j < 10; j++)
                fprintf(fp, "%6ld", (long)mat[i][j]);
            fprintf(fp, "\n");
        }
        fprintf(fp, "  ... (remaining 16 rows/cols omitted)\n\n");
    }

    /* ---- Kasiski periodicity ---- */
    fprintf(fp, "------------------------------------------------------------\n");
    fprintf(fp, "  KASISKI PERIODICITY\n");
    fprintf(fp, "------------------------------------------------------------\n");

    int64_t factor_counts[KASISKI_MAX_FACTOR + 1];
    kasiski_test(text, N, factor_counts);

    /* Find top 15 factors */
    typedef struct { int factor; int64_t count; } FactorEntry;
    FactorEntry tops[15];
    int n_tops = 0;
    for (int f = 2; f <= KASISKI_MAX_FACTOR; f++) {
        if (factor_counts[f] == 0) continue;
        FactorEntry e = { f, factor_counts[f] };
        if (n_tops < 15) {
            tops[n_tops++] = e;
            /* Min-heap on count */
            int i = n_tops - 1;
            while (i > 0) {
                int parent = (i - 1) / 2;
                if (tops[parent].count > tops[i].count) {
                    FactorEntry tmp = tops[parent]; tops[parent] = tops[i]; tops[i] = tmp;
                    i = parent;
                } else break;
            }
        } else if (e.count > tops[0].count) {
            tops[0] = e;
            int i = 0;
            for (;;) {
                int l = 2*i+1, r = 2*i+2, s = i;
                if (l < 15 && tops[l].count < tops[s].count) s = l;
                if (r < 15 && tops[r].count < tops[s].count) s = r;
                if (s == i) break;
                FactorEntry tmp = tops[i]; tops[i] = tops[s]; tops[s] = tmp;
                i = s;
            }
        }
    }
    /* Sort descending */
    for (int i = 0; i < n_tops - 1; i++) {
        for (int j = i + 1; j < n_tops; j++) {
            if (tops[j].count > tops[i].count) {
                FactorEntry tmp = tops[i]; tops[i] = tops[j]; tops[j] = tmp;
            }
        }
    }

    fprintf(fp, "  %8s %14s\n", "Period", "Factor count");
    fprintf(fp, "  -------- --------------\n");
    for (int i = 0; i < n_tops; i++) {
        fprintf(fp, "  %8d %14s\n", tops[i].factor,
                fmt_comma(tops[i].count, b1, sizeof(b1)));
    }
    fprintf(fp, "\n");

    /* Check exceptional dimensions */
    fprintf(fp, "  Exceptional dimensions in Kasiski factors:\n");
    for (int i = 0; i < N_EXCEPTIONAL; i++) {
        int p = EXCEPTIONAL_PERIODS[i].period;
        int64_t cnt = (p <= KASISKI_MAX_FACTOR) ? factor_counts[p] : 0;
        const char *marker = (cnt > 0) ? "  ***" : "";
        fprintf(fp, "    %4d (%-14s): %10s hits%s\n",
                p, EXCEPTIONAL_PERIODS[i].label,
                fmt_comma(cnt, b1, sizeof(b1)), marker);
    }
    fprintf(fp, "\n");

    fprintf(fp, "============================================================\n");
    fprintf(fp, "  END OF REPORT\n");
    fprintf(fp, "============================================================\n");
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    MLConfig cfg = parse_ml_args(argc, argv);
    char b1[32], b2[32];

    printf("Monstrous Linguistics (C/OpenMP)\n");
    printf("==================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, b1, sizeof(b1)));
    printf("N-gram max : %d\n", cfg.ngram_max);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("==================================================\n\n");

    /* ---- Step 1: Primes ---- */
    printf("Generating primes via sieve...\n");
    tic();
    int64_t n_primes;
    int64_t *primes = sieve_primes(cfg.max_primes, &n_primes);
    printf("  %s primes in %.2fs (range [%ld, %ld])\n",
           fmt_comma(n_primes, b1, sizeof(b1)), toc(),
           (long)primes[0], (long)primes[n_primes - 1]);

    /* ---- Step 2: Lattices ---- */
    printf("Initializing E8 (240 roots) + F4 (48 roots)...\n");
    tic();
    E8Lattice e8;   e8_init(&e8);
    F4Lattice f4;   f4_init(&f4, &e8);
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 3: F4 string ---- */
    printf("Generating F4 string (Method 14 pipeline)...\n");
    tic();
    int64_t f4_len;
    uint8_t *text = generate_f4_string(primes, n_primes, &e8, &f4, &f4_len);
    int64_t n_gaps = n_primes - 1;
    double f4_pct = 100.0 * f4_len / n_gaps;
    printf("  %s characters from %s gaps (%.1f%% F4-mapped) in %.2fs\n",
           fmt_comma(f4_len, b1, sizeof(b1)),
           fmt_comma(n_gaps, b2, sizeof(b2)),
           f4_pct, toc());

    free(primes);

    /* ---- Step 4: Analysis + Report ---- */
    printf("Running analysis (ngram_max=%d)...\n", cfg.ngram_max);
    tic();

    /* Build report in memory (write to file + stdout) */
    /* Open output file */
    FILE *outfp = fopen(cfg.output, "w");
    if (!outfp) {
        fprintf(stderr, "Warning: cannot open %s, writing to stdout only\n", cfg.output);
        outfp = NULL;
    }

    /* Generate report to a temp file, then replay to stdout + output */
    char tmpname[] = "/tmp/ml_report_XXXXXX";
    int tmpfd = mkstemp(tmpname);
    if (tmpfd < 0) { fprintf(stderr, "mkstemp failed\n"); exit(1); }
    FILE *tmpfp = fdopen(tmpfd, "w+");

    generate_report(tmpfp, text, f4_len, cfg.ngram_max);
    printf("  Analysis done in %.2fs\n\n", toc());

    /* Replay to stdout + file */
    fseek(tmpfp, 0, SEEK_SET);
    char line[4096];
    while (fgets(line, sizeof(line), tmpfp)) {
        fputs(line, stdout);
        if (outfp) fputs(line, outfp);
    }

    fclose(tmpfp);
    unlink(tmpname);
    if (outfp) {
        fclose(outfp);
        printf("\nReport saved to %s\n", cfg.output);
    }

    free(text);
    return 0;
}
