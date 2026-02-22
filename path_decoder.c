/*
 * path_decoder.c — Crystalline Path Decoder
 *
 * Extracts the top-K crystalline vertices (highest triplet coherence)
 * from prime gap E8 assignments, connects them in prime-index order
 * (Hamiltonian path), and decodes the sequence of E8/F4/G2 root
 * transitions along the path.
 *
 * Outputs:
 *   path_vertices.csv — per-vertex lattice data
 *   path_edges.csv    — per-edge transition data
 *   path_report.txt   — summary statistics and transition matrices
 *
 * Build:
 *   gcc -O3 -march=native -Wall -fopenmp -o path_decoder path_decoder.c -lm
 *
 * Usage:
 *   ./path_decoder --max-primes 100000000 --vertices 500 --output-dir spiral_outputs
 */

#include "e8_common.h"
#include <sys/stat.h>
#include <time.h>

/* ================================================================
 * Configuration
 * ================================================================ */

typedef struct {
    int64_t max_primes;
    int     n_vertices;
    int     n_null_perms;
    char    output_dir[512];
} PDConfig;

static PDConfig pd_parse_args(int argc, char **argv)
{
    PDConfig cfg = {
        .max_primes   = 100000000,
        .n_vertices   = 500,
        .n_null_perms = 1000,
    };
    strncpy(cfg.output_dir, "spiral_outputs", sizeof(cfg.output_dir) - 1);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-primes") && i+1 < argc) cfg.max_primes = atol(argv[++i]);
        else if (!strcmp(argv[i], "--vertices") && i+1 < argc) cfg.n_vertices = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--null-perms") && i+1 < argc) cfg.n_null_perms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc)
            strncpy(cfg.output_dir, argv[++i], sizeof(cfg.output_dir) - 1);
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); exit(1); }
    }
    return cfg;
}

/* ================================================================
 * Vertex / Edge structures
 * ================================================================ */

typedef struct {
    int64_t prime_idx;       /* index in the prime array */
    int64_t prime;           /* the actual prime */
    double  coherence;       /* triplet coherence κ */
    int     e8_root;         /* E8 root index 0-239 */
    int     e8_type;         /* 0 = Type I (integer), 1 = Type II (half-int) */
    int     f4_root;         /* F4 sublattice index (-1 if not member) */
    int     g2_root;         /* G2 sublattice index (-1 if not member) */
    int     is_f4;           /* 1 if E8 root is F4 member */
    int     is_g2;           /* 1 if E8 root is G2 member */
    double  jordan_trace;    /* sum of E8 root coordinates */
    int32_t ulam_x, ulam_y;  /* Ulam spiral coordinates */
    double  norm_gap;        /* normalized gap at this vertex */
} PathVertex;

typedef struct {
    int     prime_gap;       /* gap in prime indices */
    double  ulam_dx, ulam_dy;
    double  ulam_dist;       /* Euclidean distance in Ulam plane */
    double  angle;           /* atan2(dy, dx) in degrees */
    double  turning_angle;   /* angle[i] - angle[i-1] in degrees */
    int     e8_from, e8_to;  /* E8 root indices at endpoints */
    int     same_e8;         /* 1 if same E8 root */
    int     same_f4;         /* 1 if both F4 members with same index */
    double  dot_product;     /* inner product of E8 root vectors */
} PathEdge;

/* ================================================================
 * Min-heap for top-K extraction
 * ================================================================ */

typedef struct { int64_t idx; double score; } HeapEntry;

static void heap_sift_down(HeapEntry *h, int size, int k)
{
    for (;;) {
        int left = 2*k+1, right = 2*k+2, smallest = k;
        if (left < size && h[left].score < h[smallest].score) smallest = left;
        if (right < size && h[right].score < h[smallest].score) smallest = right;
        if (smallest == k) break;
        HeapEntry tmp = h[k]; h[k] = h[smallest]; h[smallest] = tmp;
        k = smallest;
    }
}

static void heap_bubble_up(HeapEntry *h, int k)
{
    while (k > 0) {
        int parent = (k - 1) / 2;
        if (h[parent].score > h[k].score) {
            HeapEntry tmp = h[parent]; h[parent] = h[k]; h[k] = tmp;
            k = parent;
        } else break;
    }
}

/* ================================================================
 * Comparison for qsort (by idx ascending)
 * ================================================================ */

static int cmp_heap_by_idx(const void *a, const void *b)
{
    int64_t va = ((const HeapEntry *)a)->idx;
    int64_t vb = ((const HeapEntry *)b)->idx;
    return (va > vb) - (va < vb);
}

/* ================================================================
 * E8 root type: Type I (integer coords, ±e_i±e_j) = 0-111
 *               Type II (half-integer coords) = 112-239
 * ================================================================ */

static inline int e8_root_type(const E8Lattice *e8, int root_idx)
{
    /* Type I has exactly 2 nonzero integer coords */
    int nonzero = 0;
    for (int d = 0; d < E8_DIM; d++)
        if (fabs(e8->roots[root_idx][d]) > 0.01) nonzero++;
    return (nonzero == 2) ? 0 : 1;  /* 0=Type I, 1=Type II */
}

/* ================================================================
 * Compute dot product of two E8 root vectors
 * ================================================================ */

static inline double e8_dot(const E8Lattice *e8, int r1, int r2)
{
    double d = 0;
    for (int i = 0; i < E8_DIM; i++)
        d += e8->roots[r1][i] * e8->roots[r2][i];
    return d;
}

/* ================================================================
 * Null model: compute statistics on a permuted path
 * ================================================================ */

typedef struct {
    double mean_dot;       /* mean dot product along path */
    double same_e8_frac;   /* fraction of edges with same E8 root */
    double same_f4_frac;   /* fraction with same F4 */
    double angle_ks;       /* KS statistic for angle uniformity */
    double mean_prime_gap; /* mean prime-index gap */
} PathStats;

/* Angle wrapping to [0, 360) */
static inline double wrap360(double a)
{
    a = fmod(a, 360.0);
    if (a < 0) a += 360.0;
    return a;
}

/* Comparator for qsort on doubles */
static int cmp_double(const void *a, const void *b)
{
    double va = *(const double *)a;
    double vb = *(const double *)b;
    return (va > vb) - (va < vb);
}

/* KS statistic for uniform distribution on [0, 360) */
static double ks_uniform_360(const double *angles, int n)
{
    /* Sort a copy — O(N log N) via qsort */
    double *sorted = (double *)malloc(n * sizeof(double));
    memcpy(sorted, angles, n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);

    double max_d = 0;
    for (int i = 0; i < n; i++) {
        double expected = (i + 1.0) / n;
        double empirical = sorted[i] / 360.0;
        double d = fabs(empirical - expected);
        if (d > max_d) max_d = d;
        d = fabs(empirical - (double)i / n);
        if (d > max_d) max_d = d;
    }
    free(sorted);
    return max_d;
}

/* Compute statistics for a given vertex ordering */
static PathStats compute_path_stats(const PathVertex *verts, int n_verts,
                                     const int *order,
                                     const E8Lattice *e8,
                                     const F4Lattice *f4)
{
    PathStats s = {0};
    int n_edges = n_verts - 1;
    if (n_edges <= 0) return s;

    double dot_sum = 0;
    int same_e8 = 0, same_f4 = 0;
    double gap_sum = 0;
    double *angles = (double *)malloc(n_edges * sizeof(double));

    for (int i = 0; i < n_edges; i++) {
        const PathVertex *va = &verts[order[i]];
        const PathVertex *vb = &verts[order[i+1]];

        /* Dot product */
        double dp = e8_dot(e8, va->e8_root, vb->e8_root);
        dot_sum += dp;

        /* Same roots */
        if (va->e8_root == vb->e8_root) same_e8++;
        if (va->is_f4 && vb->is_f4 && va->f4_root == vb->f4_root) same_f4++;

        /* Angle in Ulam plane */
        double dx = (double)(vb->ulam_x - va->ulam_x);
        double dy = (double)(vb->ulam_y - va->ulam_y);
        angles[i] = wrap360(atan2(dy, dx) * 180.0 / M_PI);

        /* Prime-index gap */
        gap_sum += fabs((double)(vb->prime_idx - va->prime_idx));
    }

    s.mean_dot = dot_sum / n_edges;
    s.same_e8_frac = (double)same_e8 / n_edges;
    s.same_f4_frac = (double)same_f4 / n_edges;
    s.angle_ks = ks_uniform_360(angles, n_edges);
    s.mean_prime_gap = gap_sum / n_edges;

    free(angles);
    return s;
}

/* Fisher-Yates shuffle */
static void shuffle(int *arr, int n, unsigned int *seed)
{
    for (int i = n - 1; i > 0; i--) {
        int j = (int)((unsigned long)rand_r(seed) % (i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    PDConfig cfg = pd_parse_args(argc, argv);
    char b1[32], b2[32];

    printf("================================================================\n");
    printf("  Crystalline Path Decoder\n");
    printf("================================================================\n");
    printf("Max primes : %s\n", fmt_comma(cfg.max_primes, b1, sizeof(b1)));
    printf("Vertices   : %d\n", cfg.n_vertices);
    printf("Null perms : %d\n", cfg.n_null_perms);
    printf("Output dir : %s\n", cfg.output_dir);
    printf("Threads    : %d\n", omp_get_max_threads());
    printf("================================================================\n\n");

    /* ---- Step 1: Generate primes ---- */
    printf("Step 1: Generating primes...\n");
    tic();
    int64_t n_primes;
    int64_t *primes = sieve_primes(cfg.max_primes, &n_primes);
    printf("  %s primes in %.2fs (range: 2 to %s)\n",
           fmt_comma(n_primes, b1, sizeof(b1)), toc(),
           fmt_comma(primes[n_primes - 1], b2, sizeof(b2)));

    int64_t n_gaps = n_primes - 1;

    /* ---- Step 2: Initialize lattices ---- */
    printf("\nStep 2: Initializing lattices...\n");
    tic();
    E8Lattice e8;  e8_init(&e8);
    F4Lattice f4;  f4_init(&f4, &e8);
    G2Lattice g2;  g2_init(&g2, &e8);
    printf("  E8: %d roots, F4: %d roots, G2: %d roots (%.2fs)\n",
           E8_NUM_ROOTS, F4_NUM_ROOTS, G2_NUM_ROOTS, toc());

    /* ---- Step 3: E8 assignments + normalized gaps ---- */
    printf("\nStep 3: Computing E8 assignments...\n");
    tic();
    int *e8_assignments = (int *)malloc(n_gaps * sizeof(int));
    double *norm_gaps = (double *)malloc(n_gaps * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_gaps; i++) {
        double gap = (double)(primes[i + 1] - primes[i]);
        double log_p = log((double)primes[i]);
        if (log_p < 1.0) log_p = 1.0;
        norm_gaps[i] = gap / log_p;
        e8_assignments[i] = e8_assign_root(&e8, norm_gaps[i]);
    }
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 4: Triplet coherence ---- */
    printf("\nStep 4: Computing triplet coherence...\n");
    tic();
    float *coherence = (float *)calloc(n_gaps, sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n_gaps - 1; i++) {
        int r0 = e8_assignments[i - 1];
        int r1 = e8_assignments[i];
        int r2 = e8_assignments[i + 1];
        double sum_sq = 0;
        for (int d = 0; d < E8_DIM; d++) {
            double s = e8.roots[r0][d] + e8.roots[r1][d] + e8.roots[r2][d];
            sum_sq += s * s;
        }
        coherence[i] = (float)(sum_sq / 6.0);
    }
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 5: Extract top-K vertices ---- */
    printf("\nStep 5: Extracting top-%d crystalline vertices...\n", cfg.n_vertices);
    tic();

    int heap_cap = cfg.n_vertices;
    if (heap_cap > n_gaps) heap_cap = (int)n_gaps;
    HeapEntry *heap = (HeapEntry *)malloc(heap_cap * sizeof(HeapEntry));
    int heap_size = 0;

    for (int64_t i = 1; i < n_gaps - 1; i++) {
        double score = (double)coherence[i];
        if (score <= 0.0) continue;

        if (heap_size < heap_cap) {
            heap[heap_size].idx = i;
            heap[heap_size].score = score;
            heap_size++;
            heap_bubble_up(heap, heap_size - 1);
        } else if (score > heap[0].score) {
            heap[0].idx = i;
            heap[0].score = score;
            heap_sift_down(heap, heap_size, 0);
        }
    }

    int n_verts = heap_size < cfg.n_vertices ? heap_size : cfg.n_vertices;

    /* Sort by prime index (ascending) */
    qsort(heap, n_verts, sizeof(HeapEntry), cmp_heap_by_idx);

    printf("  Extracted %d vertices, min coherence = %.4f, max = %.4f\n",
           n_verts, heap[0].score, heap[n_verts-1].score);

    /* Find actual min/max coherence in the selected set */
    double coh_min = 1e30, coh_max = -1e30;
    for (int i = 0; i < n_verts; i++) {
        if (heap[i].score < coh_min) coh_min = heap[i].score;
        if (heap[i].score > coh_max) coh_max = heap[i].score;
    }
    printf("  Coherence range: [%.4f, %.4f]\n", coh_min, coh_max);
    printf("  Done in %.2fs\n", toc());

    /* ---- Step 6: Build vertex records ---- */
    printf("\nStep 6: Building vertex + edge records...\n");
    tic();

    PathVertex *verts = (PathVertex *)malloc(n_verts * sizeof(PathVertex));

    for (int v = 0; v < n_verts; v++) {
        int64_t gi = heap[v].idx;   /* gap index */
        int64_t pi = gi + 1;       /* prime index (prime[pi] inherits gap[gi]) */
        int e8r = e8_assignments[gi];

        verts[v].prime_idx    = pi;
        verts[v].prime        = primes[pi];
        verts[v].coherence    = heap[v].score;
        verts[v].e8_root      = e8r;
        verts[v].e8_type      = e8_root_type(&e8, e8r);
        verts[v].f4_root      = f4.e8_to_f4[e8r];
        verts[v].g2_root      = g2.e8_to_g2[e8r];
        verts[v].is_f4        = f4.e8_is_f4[e8r];
        verts[v].is_g2        = g2.e8_is_g2[e8r];

        /* Jordan trace = sum of E8 root coords */
        double jt = 0;
        for (int d = 0; d < E8_DIM; d++) jt += e8.roots[e8r][d];
        verts[v].jordan_trace = jt;

        ulam_coord(primes[pi], &verts[v].ulam_x, &verts[v].ulam_y);
        verts[v].norm_gap = norm_gaps[gi];
    }

    /* Build edge records */
    int n_edges = n_verts - 1;
    PathEdge *edges = NULL;
    if (n_edges > 0) {
        edges = (PathEdge *)malloc(n_edges * sizeof(PathEdge));

        for (int i = 0; i < n_edges; i++) {
            PathVertex *va = &verts[i];
            PathVertex *vb = &verts[i+1];

            edges[i].prime_gap = (int)(vb->prime_idx - va->prime_idx);
            edges[i].ulam_dx   = (double)(vb->ulam_x - va->ulam_x);
            edges[i].ulam_dy   = (double)(vb->ulam_y - va->ulam_y);
            edges[i].ulam_dist = sqrt(edges[i].ulam_dx * edges[i].ulam_dx +
                                      edges[i].ulam_dy * edges[i].ulam_dy);
            edges[i].angle     = atan2(edges[i].ulam_dy, edges[i].ulam_dx) * 180.0 / M_PI;
            if (edges[i].angle < 0) edges[i].angle += 360.0;

            /* Turning angle */
            if (i > 0) {
                double da = edges[i].angle - edges[i-1].angle;
                /* Wrap to [-180, 180] */
                while (da > 180.0)  da -= 360.0;
                while (da < -180.0) da += 360.0;
                edges[i].turning_angle = da;
            } else {
                edges[i].turning_angle = 0.0;
            }

            edges[i].e8_from    = va->e8_root;
            edges[i].e8_to      = vb->e8_root;
            edges[i].same_e8    = (va->e8_root == vb->e8_root) ? 1 : 0;
            edges[i].same_f4    = (va->is_f4 && vb->is_f4 && va->f4_root == vb->f4_root) ? 1 : 0;
            edges[i].dot_product = e8_dot(&e8, va->e8_root, vb->e8_root);
        }
    }
    printf("  %d vertices, %d edges built in %.2fs\n", n_verts, n_edges, toc());

    /* ---- Step 7: Write CSVs ---- */
    printf("\nStep 7: Writing output files...\n");
    tic();

    /* Create output directory (recursive: walk path components) */
    {
        char tmp[512];
        strncpy(tmp, cfg.output_dir, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        for (char *p = tmp + 1; *p; p++) {
            if (*p == '/') {
                *p = '\0';
                mkdir(tmp, 0755);
                *p = '/';
            }
        }
        mkdir(tmp, 0755);
    }

    /* path_vertices.csv */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/path_vertices.csv", cfg.output_dir);
        FILE *fp = fopen(path, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
        fprintf(fp, "seq,prime_idx,prime,coherence,e8_root,e8_type,f4_root,is_f4,g2_root,is_g2,"
                    "jordan_trace,ulam_x,ulam_y,norm_gap\n");
        for (int v = 0; v < n_verts; v++) {
            PathVertex *pv = &verts[v];
            fprintf(fp, "%d,%ld,%ld,%.6f,%d,%s,%d,%d,%d,%d,%.4f,%d,%d,%.6f\n",
                    v, (long)pv->prime_idx, (long)pv->prime, pv->coherence,
                    pv->e8_root, pv->e8_type == 0 ? "I" : "II",
                    pv->f4_root, pv->is_f4,
                    pv->g2_root, pv->is_g2,
                    pv->jordan_trace,
                    pv->ulam_x, pv->ulam_y, pv->norm_gap);
        }
        fclose(fp);
        printf("  Written: %s\n", path);
    }

    /* path_edges.csv */
    if (n_edges > 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/path_edges.csv", cfg.output_dir);
        FILE *fp = fopen(path, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
        fprintf(fp, "edge,prime_gap,ulam_dx,ulam_dy,ulam_dist,angle,turning_angle,"
                    "e8_from,e8_to,same_e8,same_f4,dot_product\n");
        for (int i = 0; i < n_edges; i++) {
            PathEdge *pe = &edges[i];
            fprintf(fp, "%d,%d,%.1f,%.1f,%.4f,%.2f,%.2f,%d,%d,%d,%d,%.4f\n",
                    i, pe->prime_gap,
                    pe->ulam_dx, pe->ulam_dy, pe->ulam_dist,
                    pe->angle, pe->turning_angle,
                    pe->e8_from, pe->e8_to,
                    pe->same_e8, pe->same_f4, pe->dot_product);
        }
        fclose(fp);
        printf("  Written: %s\n", path);
    }

    /* ---- Step 8: Analysis report ---- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/path_report.txt", cfg.output_dir);
        FILE *fp = fopen(path, "w");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

        fprintf(fp, "================================================================\n");
        fprintf(fp, "  Crystalline Path Decoder — Analysis Report\n");
        fprintf(fp, "================================================================\n");
        fprintf(fp, "Primes:    %s\n", fmt_comma(n_primes, b1, sizeof(b1)));
        fprintf(fp, "Vertices:  %d (top by triplet coherence)\n", n_verts);
        fprintf(fp, "Edges:     %d (consecutive in prime-index order)\n", n_edges);
        fprintf(fp, "Coherence: [%.4f, %.4f]\n\n", coh_min, coh_max);

        /* ---- A. E8 Root Distribution ---- */
        fprintf(fp, "================================================================\n");
        fprintf(fp, "  A. E8 Root Distribution at Vertices\n");
        fprintf(fp, "================================================================\n");

        int root_hist[E8_NUM_ROOTS];
        memset(root_hist, 0, sizeof(root_hist));
        for (int v = 0; v < n_verts; v++) root_hist[verts[v].e8_root]++;

        /* Count distinct roots used */
        int distinct_roots = 0;
        for (int r = 0; r < E8_NUM_ROOTS; r++)
            if (root_hist[r] > 0) distinct_roots++;

        fprintf(fp, "Distinct E8 roots used: %d / %d\n", distinct_roots, E8_NUM_ROOTS);

        /* Type I vs Type II */
        int n_type1 = 0, n_type2 = 0;
        for (int v = 0; v < n_verts; v++) {
            if (verts[v].e8_type == 0) n_type1++;
            else n_type2++;
        }
        fprintf(fp, "Type I  (±eᵢ±eⱼ): %d (%.1f%%)\n", n_type1, 100.0 * n_type1 / n_verts);
        fprintf(fp, "Type II (½-int):   %d (%.1f%%)\n", n_type2, 100.0 * n_type2 / n_verts);
        fprintf(fp, "  Expected (uniform): Type I 46.7%%, Type II 53.3%%\n");

        /* Top 20 most frequent roots */
        fprintf(fp, "\nTop 20 most frequent E8 roots:\n");
        fprintf(fp, "  %-6s  %-6s  %-5s  %-40s\n", "Root", "Count", "Type", "Coordinates");
        int sorted_roots[E8_NUM_ROOTS];
        for (int r = 0; r < E8_NUM_ROOTS; r++) sorted_roots[r] = r;
        for (int i = 0; i < E8_NUM_ROOTS - 1; i++)
            for (int j = i + 1; j < E8_NUM_ROOTS; j++)
                if (root_hist[sorted_roots[j]] > root_hist[sorted_roots[i]]) {
                    int t = sorted_roots[i]; sorted_roots[i] = sorted_roots[j]; sorted_roots[j] = t;
                }
        for (int i = 0; i < 20 && i < E8_NUM_ROOTS; i++) {
            int r = sorted_roots[i];
            if (root_hist[r] == 0) break;
            char coord_str[80];
            int pos = 0;
            pos += snprintf(coord_str + pos, sizeof(coord_str) - pos, "(");
            for (int d = 0; d < E8_DIM; d++) {
                if (d > 0) pos += snprintf(coord_str + pos, sizeof(coord_str) - pos, ",");
                if (fabs(e8.roots[r][d] - 0.5) < 0.01)
                    pos += snprintf(coord_str + pos, sizeof(coord_str) - pos, "½");
                else if (fabs(e8.roots[r][d] + 0.5) < 0.01)
                    pos += snprintf(coord_str + pos, sizeof(coord_str) - pos, "-½");
                else
                    pos += snprintf(coord_str + pos, sizeof(coord_str) - pos, "%.0f", e8.roots[r][d]);
            }
            snprintf(coord_str + pos, sizeof(coord_str) - pos, ")");
            fprintf(fp, "  %-6d  %-6d  %-5s  %s\n",
                    r, root_hist[r],
                    e8_root_type(&e8, r) == 0 ? "I" : "II",
                    coord_str);
        }

        /* Chi-squared test against uniform */
        double expected = (double)n_verts / E8_NUM_ROOTS;
        double chi2 = 0;
        for (int r = 0; r < E8_NUM_ROOTS; r++) {
            double diff = root_hist[r] - expected;
            chi2 += diff * diff / expected;
        }
        fprintf(fp, "\nχ² vs uniform: %.2f (df=%d, expected ~%d for uniform)\n",
                chi2, E8_NUM_ROOTS - 1, E8_NUM_ROOTS - 1);

        /* ---- A2. E8 Root Transition Matrix (condensed) ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  A2. E8 Root Transitions (Top 20)\n");
        fprintf(fp, "================================================================\n");

        /* Count transitions */
        typedef struct { int from, to, count; } TransCount;
        /* Use a hash or just brute-force for 240×240 */
        int trans_matrix_flat[240 * 240];
        memset(trans_matrix_flat, 0, sizeof(trans_matrix_flat));
        for (int i = 0; i < n_edges; i++) {
            int fr = edges[i].e8_from;
            int to = edges[i].e8_to;
            trans_matrix_flat[fr * 240 + to]++;
        }

        /* Find top 20 transitions */
        TransCount top_trans[20];
        memset(top_trans, 0, sizeof(top_trans));
        for (int fr = 0; fr < 240; fr++) {
            for (int to = 0; to < 240; to++) {
                int c = trans_matrix_flat[fr * 240 + to];
                if (c == 0) continue;
                /* Insert into top-20 */
                for (int t = 0; t < 20; t++) {
                    if (c > top_trans[t].count) {
                        /* Shift down */
                        for (int s = 19; s > t; s--) top_trans[s] = top_trans[s-1];
                        top_trans[t] = (TransCount){fr, to, c};
                        break;
                    }
                }
            }
        }

        fprintf(fp, "%-8s  %-8s  %-6s  %-8s\n", "From", "To", "Count", "DotProd");
        for (int t = 0; t < 20; t++) {
            if (top_trans[t].count == 0) break;
            double dp = e8_dot(&e8, top_trans[t].from, top_trans[t].to);
            fprintf(fp, "%-8d  %-8d  %-6d  %+.1f\n",
                    top_trans[t].from, top_trans[t].to,
                    top_trans[t].count, dp);
        }

        int same_count = 0;
        for (int i = 0; i < n_edges; i++)
            if (edges[i].same_e8) same_count++;
        fprintf(fp, "\nSame E8 root on both ends: %d / %d (%.1f%%)\n",
                same_count, n_edges, 100.0 * same_count / n_edges);
        fprintf(fp, "  Expected (uniform random): %.1f%%\n", 100.0 / E8_NUM_ROOTS);

        /* ---- B. Sublattice Transition Analysis ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  B. Sublattice Membership\n");
        fprintf(fp, "================================================================\n");

        int n_f4_verts = 0, n_g2_verts = 0;
        for (int v = 0; v < n_verts; v++) {
            if (verts[v].is_f4) n_f4_verts++;
            if (verts[v].is_g2) n_g2_verts++;
        }
        fprintf(fp, "F4 members: %d / %d (%.1f%%)\n", n_f4_verts, n_verts, 100.0 * n_f4_verts / n_verts);
        fprintf(fp, "G2 members: %d / %d (%.1f%%)\n", n_g2_verts, n_verts, 100.0 * n_g2_verts / n_verts);

        /* G2 12×12 transition matrix */
        fprintf(fp, "\nG2 Transition Matrix (12x12):\n");
        int g2_trans[G2_NUM_ROOTS][G2_NUM_ROOTS];
        memset(g2_trans, 0, sizeof(g2_trans));
        int g2_edge_count = 0;
        for (int i = 0; i < n_edges; i++) {
            int g_from = verts[i].g2_root;
            int g_to   = verts[i+1].g2_root;
            if (g_from >= 0 && g_from < G2_NUM_ROOTS &&
                g_to >= 0 && g_to < G2_NUM_ROOTS &&
                verts[i].is_g2 && verts[i+1].is_g2) {
                g2_trans[g_from][g_to]++;
                g2_edge_count++;
            }
        }

        fprintf(fp, "  G2-G2 edges: %d\n", g2_edge_count);
        if (g2_edge_count > 0) {
            fprintf(fp, "     ");
            for (int j = 0; j < G2_NUM_ROOTS; j++) fprintf(fp, "%4d", j);
            fprintf(fp, "\n");
            for (int i = 0; i < G2_NUM_ROOTS; i++) {
                fprintf(fp, "  %2d:", i);
                for (int j = 0; j < G2_NUM_ROOTS; j++)
                    fprintf(fp, "%4d", g2_trans[i][j]);
                fprintf(fp, "\n");
            }
        }

        /* F4 condensed: top 10 transitions */
        fprintf(fp, "\nF4 Transitions (top 10):\n");
        int f4_trans_flat[F4_NUM_ROOTS * F4_NUM_ROOTS];
        memset(f4_trans_flat, 0, sizeof(f4_trans_flat));
        for (int i = 0; i < n_edges; i++) {
            int f_from = verts[i].f4_root;
            int f_to   = verts[i+1].f4_root;
            if (f_from >= 0 && f_from < F4_NUM_ROOTS &&
                f_to >= 0 && f_to < F4_NUM_ROOTS &&
                verts[i].is_f4 && verts[i+1].is_f4) {
                f4_trans_flat[f_from * F4_NUM_ROOTS + f_to]++;
            }
        }
        TransCount f4_top[10];
        memset(f4_top, 0, sizeof(f4_top));
        for (int fr = 0; fr < F4_NUM_ROOTS; fr++) {
            for (int to = 0; to < F4_NUM_ROOTS; to++) {
                int c = f4_trans_flat[fr * F4_NUM_ROOTS + to];
                if (c == 0) continue;
                for (int t = 0; t < 10; t++) {
                    if (c > f4_top[t].count) {
                        for (int s = 9; s > t; s--) f4_top[s] = f4_top[s-1];
                        f4_top[t] = (TransCount){fr, to, c};
                        break;
                    }
                }
            }
        }
        fprintf(fp, "  %-8s  %-8s  %-6s\n", "From", "To", "Count");
        for (int t = 0; t < 10; t++) {
            if (f4_top[t].count == 0) break;
            fprintf(fp, "  %-8d  %-8d  %-6d\n", f4_top[t].from, f4_top[t].to, f4_top[t].count);
        }

        /* ---- C. Angular Structure ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  C. Angular Structure of Path in Ulam Plane\n");
        fprintf(fp, "================================================================\n");

        if (n_edges > 0) {
            /* Edge angle histogram (36 bins, 10° each) */
            int angle_hist[36];
            memset(angle_hist, 0, sizeof(angle_hist));
            for (int i = 0; i < n_edges; i++) {
                int bin = (int)(edges[i].angle / 10.0);
                if (bin < 0) bin = 0;
                if (bin > 35) bin = 35;
                angle_hist[bin]++;
            }
            fprintf(fp, "Edge angle histogram (10° bins):\n");
            for (int b = 0; b < 36; b++) {
                if (angle_hist[b] > 0)
                    fprintf(fp, "  [%3d°-%3d°): %d\n", b*10, (b+1)*10, angle_hist[b]);
            }

            /* KS test for uniformity */
            double *edge_angles = (double *)malloc(n_edges * sizeof(double));
            for (int i = 0; i < n_edges; i++) edge_angles[i] = edges[i].angle;
            double ks = ks_uniform_360(edge_angles, n_edges);
            fprintf(fp, "\nKS statistic vs uniform: %.4f\n", ks);
            fprintf(fp, "  Critical values: 0.05 level = %.4f, 0.01 level = %.4f\n",
                    1.36 / sqrt((double)n_edges), 1.63 / sqrt((double)n_edges));
            free(edge_angles);

            /* Turning angle histogram */
            fprintf(fp, "\nTurning angle histogram (10° bins, [-180°, 180°)):\n");
            int turn_hist[36];
            memset(turn_hist, 0, sizeof(turn_hist));
            for (int i = 1; i < n_edges; i++) {
                int bin = (int)((edges[i].turning_angle + 180.0) / 10.0);
                if (bin < 0) bin = 0;
                if (bin > 35) bin = 35;
                turn_hist[bin]++;
            }
            for (int b = 0; b < 36; b++) {
                if (turn_hist[b] > 0)
                    fprintf(fp, "  [%+4d°,%+4d°): %d\n",
                            b*10 - 180, (b+1)*10 - 180, turn_hist[b]);
            }
        }

        /* ---- D. Spacing Structure ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  D. Spacing Structure (Prime-Index Gaps)\n");
        fprintf(fp, "================================================================\n");

        if (n_edges > 0) {
            double gap_sum = 0, gap_sq_sum = 0;
            int gap_min = edges[0].prime_gap, gap_max = edges[0].prime_gap;
            for (int i = 0; i < n_edges; i++) {
                double g = edges[i].prime_gap;
                gap_sum += g;
                gap_sq_sum += g * g;
                if (edges[i].prime_gap < gap_min) gap_min = edges[i].prime_gap;
                if (edges[i].prime_gap > gap_max) gap_max = edges[i].prime_gap;
            }
            double gap_mean = gap_sum / n_edges;
            double gap_std = sqrt(gap_sq_sum / n_edges - gap_mean * gap_mean);
            fprintf(fp, "Mean prime-index gap:  %.1f\n", gap_mean);
            fprintf(fp, "Std:                   %.1f\n", gap_std);
            fprintf(fp, "Min:                   %d\n", gap_min);
            fprintf(fp, "Max:                   %d\n", gap_max);
            fprintf(fp, "Expected (uniform):    %.1f (if %d vertices among %s gaps)\n",
                    (double)n_gaps / n_verts, n_verts, fmt_comma(n_gaps, b1, sizeof(b1)));

            /* Ulam distance stats */
            double ud_sum = 0, ud_max = 0;
            for (int i = 0; i < n_edges; i++) {
                ud_sum += edges[i].ulam_dist;
                if (edges[i].ulam_dist > ud_max) ud_max = edges[i].ulam_dist;
            }
            fprintf(fp, "\nMean Ulam distance:    %.1f\n", ud_sum / n_edges);
            fprintf(fp, "Max Ulam distance:     %.1f\n", ud_max);
        }

        /* ---- E. Root Autocorrelation ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  E. E8 Root Autocorrelation (lag 1-10)\n");
        fprintf(fp, "================================================================\n");

        fprintf(fp, "Autocorrelation of e8_root index sequence:\n");
        fprintf(fp, "  (1 = identical sequence, 0 = uncorrelated)\n\n");

        if (n_verts > 10) {
            /* Compute mean root index */
            double root_mean = 0;
            for (int v = 0; v < n_verts; v++) root_mean += verts[v].e8_root;
            root_mean /= n_verts;

            /* Variance */
            double var = 0;
            for (int v = 0; v < n_verts; v++) {
                double d = verts[v].e8_root - root_mean;
                var += d * d;
            }
            var /= n_verts;

            fprintf(fp, "  %-5s  %-10s  %-10s\n", "Lag", "Autocorr", "Same-root%");
            for (int lag = 1; lag <= 10 && lag < n_verts; lag++) {
                double cov = 0;
                int same_at_lag = 0;
                int pairs = n_verts - lag;
                for (int v = 0; v < pairs; v++) {
                    double d1 = verts[v].e8_root - root_mean;
                    double d2 = verts[v + lag].e8_root - root_mean;
                    cov += d1 * d2;
                    if (verts[v].e8_root == verts[v + lag].e8_root) same_at_lag++;
                }
                cov /= pairs;
                double ac = (var > 1e-10) ? cov / var : 0.0;
                fprintf(fp, "  %-5d  %+.6f  %.2f%%\n", lag, ac, 100.0 * same_at_lag / pairs);
            }
        }

        /* Dot-product autocorrelation */
        fprintf(fp, "\nDot-product autocorrelation (consecutive root pairs):\n");
        if (n_edges > 10) {
            double dp_mean = 0;
            for (int i = 0; i < n_edges; i++) dp_mean += edges[i].dot_product;
            dp_mean /= n_edges;
            double dp_var = 0;
            for (int i = 0; i < n_edges; i++) {
                double d = edges[i].dot_product - dp_mean;
                dp_var += d * d;
            }
            dp_var /= n_edges;

            fprintf(fp, "  %-5s  %-10s\n", "Lag", "Autocorr");
            for (int lag = 1; lag <= 10 && lag < n_edges; lag++) {
                double cov = 0;
                int pairs = n_edges - lag;
                for (int i = 0; i < pairs; i++) {
                    cov += (edges[i].dot_product - dp_mean) *
                           (edges[i + lag].dot_product - dp_mean);
                }
                cov /= pairs;
                double ac = (dp_var > 1e-10) ? cov / dp_var : 0.0;
                fprintf(fp, "  %-5d  %+.6f\n", lag, ac);
            }
        }

        /* ---- F. Dot-Product Spectrum ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  F. Dot-Product Spectrum Along Path\n");
        fprintf(fp, "================================================================\n");

        if (n_edges > 0) {
            /* E8 roots have dot products in {-2, -1, -0.5, 0, 0.5, 1, 2} etc. */
            /* Bin by rounded value */
            int dp_bins[9] = {0};  /* -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2 */
            double dp_sum = 0;
            for (int i = 0; i < n_edges; i++) {
                dp_sum += edges[i].dot_product;
                int bin = (int)round(edges[i].dot_product * 2.0) + 4;  /* map [-2,2] to [0,8] */
                if (bin < 0) bin = 0;
                if (bin > 8) bin = 8;
                dp_bins[bin]++;
            }
            fprintf(fp, "Inner product distribution (consecutive roots):\n");
            const char *dp_labels[] = {"-2.0", "-1.5", "-1.0", "-0.5", " 0.0", "+0.5", "+1.0", "+1.5", "+2.0"};
            for (int b = 0; b < 9; b++) {
                if (dp_bins[b] > 0)
                    fprintf(fp, "  ⟨α,β⟩ = %s : %d (%.1f%%)\n",
                            dp_labels[b], dp_bins[b], 100.0 * dp_bins[b] / n_edges);
            }
            fprintf(fp, "\nMean dot product: %.4f\n", dp_sum / n_edges);
            fprintf(fp, "  Expected (independent random roots): ~0.0\n");
        }

        /* ---- G. Jordan Trace Along Path ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  G. Jordan Trace Statistics\n");
        fprintf(fp, "================================================================\n");

        {
            double jt_sum = 0, jt_sq = 0;
            double jt_min = 1e30, jt_max = -1e30;
            for (int v = 0; v < n_verts; v++) {
                double j = verts[v].jordan_trace;
                jt_sum += j;
                jt_sq += j * j;
                if (j < jt_min) jt_min = j;
                if (j > jt_max) jt_max = j;
            }
            double jt_mean = jt_sum / n_verts;
            double jt_std = sqrt(jt_sq / n_verts - jt_mean * jt_mean);
            fprintf(fp, "Mean Jordan trace: %.4f\n", jt_mean);
            fprintf(fp, "Std:               %.4f\n", jt_std);
            fprintf(fp, "Range:             [%.2f, %.2f]\n", jt_min, jt_max);
        }

        /* ---- H. Null Model ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  H. Null Model Comparison (%d random permutations)\n", cfg.n_null_perms);
        fprintf(fp, "================================================================\n\n");

        printf("  Computing null model (%d permutations)...\n", cfg.n_null_perms);

        /* Compute true path stats */
        int *true_order = (int *)malloc(n_verts * sizeof(int));
        for (int i = 0; i < n_verts; i++) true_order[i] = i;
        PathStats true_stats = compute_path_stats(verts, n_verts, true_order, &e8, &f4);

        /* Run null permutations */
        double *null_dot     = (double *)calloc(cfg.n_null_perms, sizeof(double));
        double *null_same_e8 = (double *)calloc(cfg.n_null_perms, sizeof(double));
        double *null_same_f4 = (double *)calloc(cfg.n_null_perms, sizeof(double));
        double *null_ks      = (double *)calloc(cfg.n_null_perms, sizeof(double));
        double *null_gap     = (double *)calloc(cfg.n_null_perms, sizeof(double));

        #pragma omp parallel
        {
            int *perm = (int *)malloc(n_verts * sizeof(int));
            for (int i = 0; i < n_verts; i++) perm[i] = i;
            unsigned int seed = (unsigned int)(time(NULL) ^ omp_get_thread_num());

            #pragma omp for schedule(dynamic, 10)
            for (int p = 0; p < cfg.n_null_perms; p++) {
                shuffle(perm, n_verts, &seed);
                PathStats ns = compute_path_stats(verts, n_verts, perm, &e8, &f4);
                null_dot[p]     = ns.mean_dot;
                null_same_e8[p] = ns.same_e8_frac;
                null_same_f4[p] = ns.same_f4_frac;
                null_ks[p]      = ns.angle_ks;
                null_gap[p]     = ns.mean_prime_gap;
            }
            free(perm);
        }

        /* Compute z-scores */
        double metrics_true[5] = {true_stats.mean_dot, true_stats.same_e8_frac,
                                   true_stats.same_f4_frac, true_stats.angle_ks,
                                   true_stats.mean_prime_gap};
        double *metric_null[5] = {null_dot, null_same_e8, null_same_f4, null_ks, null_gap};
        const char *metric_names[] = {"Mean dot product", "Same-E8 fraction",
                                       "Same-F4 fraction", "Angle KS stat",
                                       "Mean prime-idx gap"};

        fprintf(fp, "%-22s  %-10s  %-10s  %-10s  %-10s\n",
                "Metric", "True", "Null mean", "Null std", "z-score");
        fprintf(fp, "%-22s  %-10s  %-10s  %-10s  %-10s\n",
                "------", "----", "---------", "--------", "-------");

        for (int m = 0; m < 5; m++) {
            double sum = 0, sq = 0;
            for (int p = 0; p < cfg.n_null_perms; p++) {
                sum += metric_null[m][p];
                sq += metric_null[m][p] * metric_null[m][p];
            }
            double nm = sum / cfg.n_null_perms;
            double ns = sqrt(sq / cfg.n_null_perms - nm * nm);
            double z = (ns > 1e-15) ? (metrics_true[m] - nm) / ns : 0.0;

            fprintf(fp, "%-22s  %+.6f  %+.6f  %.6f  %+.2f\n",
                    metric_names[m], metrics_true[m], nm, ns, z);
        }

        fprintf(fp, "\nInterpretation:\n");
        fprintf(fp, "  |z| > 2.0 : statistically significant at 95%% level\n");
        fprintf(fp, "  |z| > 3.0 : highly significant at 99.7%% level\n");
        fprintf(fp, "  Positive z for 'same-root' metrics: path has MORE root repetition than random\n");
        fprintf(fp, "  Positive z for KS stat: path angles MORE non-uniform than random\n");

        free(null_dot);
        free(null_same_e8);
        free(null_same_f4);
        free(null_ks);
        free(null_gap);
        free(true_order);

        /* ---- I. Run-Length Analysis ---- */
        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  I. Run-Length Analysis (Phase-Length Modulation)\n");
        fprintf(fp, "================================================================\n\n");

        if (n_edges > 0) {
            /* Extract runs of consecutive same-root edges */
            typedef struct {
                int root;           /* E8 root index */
                int length;         /* number of consecutive edges */
                int start_edge;     /* first edge index in the run */
                double exit_dot;    /* dot product to next run's root */
            } Run;

            int run_cap = n_edges + 1;
            Run *runs = (Run *)malloc(run_cap * sizeof(Run));
            int n_runs = 0;

            int cur_root = edges[0].e8_from;
            int cur_len = 0;
            int cur_start = 0;

            for (int i = 0; i < n_edges; i++) {
                if (edges[i].e8_from == cur_root) {
                    cur_len++;
                    if (edges[i].e8_to != cur_root) {
                        /* Run ends here */
                        runs[n_runs].root = cur_root;
                        runs[n_runs].length = cur_len;
                        runs[n_runs].start_edge = cur_start;
                        runs[n_runs].exit_dot = edges[i].dot_product;
                        n_runs++;
                        cur_root = edges[i].e8_to;
                        cur_start = i + 1;
                        cur_len = 0;
                    }
                } else {
                    /* Root changed unexpectedly — close current run */
                    if (cur_len > 0) {
                        runs[n_runs].root = cur_root;
                        runs[n_runs].length = cur_len;
                        runs[n_runs].start_edge = cur_start;
                        runs[n_runs].exit_dot = (i > 0) ? edges[i-1].dot_product : 0;
                        n_runs++;
                    }
                    /* Singleton for this edge's from */
                    if (edges[i].e8_to == edges[i].e8_from) {
                        cur_root = edges[i].e8_from;
                        cur_start = i;
                        cur_len = 1;
                    } else {
                        runs[n_runs].root = edges[i].e8_from;
                        runs[n_runs].length = 1;
                        runs[n_runs].start_edge = i;
                        runs[n_runs].exit_dot = edges[i].dot_product;
                        n_runs++;
                        cur_root = edges[i].e8_to;
                        cur_start = i + 1;
                        cur_len = 0;
                    }
                }
            }
            /* Final run */
            if (cur_len > 0 || cur_start <= n_edges) {
                runs[n_runs].root = cur_root;
                runs[n_runs].length = (cur_len > 0) ? cur_len : 1;
                runs[n_runs].start_edge = cur_start;
                runs[n_runs].exit_dot = -999;
                n_runs++;
            }

            fprintf(fp, "Compression: %d edges -> %d runs (%.1f%%)\n",
                    n_edges, n_runs, 100.0 * n_runs / n_edges);

            /* Run length distribution */
            int max_run_len = 0;
            double mean_run_len = 0;
            for (int r = 0; r < n_runs; r++) {
                mean_run_len += runs[r].length;
                if (runs[r].length > max_run_len) max_run_len = runs[r].length;
            }
            mean_run_len /= n_runs;

            fprintf(fp, "Mean run length: %.2f\n", mean_run_len);
            fprintf(fp, "Max run length:  %d\n\n", max_run_len);

            /* Run length histogram */
            fprintf(fp, "Run length distribution:\n");
            int *len_hist = (int *)calloc(max_run_len + 1, sizeof(int));
            for (int r = 0; r < n_runs; r++) len_hist[runs[r].length]++;
            for (int l = 1; l <= max_run_len; l++) {
                if (len_hist[l] > 0)
                    fprintf(fp, "  length %2d: %3d runs (%.1f%%)\n",
                            l, len_hist[l], 100.0 * len_hist[l] / n_runs);
            }
            free(len_hist);

            /* Top roots by total edge count */
            fprintf(fp, "\nRoot distribution (by total edges in runs):\n");
            fprintf(fp, "  %-6s  %-6s  %-7s  %-5s  %-10s\n",
                    "Root", "Runs", "Edges", "Type", "MeanLen");
            typedef struct { int root, n_runs, total_edges; } RootRunStat;
            RootRunStat *rstats = (RootRunStat *)calloc(E8_NUM_ROOTS, sizeof(RootRunStat));
            for (int r = 0; r < n_runs; r++) {
                rstats[runs[r].root].root = runs[r].root;
                rstats[runs[r].root].n_runs++;
                rstats[runs[r].root].total_edges += runs[r].length;
            }
            /* Sort by total_edges descending */
            int rstat_order[E8_NUM_ROOTS];
            for (int i = 0; i < E8_NUM_ROOTS; i++) rstat_order[i] = i;
            for (int i = 0; i < E8_NUM_ROOTS - 1; i++)
                for (int j = i + 1; j < E8_NUM_ROOTS; j++)
                    if (rstats[rstat_order[j]].total_edges > rstats[rstat_order[i]].total_edges) {
                        int t = rstat_order[i]; rstat_order[i] = rstat_order[j]; rstat_order[j] = t;
                    }
            for (int i = 0; i < 20; i++) {
                int ri = rstat_order[i];
                if (rstats[ri].total_edges == 0) break;
                fprintf(fp, "  %-6d  %-6d  %-7d  %-5s  %.2f\n",
                        ri, rstats[ri].n_runs, rstats[ri].total_edges,
                        e8_root_type(&e8, ri) == 0 ? "I" : "II",
                        (double)rstats[ri].total_edges / rstats[ri].n_runs);
            }
            free(rstats);

            /* Exit operator distribution */
            fprintf(fp, "\nExit operator distribution (dot product at run boundaries):\n");
            int exit_dp_hist[9] = {0};
            int n_exits = 0;
            for (int r = 0; r < n_runs; r++) {
                if (runs[r].exit_dot > -900) {
                    int bin = (int)round(runs[r].exit_dot * 2.0) + 4;
                    if (bin < 0) bin = 0;
                    if (bin > 8) bin = 8;
                    exit_dp_hist[bin]++;
                    n_exits++;
                }
            }
            {
                const char *dp_labels[] = {"-2.0","-1.5","-1.0","-0.5"," 0.0","+0.5","+1.0","+1.5","+2.0"};
                for (int b = 0; b < 9; b++) {
                    if (exit_dp_hist[b] > 0)
                        fprintf(fp, "  dp=%s: %3d (%.1f%%)\n",
                                dp_labels[b], exit_dp_hist[b], 100.0 * exit_dp_hist[b] / n_exits);
                }
            }

            /* Top run-to-run adjacency patterns */
            fprintf(fp, "\nTop 15 run-to-run transitions (root_i -> root_{i+1}):\n");
            fprintf(fp, "  %-6s  %-6s  %-6s  %-8s\n", "From", "To", "Count", "DotProd");
            typedef struct { int from, to, count; } RunTrans;
            /* Collect transitions */
            int n_run_trans = n_runs - 1;
            if (n_run_trans > 0) {
                /* Use flat array for 240x240 — same as edge transition matrix */
                int *run_trans_flat = (int *)calloc(240 * 240, sizeof(int));
                for (int r = 0; r < n_run_trans; r++)
                    run_trans_flat[runs[r].root * 240 + runs[r+1].root]++;

                RunTrans rt_top[15];
                memset(rt_top, 0, sizeof(rt_top));
                for (int fr = 0; fr < 240; fr++) {
                    for (int to = 0; to < 240; to++) {
                        int c = run_trans_flat[fr * 240 + to];
                        if (c == 0) continue;
                        for (int t = 0; t < 15; t++) {
                            if (c > rt_top[t].count) {
                                for (int s = 14; s > t; s--) rt_top[s] = rt_top[s-1];
                                rt_top[t] = (RunTrans){fr, to, c};
                                break;
                            }
                        }
                    }
                }
                for (int t = 0; t < 15; t++) {
                    if (rt_top[t].count == 0) break;
                    double dp = e8_dot(&e8, rt_top[t].from, rt_top[t].to);
                    fprintf(fp, "  %-6d  %-6d  %-6d  %+.1f\n",
                            rt_top[t].from, rt_top[t].to, rt_top[t].count, dp);
                }
                free(run_trans_flat);
            }

            /* Satellite oscillation detection */
            fprintf(fp, "\nSatellite oscillation (A->B->A->B... patterns):\n");
            for (int r = 0; r < n_runs - 3; r++) {
                if (runs[r].root == runs[r+2].root &&
                    runs[r+1].root == runs[r+3].root &&
                    runs[r].root != runs[r+1].root) {
                    /* Count how long the oscillation continues */
                    int osc_len = 2;
                    int root_a = runs[r].root;
                    int root_b = runs[r+1].root;
                    for (int s = r + 2; s < n_runs - 1; s += 2) {
                        if (runs[s].root == root_a && runs[s+1].root == root_b)
                            osc_len++;
                        else
                            break;
                    }
                    if (osc_len >= 3) {
                        double dp = e8_dot(&e8, root_a, root_b);
                        fprintf(fp, "  roots %d <-> %d: %d cycles (dp=%+.1f) starting at run %d\n",
                                root_a, root_b, osc_len, dp, r);
                        r += 2 * osc_len - 2;  /* skip past this oscillation */
                    }
                }
            }

            /* Null model for run statistics */
            fprintf(fp, "\nRun-length null model (%d permutations):\n", cfg.n_null_perms);
            {
                /* Collect all edge root assignments for permutation */
                int *all_roots = (int *)malloc(n_edges * sizeof(int));
                for (int i = 0; i < n_edges; i++) all_roots[i] = edges[i].e8_from;

                double null_mean_runs = 0, null_sq_runs = 0;
                double null_mean_maxlen = 0, null_sq_maxlen = 0;
                double null_mean_meanlen = 0, null_sq_meanlen = 0;

                #pragma omp parallel
                {
                    int *perm_roots = (int *)malloc(n_edges * sizeof(int));
                    unsigned int seed = (unsigned int)(time(NULL) ^ omp_get_thread_num() ^ 0xCAFE);

                    #pragma omp for schedule(dynamic, 10) \
                        reduction(+:null_mean_runs,null_sq_runs,null_mean_maxlen,null_sq_maxlen,null_mean_meanlen,null_sq_meanlen)
                    for (int p = 0; p < cfg.n_null_perms; p++) {
                        memcpy(perm_roots, all_roots, n_edges * sizeof(int));
                        /* Fisher-Yates on the root sequence */
                        for (int i = n_edges - 1; i > 0; i--) {
                            int j = (int)((unsigned long)rand_r(&seed) % (i + 1));
                            int tmp = perm_roots[i]; perm_roots[i] = perm_roots[j]; perm_roots[j] = tmp;
                        }
                        /* Count runs */
                        int nr = 1, ml = 1, cur = 1;
                        for (int i = 1; i < n_edges; i++) {
                            if (perm_roots[i] == perm_roots[i-1]) {
                                cur++;
                            } else {
                                if (cur > ml) ml = cur;
                                nr++;
                                cur = 1;
                            }
                        }
                        if (cur > ml) ml = cur;
                        double mean_l = (double)n_edges / nr;
                        null_mean_runs += nr;
                        null_sq_runs += (double)nr * nr;
                        null_mean_maxlen += ml;
                        null_sq_maxlen += (double)ml * ml;
                        null_mean_meanlen += mean_l;
                        null_sq_meanlen += mean_l * mean_l;
                    }
                    free(perm_roots);
                }

                double nm_runs = null_mean_runs / cfg.n_null_perms;
                double ns_runs = sqrt(null_sq_runs / cfg.n_null_perms - nm_runs * nm_runs);
                double nm_maxlen = null_mean_maxlen / cfg.n_null_perms;
                double ns_maxlen = sqrt(null_sq_maxlen / cfg.n_null_perms - nm_maxlen * nm_maxlen);
                double nm_meanlen = null_mean_meanlen / cfg.n_null_perms;
                double ns_meanlen = sqrt(null_sq_meanlen / cfg.n_null_perms - nm_meanlen * nm_meanlen);

                fprintf(fp, "  %-20s  %-10s  %-10s  %-10s  %-10s\n",
                        "Metric", "True", "Null mean", "Null std", "z-score");
                fprintf(fp, "  %-20s  %-10s  %-10s  %-10s  %-10s\n",
                        "------", "----", "---------", "--------", "-------");

                double z_nr = (ns_runs > 1e-10) ? (n_runs - nm_runs) / ns_runs : 0;
                fprintf(fp, "  %-20s  %-10d  %-10.1f  %-10.2f  %+.2f\n",
                        "Num runs", n_runs, nm_runs, ns_runs, z_nr);

                double z_ml = (ns_maxlen > 1e-10) ? (max_run_len - nm_maxlen) / ns_maxlen : 0;
                fprintf(fp, "  %-20s  %-10d  %-10.1f  %-10.2f  %+.2f\n",
                        "Max run length", max_run_len, nm_maxlen, ns_maxlen, z_ml);

                double z_mrl = (ns_meanlen > 1e-10) ? (mean_run_len - nm_meanlen) / ns_meanlen : 0;
                fprintf(fp, "  %-20s  %-10.2f  %-10.4f  %-10.4f  %+.2f\n",
                        "Mean run length", mean_run_len, nm_meanlen, ns_meanlen, z_mrl);

                fprintf(fp, "\n  Interpretation: fewer runs / longer runs = MORE clustering\n");

                free(all_roots);
            }

            free(runs);
        }

        fprintf(fp, "\n================================================================\n");
        fprintf(fp, "  End of Report\n");
        fprintf(fp, "================================================================\n");
        fclose(fp);

        printf("  Written: %s\n", path);
    }

    printf("  Output files written in %.2fs\n", toc());

    /* ---- Summary ---- */
    printf("\n================================================================\n");
    printf("  Path Decoder Complete\n");
    printf("  %d vertices, %d edges analyzed\n", n_verts, n_edges);
    printf("  Output directory: %s\n", cfg.output_dir);
    printf("================================================================\n");

    /* Cleanup */
    free(edges);
    free(verts);
    free(heap);
    free(coherence);
    free(e8_assignments);
    free(norm_gaps);
    free(primes);

    return 0;
}
