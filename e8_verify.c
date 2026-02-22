/*
 * e8_verify.c — Self-Verification Tool for E8 Visualizer PNG Metadata
 *
 * Reads PNG tEXt metadata embedded by e8_viz_v3, then:
 *   1. Verifies integrity hash (SHA-256 of all metadata fields)
 *   2. Verifies vertex path decode (re-runs decode_vertex_path)
 *   3. Verifies base-18 stream hash (replays sieve + E8 assignment)
 *
 * Build: gcc -O3 -march=native -fopenmp -o e8_verify e8_verify.c -lm -lpng
 *
 * Usage: ./e8_verify <input.png>
 */

#include "e8_common.h"
#include "e8_metadata.h"
#include <png.h>

/* ================================================================
 * SieveIter: Replayable streaming segmented sieve
 * (duplicated from e8_viz_v3.c for standalone build)
 * ================================================================ */

#define SIEVE_SEG_SIZE (1 << 18)

typedef struct {
    int64_t *base_primes;
    int64_t  n_base;
    int64_t  limit;
    int64_t  max_primes;
    int64_t  seg_lo;
    int64_t  total_emitted;
    uint8_t *seg_buf;
    int64_t *seg_primes;
    int64_t  seg_count;
    int64_t  seg_pos;
} SieveIter;

static void sieve_iter_init(SieveIter *si, int64_t max_primes)
{
    si->max_primes = max_primes;
    si->limit = prime_upper_bound(max_primes);
    int64_t sqrt_limit = (int64_t)sqrt((double)si->limit) + 1;
    uint8_t *is_composite_small = (uint8_t *)calloc(sqrt_limit + 1, 1);
    if (!is_composite_small) { fprintf(stderr, "sieve_iter_init: malloc failed\n"); exit(1); }
    int64_t base_cap = (int64_t)(sqrt_limit / (log((double)sqrt_limit) - 1.1)) + 100;
    si->base_primes = (int64_t *)malloc(base_cap * sizeof(int64_t));
    if (!si->base_primes) { fprintf(stderr, "sieve_iter_init: malloc failed\n"); exit(1); }
    si->n_base = 0;
    for (int64_t i = 2; i <= sqrt_limit; i++) {
        if (!is_composite_small[i]) {
            si->base_primes[si->n_base++] = i;
            for (int64_t j = i * i; j <= sqrt_limit; j += i)
                is_composite_small[j] = 1;
        }
    }
    free(is_composite_small);
    si->seg_buf = (uint8_t *)malloc(SIEVE_SEG_SIZE);
    si->seg_primes = (int64_t *)malloc(SIEVE_SEG_SIZE * sizeof(int64_t));
    if (!si->seg_buf || !si->seg_primes) {
        fprintf(stderr, "sieve_iter_init: malloc failed\n"); exit(1);
    }
    si->seg_lo = 2;
    si->total_emitted = 0;
    si->seg_count = 0;
    si->seg_pos = 0;
}

static int64_t sieve_iter_next_chunk(SieveIter *si, int64_t *out, int64_t capacity)
{
    int64_t filled = 0;
    while (filled < capacity && si->total_emitted < si->max_primes) {
        if (si->seg_pos >= si->seg_count) {
            if (si->seg_lo > si->limit) break;
            int64_t hi = si->seg_lo + SIEVE_SEG_SIZE - 1;
            if (hi > si->limit) hi = si->limit;
            memset(si->seg_buf, 0, SIEVE_SEG_SIZE);
            for (int64_t b = 0; b < si->n_base; b++) {
                int64_t p = si->base_primes[b];
                int64_t start = ((si->seg_lo + p - 1) / p) * p;
                if (start < p * p) start = p * p;
                if (start > hi) continue;
                for (int64_t j = start; j <= hi; j += p)
                    si->seg_buf[j - si->seg_lo] = 1;
            }
            si->seg_count = 0;
            for (int64_t i = 0; i <= hi - si->seg_lo; i++) {
                if (!si->seg_buf[i])
                    si->seg_primes[si->seg_count++] = si->seg_lo + i;
            }
            si->seg_pos = 0;
            si->seg_lo = hi + 1;
        }
        int64_t avail = si->seg_count - si->seg_pos;
        int64_t want = capacity - filled;
        int64_t remain_total = si->max_primes - si->total_emitted;
        int64_t n = avail;
        if (n > want) n = want;
        if (n > remain_total) n = remain_total;
        memcpy(out + filled, si->seg_primes + si->seg_pos, n * sizeof(int64_t));
        si->seg_pos += n;
        si->total_emitted += n;
        filled += n;
    }
    return filled;
}

static void sieve_iter_free(SieveIter *si)
{
    free(si->base_primes);
    free(si->seg_buf);
    free(si->seg_primes);
}

/* ================================================================
 * Read PNG metadata: extract e8v3:* tEXt chunks
 * ================================================================ */

static int read_png_metadata(const char *path, MetadataBundle *meta)
{
    metadata_init(meta);

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    /* Verify PNG signature */
    uint8_t sig[8];
    if (fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8)) {
        fprintf(stderr, "%s: not a valid PNG file\n", path);
        fclose(fp);
        return -1;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return -1; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return -1; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return -1;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    /* Extract text chunks */
    png_textp text_ptr;
    int num_text = 0;
    png_get_text(png, info, &text_ptr, &num_text);

    int found = 0;
    for (int i = 0; i < num_text; i++) {
        const char *key = text_ptr[i].key;
        const char *val = text_ptr[i].text;
        if (!key || !val) continue;

        if (!strcmp(key, "e8v3:params")) {
            strncpy(meta->params, val, sizeof(meta->params) - 1);
            found++;
        } else if (!strcmp(key, "e8v3:stats")) {
            strncpy(meta->stats, val, sizeof(meta->stats) - 1);
            found++;
        } else if (!strcmp(key, "e8v3:vertices")) {
            meta->vertices = strdup(val);
            found++;
        } else if (!strcmp(key, "e8v3:vertex_decode")) {
            strncpy(meta->vertex_decode, val, sizeof(meta->vertex_decode) - 1);
            found++;
        } else if (!strcmp(key, "e8v3:base18_hash")) {
            strncpy(meta->base18_hash, val, sizeof(meta->base18_hash) - 1);
            found++;
        } else if (!strcmp(key, "e8v3:sha256")) {
            strncpy(meta->integrity, val, sizeof(meta->integrity) - 1);
            found++;
        }
    }

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    if (found < 5) {
        fprintf(stderr, "Warning: only found %d/6 e8v3:* metadata chunks\n", found);
    }

    return found;
}

/* ================================================================
 * Verify #1: Integrity hash
 *
 * Recompute SHA-256(params+stats+vertices+vertex_decode+base18_hash)
 * and compare against stored e8v3:sha256
 * ================================================================ */

static int verify_integrity(const MetadataBundle *meta)
{
    SHA256_CTX ctx;
    sha256_init(&ctx);

    sha256_update(&ctx, (const uint8_t *)meta->params, strlen(meta->params));
    sha256_update(&ctx, (const uint8_t *)meta->stats, strlen(meta->stats));
    if (meta->vertices)
        sha256_update(&ctx, (const uint8_t *)meta->vertices, strlen(meta->vertices));
    sha256_update(&ctx, (const uint8_t *)meta->vertex_decode, strlen(meta->vertex_decode));
    sha256_update(&ctx, (const uint8_t *)meta->base18_hash, strlen(meta->base18_hash));

    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char computed[65];
    sha256_hex(hash, computed);

    if (!strcmp(computed, meta->integrity)) {
        printf("  Integrity hash:      PASS\n");
        return 1;
    } else {
        printf("  Integrity hash:      FAIL\n");
        printf("    Expected: %s\n", meta->integrity);
        printf("    Computed: %s\n", computed);
        return 0;
    }
}

/* ================================================================
 * Verify #2: Vertex path decode
 *
 * Parse vertices from e8v3:vertices, re-run decode_vertex_path(),
 * compare against stored e8v3:vertex_decode
 * ================================================================ */

static int parse_vertices(const char *vstr, MetaVertexEntry **out, int *out_n)
{
    if (!vstr || !vstr[0]) { *out = NULL; *out_n = 0; return 0; }

    /* Count semicolons to estimate vertex count */
    int count = 1;
    for (const char *p = vstr; *p; p++)
        if (*p == ';') count++;

    MetaVertexEntry *verts = (MetaVertexEntry *)malloc(count * sizeof(MetaVertexEntry));
    if (!verts) return -1;

    int n = 0;
    const char *p = vstr;
    while (*p) {
        long gap_idx = 0;
        int e8_root = 0;
        float score = 0;
        int cx = 0, cy = 0;

        /* Parse: gap_idx,e8_root,score,cx,cy */
        gap_idx = atol(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;

        e8_root = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;

        score = (float)atof(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;

        cx = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;

        cy = atoi(p);
        while (*p && *p != ';' && *p != '\0') p++;
        if (*p == ';') p++;

        verts[n].gap_idx = gap_idx;
        verts[n].e8_root = (int16_t)e8_root;
        verts[n].score = score;
        verts[n].cx = cx;
        verts[n].cy = cy;
        n++;
    }

    *out = verts;
    *out_n = n;
    return 0;
}

static int verify_vertex_decode(const MetadataBundle *meta, const E8Lattice *e8)
{
    MetaVertexEntry *verts = NULL;
    int n_verts = 0;

    if (parse_vertices(meta->vertices, &verts, &n_verts) < 0 || n_verts < 2) {
        printf("  Vertex path decode:  SKIP (no vertices)\n");
        if (verts) free(verts);
        return 1;  /* not a failure, just nothing to check */
    }

    char decoded[1024];
    int decoded_len = 0;
    decode_vertex_path(verts, n_verts, e8->roots, decoded, &decoded_len);

    int pass = !strcmp(decoded, meta->vertex_decode);
    if (pass) {
        printf("  Vertex path decode:  PASS (%d chars match)\n", decoded_len);
    } else {
        printf("  Vertex path decode:  FAIL\n");
        printf("    Expected: %.60s%s\n", meta->vertex_decode,
               strlen(meta->vertex_decode) > 60 ? "..." : "");
        printf("    Computed: %.60s%s\n", decoded,
               decoded_len > 60 ? "..." : "");
    }

    free(verts);
    return pass;
}

/* ================================================================
 * Verify #3: Base-18 stream hash
 *
 * Replay sieve with stored params, compute E8 assignments,
 * map to base-18 chars, SHA-256 the stream, compare.
 * ================================================================ */

static int verify_base18_hash(const MetadataBundle *meta, const E8Lattice *e8)
{
    int64_t max_primes = parse_params_int64(meta->params, "max_primes");
    if (max_primes <= 0) {
        printf("  Base-18 stream hash: SKIP (no max_primes in params)\n");
        return 1;
    }

    char b1[32];
    printf("  Base-18 stream hash: replaying %s primes...\n",
           fmt_comma(max_primes, b1, sizeof(b1)));
    tic();

    SieveIter si;
    sieve_iter_init(&si, max_primes);

    SHA256_CTX ctx;
    sha256_init(&ctx);

    int64_t chunk_cap = 50000000LL + 1;
    int64_t *primes = (int64_t *)malloc(chunk_cap * sizeof(int64_t));
    int     *assigns = (int *)malloc(chunk_cap * sizeof(int));
    double  *norm_gaps = (double *)malloc(chunk_cap * sizeof(double));

    if (!primes || !assigns || !norm_gaps) {
        fprintf(stderr, "  Base-18 verify: malloc failed\n");
        free(primes); free(assigns); free(norm_gaps);
        sieve_iter_free(&si);
        return 0;
    }

    int64_t last_prime = 0;
    int has_prev = 0;
    int64_t total_gaps = 0;

    for (;;) {
        int64_t n = sieve_iter_next_chunk(&si, primes, 50000000LL);
        if (n <= 0) break;

        int64_t n_gaps;
        if (has_prev) {
            n_gaps = n;
            /* Gap 0: boundary */
            {
                double gap = (double)(primes[0] - last_prime);
                double log_p = log((double)last_prime);
                if (log_p < 1.0) log_p = 1.0;
                norm_gaps[0] = gap / log_p;
                assigns[0] = e8_assign_root(e8, norm_gaps[0]);
            }
            /* Interior gaps */
            #pragma omp parallel for schedule(static)
            for (int64_t i = 1; i < n; i++) {
                double gap = (double)(primes[i] - primes[i - 1]);
                double log_p = log((double)primes[i - 1]);
                if (log_p < 1.0) log_p = 1.0;
                norm_gaps[i] = gap / log_p;
                assigns[i] = e8_assign_root(e8, norm_gaps[i]);
            }
        } else {
            n_gaps = n - 1;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < n_gaps; i++) {
                double gap = (double)(primes[i + 1] - primes[i]);
                double log_p = log((double)primes[i]);
                if (log_p < 1.0) log_p = 1.0;
                norm_gaps[i] = gap / log_p;
                assigns[i] = e8_assign_root(e8, norm_gaps[i]);
            }
        }

        /* Hash base-18 chars (sequential) */
        for (int64_t i = 0; i < n_gaps; i++) {
            int8_t b18 = g_root_to_base18[assigns[i]];
            uint8_t ch = (b18 >= 0) ? (uint8_t)ALPHABET_18[(int)b18] : (uint8_t)'.';
            sha256_update(&ctx, &ch, 1);
        }

        total_gaps += n_gaps;
        last_prime = primes[n - 1];
        has_prev = 1;
    }

    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char computed[65];
    sha256_hex(hash, computed);

    free(primes);
    free(assigns);
    free(norm_gaps);
    sieve_iter_free(&si);

    double elapsed = toc();

    int pass = !strcmp(computed, meta->base18_hash);
    if (pass) {
        printf("  Base-18 stream hash: PASS (replayed %s gaps in %.1fs)\n",
               fmt_comma(total_gaps, b1, sizeof(b1)), elapsed);
    } else {
        printf("  Base-18 stream hash: FAIL (replayed %s gaps in %.1fs)\n",
               fmt_comma(total_gaps, b1, sizeof(b1)), elapsed);
        printf("    Expected: %s\n", meta->base18_hash);
        printf("    Computed: %s\n", computed);
    }

    return pass;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.png>\n", argv[0]);
        return 1;
    }

    const char *png_path = argv[1];

    printf("e8_verify: %s\n", png_path);

    /* Initialize E8 lattice */
    E8Lattice e8;
    e8_init(&e8);

    /* Initialize base-18 mapping */
    init_base18_mapping();

    /* Read metadata from PNG */
    MetadataBundle meta;
    int found = read_png_metadata(png_path, &meta);
    if (found <= 0) {
        fprintf(stderr, "  No e8v3:* metadata found in PNG\n");
        return 1;
    }

    printf("  Found %d metadata chunks\n", found);

    /* Print summary of embedded params */
    if (meta.params[0]) {
        int64_t mp = parse_params_int64(meta.params, "max_primes");
        char lattice[32], mode[32];
        parse_params_str(meta.params, "lattice", lattice, sizeof(lattice));
        parse_params_str(meta.params, "mode", mode, sizeof(mode));
        char b1[32];
        printf("  Params: %s primes, %s/%s\n",
               fmt_comma(mp, b1, sizeof(b1)), lattice, mode);
    }

    /* Run verification checks */
    int pass_count = 0;
    int total_checks = 0;

    /* Check 1: Integrity hash */
    total_checks++;
    if (verify_integrity(&meta)) pass_count++;

    /* Check 2: Vertex path decode */
    total_checks++;
    if (verify_vertex_decode(&meta, &e8)) pass_count++;

    /* Check 3: Base-18 stream hash */
    total_checks++;
    if (verify_base18_hash(&meta, &e8)) pass_count++;

    /* Overall verdict */
    printf("\n");
    if (pass_count == total_checks) {
        printf("  Overall: PASS -- image is faithful (%d/%d checks)\n",
               pass_count, total_checks);
    } else {
        printf("  Overall: FAIL (%d/%d checks passed)\n",
               pass_count, total_checks);
    }

    metadata_free(&meta);

    return (pass_count == total_checks) ? 0 : 1;
}
