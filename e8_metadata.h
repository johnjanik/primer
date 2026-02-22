/*
 * e8_metadata.h — Shared metadata, SHA-256, base-18 mapping, vertex path decoder
 *
 * Used by: e8_viz_v3.c (embed metadata), e8_verify.c (verify metadata)
 *
 * Self-contained: no OpenSSL dependency.
 */

#ifndef E8_METADATA_H
#define E8_METADATA_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ================================================================
 * Embedded SHA-256 (FIPS 180-4)
 * ================================================================ */

typedef struct {
    uint32_t state[8];
    uint64_t bitcount;
    uint8_t  buffer[64];
} SHA256_CTX;

static const uint32_t sha256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define SHA256_ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SHA256_CH(x,y,z)  (((x)&(y))^((~(x))&(z)))
#define SHA256_MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define SHA256_EP0(x)  (SHA256_ROR(x,2)^SHA256_ROR(x,13)^SHA256_ROR(x,22))
#define SHA256_EP1(x)  (SHA256_ROR(x,6)^SHA256_ROR(x,11)^SHA256_ROR(x,25))
#define SHA256_SIG0(x) (SHA256_ROR(x,7)^SHA256_ROR(x,18)^((x)>>3))
#define SHA256_SIG1(x) (SHA256_ROR(x,17)^SHA256_ROR(x,19)^((x)>>10))

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[64])
{
    uint32_t a,b,c,d,e,f,g,h,t1,t2,w[64];
    int i;

    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)data[i*4]<<24)|((uint32_t)data[i*4+1]<<16)|
               ((uint32_t)data[i*4+2]<<8)|((uint32_t)data[i*4+3]);
    for (i = 16; i < 64; i++)
        w[i] = SHA256_SIG1(w[i-2]) + w[i-7] + SHA256_SIG0(w[i-15]) + w[i-16];

    a=ctx->state[0]; b=ctx->state[1]; c=ctx->state[2]; d=ctx->state[3];
    e=ctx->state[4]; f=ctx->state[5]; g=ctx->state[6]; h=ctx->state[7];

    for (i = 0; i < 64; i++) {
        t1 = h + SHA256_EP1(e) + SHA256_CH(e,f,g) + sha256_K[i] + w[i];
        t2 = SHA256_EP0(a) + SHA256_MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
    ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void sha256_init(SHA256_CTX *ctx)
{
    ctx->bitcount = 0;
    ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t *data, size_t len)
{
    size_t i;
    uint32_t idx = (uint32_t)(ctx->bitcount / 8) % 64;

    ctx->bitcount += (uint64_t)len * 8;

    for (i = 0; i < len; i++) {
        ctx->buffer[idx++] = data[i];
        if (idx == 64) {
            sha256_transform(ctx, ctx->buffer);
            idx = 0;
        }
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[32])
{
    uint32_t idx = (uint32_t)(ctx->bitcount / 8) % 64;
    ctx->buffer[idx++] = 0x80;

    if (idx > 56) {
        while (idx < 64) ctx->buffer[idx++] = 0;
        sha256_transform(ctx, ctx->buffer);
        idx = 0;
    }
    while (idx < 56) ctx->buffer[idx++] = 0;

    /* Append bit length (big-endian) */
    for (int i = 7; i >= 0; i--)
        ctx->buffer[56 + (7 - i)] = (uint8_t)(ctx->bitcount >> (i * 8));

    sha256_transform(ctx, ctx->buffer);

    for (int i = 0; i < 8; i++) {
        hash[i*4+0] = (uint8_t)(ctx->state[i] >> 24);
        hash[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
        hash[i*4+2] = (uint8_t)(ctx->state[i] >> 8);
        hash[i*4+3] = (uint8_t)(ctx->state[i]);
    }
}

static void sha256_hex(const uint8_t hash[32], char hex[65])
{
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[i*2]   = hx[(hash[i] >> 4) & 0xf];
        hex[i*2+1] = hx[hash[i] & 0xf];
    }
    hex[64] = '\0';
}

/* ================================================================
 * Base-18 Mapping (ported from base18_decoder.c)
 * ================================================================ */

#define ACTIVE_SYMBOLS 18
#define SILENT_SYMBOLS 8

static const char ALPHABET_18[ACTIVE_SYMBOLS] = {
    'J','O','I','R','E','H','Q','P',
    'G','U','A','Y','V','C','N','F','B','M'
};

static const int SILENT_MOD26[SILENT_SYMBOLS] = {3,10,11,18,19,22,23,25};

static int8_t g_root_to_base18[E8_NUM_ROOTS]; /* -1 = silent */

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
    }
}

/* ================================================================
 * Vertex Path Decoder (ported from vertex_path_decoder.py lines 284-339)
 *
 * For each consecutive vertex pair (V_i, V_{i+1}):
 *   dot = <r_curr, r_next>   (inner product of E8 roots)
 *   cos_theta = dot / 2.0    (E8 roots have norm^2 = 2)
 *   angle = acos(clamp(cos_theta, -1, 1))
 *   char_idx = floor(angle / pi * 18) % 18
 *   tension = ||r_next - r_curr||
 *   output '.' if tension > threshold, else ALPHABET_18[char_idx]
 * ================================================================ */

typedef struct {
    int64_t gap_idx;
    float   score;
    int32_t cx, cy;
    int16_t e8_root;   /* E8 root index [0,239] */
} MetaVertexEntry;

static void decode_vertex_path(const MetaVertexEntry *verts, int n_verts,
                                const double e8_roots[][8],
                                char *out, int *out_len)
{
    int len = 0;
    for (int i = 0; i < n_verts - 1; i++) {
        int ri = verts[i].e8_root;
        int rj = verts[i + 1].e8_root;

        if (ri < 0 || ri >= E8_NUM_ROOTS || rj < 0 || rj >= E8_NUM_ROOTS) {
            out[len++] = '.';
            continue;
        }

        /* Inner product */
        double dot = 0;
        for (int d = 0; d < 8; d++)
            dot += e8_roots[ri][d] * e8_roots[rj][d];

        /* E8 roots have norm^2 = 2, so cos(theta) = dot / 2 */
        double cos_theta = dot / 2.0;
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        double angle = acos(cos_theta);

        /* Character index */
        int char_idx = (int)(angle / M_PI * 18.0) % 18;
        if (char_idx < 0) char_idx += 18;

        /* Tension = ||r_next - r_curr|| */
        double tension_sq = 0;
        for (int d = 0; d < 8; d++) {
            double diff = e8_roots[rj][d] - e8_roots[ri][d];
            tension_sq += diff * diff;
        }
        double tension = sqrt(tension_sq);

        if (tension > 1.8) {
            out[len++] = '.';
        } else {
            out[len++] = ALPHABET_18[char_idx];
        }
    }
    out[len] = '\0';
    *out_len = len;
}

/* ================================================================
 * MetadataBundle: all metadata embedded in the PNG
 * ================================================================ */

typedef struct {
    char    params[2048];        /* e8v3:params — key=value pairs */
    char    stats[2048];         /* e8v3:stats — coherence, tiers, bounds */
    char   *vertices;            /* e8v3:vertices — malloc'd, ~15KB for 500 verts */
    char    vertex_decode[1024]; /* e8v3:vertex_decode — base-18 string */
    char    base18_hash[65];     /* e8v3:base18_hash — SHA-256 hex of full stream */
    char    integrity[65];       /* e8v3:sha256 — SHA-256 of all above */
} MetadataBundle;

static void metadata_init(MetadataBundle *meta)
{
    memset(meta, 0, sizeof(*meta));
    meta->vertices = NULL;
}

static void metadata_free(MetadataBundle *meta)
{
    if (meta->vertices) {
        free(meta->vertices);
        meta->vertices = NULL;
    }
}

/* ================================================================
 * Helper: build params string from config values
 *
 * Format: max_primes=N;lattice=NAME;mode=NAME;dpi=N;fig_inches=N;
 *         n_vertices=N;draw_path=0|1;strobe_epsilon=N;strobe_phase=N
 * ================================================================ */

static void build_params_string(MetadataBundle *meta,
                                 int64_t max_primes, const char *lattice_name,
                                 const char *mode_name, int dpi, int fig_inches,
                                 int n_vertices, int draw_path,
                                 int strobe_epsilon, int strobe_phase)
{
    snprintf(meta->params, sizeof(meta->params),
             "max_primes=%ld;lattice=%s;mode=%s;dpi=%d;fig_inches=%d;"
             "n_vertices=%d;draw_path=%d;strobe_epsilon=%d;strobe_phase=%d",
             (long)max_primes, lattice_name, mode_name, dpi, fig_inches,
             n_vertices, draw_path, strobe_epsilon, strobe_phase);
}

/* ================================================================
 * Helper: build stats string from coherence stats
 * ================================================================ */

static void build_stats_string(MetadataBundle *meta,
                                double coh_mean, double coh_max,
                                int64_t tier0, int64_t tier1,
                                int64_t tier2, int64_t tier3,
                                int64_t total_triplets,
                                int64_t plot_count,
                                int32_t min_x, int32_t max_x,
                                int32_t min_y, int32_t max_y)
{
    snprintf(meta->stats, sizeof(meta->stats),
             "coh_mean=%.6f;coh_max=%.6f;"
             "tier0=%ld;tier1=%ld;tier2=%ld;tier3=%ld;total_triplets=%ld;"
             "plot_count=%ld;bounds=%d,%d,%d,%d",
             coh_mean, coh_max,
             (long)tier0, (long)tier1, (long)tier2, (long)tier3,
             (long)total_triplets, (long)plot_count,
             min_x, max_x, min_y, max_y);
}

/* ================================================================
 * Helper: build vertices string from vertex heap
 *
 * Format: gap_idx,e8_root,score,cx,cy; ...
 * ================================================================ */

static void build_vertices_string(MetadataBundle *meta,
                                   const MetaVertexEntry *heap, int n_verts)
{
    /* Estimate size: ~40 bytes per vertex */
    size_t est = (size_t)n_verts * 48 + 64;
    meta->vertices = (char *)malloc(est);
    if (!meta->vertices) {
        fprintf(stderr, "build_vertices_string: malloc failed\n");
        return;
    }

    size_t pos = 0;
    for (int i = 0; i < n_verts; i++) {
        int written = snprintf(meta->vertices + pos, est - pos,
                                "%ld,%d,%.4f,%d,%d%s",
                                (long)heap[i].gap_idx, (int)heap[i].e8_root,
                                (double)heap[i].score,
                                (int)heap[i].cx, (int)heap[i].cy,
                                (i < n_verts - 1) ? ";" : "");
        if (written < 0 || pos + (size_t)written >= est) break;
        pos += (size_t)written;
    }
}

/* ================================================================
 * Helper: compute integrity hash
 *
 * SHA-256(params + stats + vertices + vertex_decode + base18_hash)
 * ================================================================ */

static void compute_integrity_hash(MetadataBundle *meta)
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
    sha256_hex(hash, meta->integrity);
}

/* ================================================================
 * Helper: parse params string back to individual values
 *
 * Extracts key=value pairs separated by ';'
 * ================================================================ */

static int64_t parse_params_int64(const char *params, const char *key)
{
    char needle[128];
    snprintf(needle, sizeof(needle), "%s=", key);
    const char *p = strstr(params, needle);
    if (!p) return 0;
    return atol(p + strlen(needle));
}

static int parse_params_int(const char *params, const char *key)
{
    return (int)parse_params_int64(params, key);
}

static void parse_params_str(const char *params, const char *key,
                              char *out, int out_sz)
{
    char needle[128];
    snprintf(needle, sizeof(needle), "%s=", key);
    const char *p = strstr(params, needle);
    if (!p) { out[0] = '\0'; return; }
    p += strlen(needle);
    int i = 0;
    while (*p && *p != ';' && i < out_sz - 1)
        out[i++] = *p++;
    out[i] = '\0';
}

#endif /* E8_METADATA_H */
