/*
 * gemma3_safetensors.c - SafeTensors parser with memory mapping
 *
 * Parses SafeTensors format and provides efficient access to model weights.
 * Supports split files (model-00001-of-00002.safetensors, etc.)
 */

#include "gemma3.h"
#include "gemma3_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

/* ============================================================================
 * SafeTensors Data Types
 * ========================================================================== */

typedef enum {
    ST_DTYPE_F32 = 0,
    ST_DTYPE_F16 = 1,
    ST_DTYPE_BF16 = 2,
    ST_DTYPE_I32 = 3,
    ST_DTYPE_I64 = 4,
    ST_DTYPE_UNKNOWN = -1
} st_dtype;

__attribute__((unused))
static int st_dtype_size(st_dtype dtype) {
    switch (dtype) {
        case ST_DTYPE_F32: return 4;
        case ST_DTYPE_F16: return 2;
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_I32: return 4;
        case ST_DTYPE_I64: return 8;
        default: return 0;
    }
}

static st_dtype st_parse_dtype(const char *str) {
    if (strcmp(str, "F32") == 0) return ST_DTYPE_F32;
    if (strcmp(str, "F16") == 0) return ST_DTYPE_F16;
    if (strcmp(str, "BF16") == 0) return ST_DTYPE_BF16;
    if (strcmp(str, "I32") == 0) return ST_DTYPE_I32;
    if (strcmp(str, "I64") == 0) return ST_DTYPE_I64;
    return ST_DTYPE_UNKNOWN;
}

/* ============================================================================
 * Tensor Metadata
 * ========================================================================== */

#define ST_MAX_DIMS 8
#define ST_MAX_NAME_LEN 256

typedef struct {
    char name[ST_MAX_NAME_LEN];
    st_dtype dtype;
    int ndims;
    int64_t shape[ST_MAX_DIMS];
    int64_t data_offset;  /* Offset in file (after header) */
    int64_t data_size;    /* Size in bytes */
    int file_idx;         /* Index of file containing this tensor */
} st_tensor_info;

/* ============================================================================
 * SafeTensors File
 * ========================================================================== */

#define ST_MAX_TENSORS 1024

typedef struct {
    int fd;
    size_t file_size;
    void *mmap_ptr;
    size_t header_size;
    char *data_start;
} st_file;

typedef struct {
    st_file *files;
    int num_files;
    st_tensor_info *tensors;
    int num_tensors;
    char model_dir[1024];
} st_context;

/* ============================================================================
 * Simple JSON Parser (minimal, just for SafeTensors header)
 * ========================================================================== */

typedef struct {
    const char *str;
    int pos;
    int len;
} json_parser;

static void json_skip_ws(json_parser *p) {
    while (p->pos < p->len) {
        char c = p->str[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static int json_match(json_parser *p, char c) {
    json_skip_ws(p);
    if (p->pos < p->len && p->str[p->pos] == c) {
        p->pos++;
        return 1;
    }
    return 0;
}

static int json_parse_string(json_parser *p, char *out, int max_len) {
    json_skip_ws(p);
    if (p->pos >= p->len || p->str[p->pos] != '"') return 0;
    p->pos++;

    int out_pos = 0;
    while (p->pos < p->len && p->str[p->pos] != '"') {
        if (p->str[p->pos] == '\\' && p->pos + 1 < p->len) {
            p->pos++;
            char c = p->str[p->pos];
            switch (c) {
                case 'n': if (out_pos < max_len - 1) out[out_pos++] = '\n'; break;
                case 't': if (out_pos < max_len - 1) out[out_pos++] = '\t'; break;
                case 'r': if (out_pos < max_len - 1) out[out_pos++] = '\r'; break;
                case '"': if (out_pos < max_len - 1) out[out_pos++] = '"'; break;
                case '\\': if (out_pos < max_len - 1) out[out_pos++] = '\\'; break;
                default: if (out_pos < max_len - 1) out[out_pos++] = c; break;
            }
        } else {
            if (out_pos < max_len - 1) out[out_pos++] = p->str[p->pos];
        }
        p->pos++;
    }
    out[out_pos] = '\0';

    if (p->pos < p->len && p->str[p->pos] == '"') {
        p->pos++;
        return 1;
    }
    return 0;
}

static int json_parse_int64(json_parser *p, int64_t *out) {
    json_skip_ws(p);
    int64_t val = 0;
    int sign = 1;
    int started = 0;

    if (p->pos < p->len && p->str[p->pos] == '-') {
        sign = -1;
        p->pos++;
    }

    while (p->pos < p->len) {
        char c = p->str[p->pos];
        if (c >= '0' && c <= '9') {
            val = val * 10 + (c - '0');
            p->pos++;
            started = 1;
        } else {
            break;
        }
    }

    if (started) {
        *out = val * sign;
        return 1;
    }
    return 0;
}

static int json_skip_value(json_parser *p) {
    json_skip_ws(p);
    if (p->pos >= p->len) return 0;

    char c = p->str[p->pos];

    if (c == '"') {
        // String
        char tmp[4096];
        return json_parse_string(p, tmp, sizeof(tmp));
    } else if (c == '[') {
        // Array
        p->pos++;
        json_skip_ws(p);
        if (p->pos < p->len && p->str[p->pos] == ']') {
            p->pos++;
            return 1;
        }
        while (1) {
            if (!json_skip_value(p)) return 0;
            json_skip_ws(p);
            if (json_match(p, ']')) return 1;
            if (!json_match(p, ',')) return 0;
        }
    } else if (c == '{') {
        // Object
        p->pos++;
        json_skip_ws(p);
        if (p->pos < p->len && p->str[p->pos] == '}') {
            p->pos++;
            return 1;
        }
        while (1) {
            char key[256];
            if (!json_parse_string(p, key, sizeof(key))) return 0;
            if (!json_match(p, ':')) return 0;
            if (!json_skip_value(p)) return 0;
            json_skip_ws(p);
            if (json_match(p, '}')) return 1;
            if (!json_match(p, ',')) return 0;
        }
    } else if ((c >= '0' && c <= '9') || c == '-') {
        // Number
        while (p->pos < p->len) {
            c = p->str[p->pos];
            if ((c >= '0' && c <= '9') || c == '-' || c == '.' || c == 'e' || c == 'E' || c == '+') {
                p->pos++;
            } else {
                break;
            }
        }
        return 1;
    } else if (strncmp(p->str + p->pos, "true", 4) == 0) {
        p->pos += 4;
        return 1;
    } else if (strncmp(p->str + p->pos, "false", 5) == 0) {
        p->pos += 5;
        return 1;
    } else if (strncmp(p->str + p->pos, "null", 4) == 0) {
        p->pos += 4;
        return 1;
    }

    return 0;
}

/* ============================================================================
 * SafeTensors Header Parsing
 * ========================================================================== */

static int st_parse_tensor_info(json_parser *p, st_tensor_info *info) {
    // Parse: {"dtype": "BF16", "shape": [262144, 2304], "data_offsets": [0, 1207959552]}
    if (!json_match(p, '{')) return 0;

    int64_t offset_start = 0, offset_end = 0;
    info->dtype = ST_DTYPE_UNKNOWN;
    info->ndims = 0;

    while (1) {
        char key[64];
        if (!json_parse_string(p, key, sizeof(key))) return 0;
        if (!json_match(p, ':')) return 0;

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (!json_parse_string(p, dtype_str, sizeof(dtype_str))) return 0;
            info->dtype = st_parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (!json_match(p, '[')) return 0;
            info->ndims = 0;
            while (1) {
                int64_t dim;
                if (!json_parse_int64(p, &dim)) break;
                if (info->ndims < ST_MAX_DIMS) {
                    info->shape[info->ndims++] = dim;
                }
                if (!json_match(p, ',')) break;
            }
            if (!json_match(p, ']')) return 0;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (!json_match(p, '[')) return 0;
            json_parse_int64(p, &offset_start);
            json_match(p, ',');
            json_parse_int64(p, &offset_end);
            if (!json_match(p, ']')) return 0;
        } else {
            if (!json_skip_value(p)) return 0;
        }

        json_skip_ws(p);
        if (json_match(p, '}')) break;
        if (!json_match(p, ',')) return 0;
    }

    info->data_offset = offset_start;
    info->data_size = offset_end - offset_start;
    return 1;
}

static int st_parse_header(const char *json_str, int json_len,
                           st_tensor_info *tensors, int *num_tensors,
                           int file_idx, int max_tensors) {
    json_parser p = {json_str, 0, json_len};

    if (!json_match(&p, '{')) return 0;

    *num_tensors = 0;

    while (1) {
        char name[ST_MAX_NAME_LEN];
        if (!json_parse_string(&p, name, sizeof(name))) break;
        if (!json_match(&p, ':')) return 0;

        // Skip __metadata__ entry
        if (strcmp(name, "__metadata__") == 0) {
            if (!json_skip_value(&p)) return 0;
        } else {
            // Check bounds before adding tensor
            if (*num_tensors >= max_tensors) {
                fprintf(stderr, "Warning: too many tensors, skipping %s\n", name);
                if (!json_skip_value(&p)) return 0;
            } else {
                st_tensor_info *info = &tensors[*num_tensors];
                strncpy(info->name, name, ST_MAX_NAME_LEN - 1);
                info->name[ST_MAX_NAME_LEN - 1] = '\0';
                info->file_idx = file_idx;

                if (!st_parse_tensor_info(&p, info)) return 0;
                (*num_tensors)++;
            }
        }

        json_skip_ws(&p);
        if (json_match(&p, '}')) break;
        if (!json_match(&p, ',')) return 0;
    }

    return 1;
}

/* ============================================================================
 * File Operations
 * ========================================================================== */

static int st_open_file(st_file *f, const char *path) {
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) {
        return 0;
    }

    struct stat st;
    if (fstat(f->fd, &st) < 0) {
        close(f->fd);
        return 0;
    }
    f->file_size = st.st_size;

    f->mmap_ptr = mmap(NULL, f->file_size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->mmap_ptr == MAP_FAILED) {
        close(f->fd);
        return 0;
    }

    // Parse header size (first 8 bytes as little-endian uint64)
    uint64_t header_size;
    memcpy(&header_size, f->mmap_ptr, 8);
    f->header_size = header_size;

    // Data starts after 8-byte length + header
    f->data_start = (char *)f->mmap_ptr + 8 + f->header_size;

    return 1;
}

static void st_close_file(st_file *f) {
    if (f->mmap_ptr && f->mmap_ptr != MAP_FAILED) {
        munmap(f->mmap_ptr, f->file_size);
    }
    if (f->fd >= 0) {
        close(f->fd);
    }
    f->mmap_ptr = NULL;
    f->fd = -1;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

st_context *st_load(const char *model_dir) {
    st_context *ctx = (st_context *)calloc(1, sizeof(st_context));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    // Find all safetensors files
    char file_paths[16][1024];
    int num_files = 0;

    DIR *dir = opendir(model_dir);
    if (!dir) {
        free(ctx);
        return NULL;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && num_files < 16) {
        // Check if filename ends with .safetensors (not just contains it)
        size_t name_len = strlen(entry->d_name);
        const char *suffix = ".safetensors";
        size_t suffix_len = strlen(suffix);
        if (name_len > suffix_len &&
            strcmp(entry->d_name + name_len - suffix_len, suffix) == 0) {
            snprintf(file_paths[num_files], sizeof(file_paths[num_files]),
                     "%s/%s", model_dir, entry->d_name);
            num_files++;
        }
    }
    closedir(dir);

    if (num_files == 0) {
        free(ctx);
        return NULL;
    }

    // Sort files (for consistent ordering with split files)
    for (int i = 0; i < num_files - 1; i++) {
        for (int j = i + 1; j < num_files; j++) {
            if (strcmp(file_paths[i], file_paths[j]) > 0) {
                char tmp[1024];
                strcpy(tmp, file_paths[i]);
                strcpy(file_paths[i], file_paths[j]);
                strcpy(file_paths[j], tmp);
            }
        }
    }

    // Allocate files and tensors arrays
    ctx->files = (st_file *)calloc(num_files, sizeof(st_file));
    ctx->tensors = (st_tensor_info *)calloc(ST_MAX_TENSORS, sizeof(st_tensor_info));
    if (!ctx->files || !ctx->tensors) {
        free(ctx->files);
        free(ctx->tensors);
        free(ctx);
        return NULL;
    }

    ctx->num_files = num_files;
    ctx->num_tensors = 0;

    // Open and parse each file
    for (int i = 0; i < num_files; i++) {
        if (!st_open_file(&ctx->files[i], file_paths[i])) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                st_close_file(&ctx->files[j]);
            }
            free(ctx->files);
            free(ctx->tensors);
            free(ctx);
            return NULL;
        }

        // Parse header
        const char *header_str = (const char *)ctx->files[i].mmap_ptr + 8;
        int file_tensors = 0;
        int remaining_tensors = ST_MAX_TENSORS - ctx->num_tensors;
        if (!st_parse_header(header_str, ctx->files[i].header_size,
                            ctx->tensors + ctx->num_tensors, &file_tensors, i, remaining_tensors)) {
            // Cleanup on failure
            for (int j = 0; j <= i; j++) {
                st_close_file(&ctx->files[j]);
            }
            free(ctx->files);
            free(ctx->tensors);
            free(ctx);
            return NULL;
        }
        ctx->num_tensors += file_tensors;
    }

    return ctx;
}

void st_free(st_context *ctx) {
    if (!ctx) return;

    for (int i = 0; i < ctx->num_files; i++) {
        st_close_file(&ctx->files[i]);
    }
    free(ctx->files);
    free(ctx->tensors);
    free(ctx);
}

/* Find tensor by name */
st_tensor_info *st_find_tensor(st_context *ctx, const char *name) {
    for (int i = 0; i < ctx->num_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

/* Get raw pointer to tensor data (still in file format, e.g., BF16) */
void *st_get_tensor_data(st_context *ctx, st_tensor_info *info) {
    if (!ctx || !info || info->file_idx < 0 || info->file_idx >= ctx->num_files) {
        return NULL;
    }
    return ctx->files[info->file_idx].data_start + info->data_offset;
}

/* Get tensor as F32 (converting if necessary) */
float *st_get_tensor_f32(st_context *ctx, st_tensor_info *info) {
    void *raw = st_get_tensor_data(ctx, info);
    if (!raw) return NULL;

    // Calculate number of elements
    int64_t num_elements = 1;
    for (int i = 0; i < info->ndims; i++) {
        num_elements *= info->shape[i];
    }

    // Allocate output buffer
    float *out = (float *)malloc(num_elements * sizeof(float));
    if (!out) return NULL;

    // Convert based on dtype
    switch (info->dtype) {
        case ST_DTYPE_F32:
            memcpy(out, raw, num_elements * sizeof(float));
            break;
        case ST_DTYPE_BF16:
            gemma3_bf16_to_f32(out, (const uint16_t *)raw, num_elements);
            break;
        case ST_DTYPE_F16:
            // F16 conversion (simple expansion, not full precision)
            for (int64_t i = 0; i < num_elements; i++) {
                uint16_t h = ((const uint16_t *)raw)[i];
                // Extract F16 components
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;

                uint32_t f32;
                if (exp == 0) {
                    if (mant == 0) {
                        f32 = sign << 31;  // Zero
                    } else {
                        // Denormal - normalize
                        while ((mant & 0x400) == 0) {
                            mant <<= 1;
                            exp--;
                        }
                        exp++;
                        mant &= 0x3FF;
                        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7F800000 | (mant << 13);  // Inf/NaN
                } else {
                    f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&out[i], &f32, sizeof(float));
            }
            break;
        default:
            free(out);
            return NULL;
    }

    return out;
}

/* Get tensor element count */
int64_t st_tensor_numel(st_tensor_info *info) {
    int64_t n = 1;
    for (int i = 0; i < info->ndims; i++) {
        n *= info->shape[i];
    }
    return n;
}

/* Print tensor info for debugging */
void st_print_info(st_context *ctx) {
    printf("SafeTensors: %d files, %d tensors\n", ctx->num_files, ctx->num_tensors);
    for (int i = 0; i < ctx->num_tensors; i++) {
        st_tensor_info *t = &ctx->tensors[i];
        printf("  %s: dtype=%d, shape=[", t->name, t->dtype);
        for (int j = 0; j < t->ndims; j++) {
            printf("%ld%s", (long)t->shape[j], j < t->ndims - 1 ? ", " : "");
        }
        printf("], file=%d, offset=%ld, size=%ld\n",
               t->file_idx, (long)t->data_offset, (long)t->data_size);
    }
}

/* ============================================================================
 * Weight Loading for Gemma 3
 * ========================================================================== */

/* Gemma 3 weight structure - stores BF16 pointers directly to mmap'd data */
typedef struct {
    /* Embeddings */
    const uint16_t *embed_tokens;  /* [vocab_size, hidden_size] BF16 */

    /* Per-layer weights */
    struct {
        /* Self-attention */
        const uint16_t *input_layernorm;    /* [hidden_size] BF16 */
        const uint16_t *q_proj;             /* [num_heads * head_dim, hidden_size] BF16 */
        const uint16_t *k_proj;             /* [num_kv_heads * head_dim, hidden_size] BF16 */
        const uint16_t *v_proj;             /* [num_kv_heads * head_dim, hidden_size] BF16 */
        const uint16_t *o_proj;             /* [hidden_size, num_heads * head_dim] BF16 */
        const uint16_t *q_norm;             /* [head_dim] BF16 - QK normalization */
        const uint16_t *k_norm;             /* [head_dim] BF16 - QK normalization */

        /* MLP */
        const uint16_t *post_attention_layernorm;  /* [hidden_size] BF16 */
        const uint16_t *gate_proj;          /* [intermediate_size, hidden_size] BF16 */
        const uint16_t *up_proj;            /* [intermediate_size, hidden_size] BF16 */
        const uint16_t *down_proj;          /* [hidden_size, intermediate_size] BF16 */

        /* Pre-feedforward layernorm (Gemma 3 specific) */
        const uint16_t *pre_feedforward_layernorm;  /* [hidden_size] BF16 */
        const uint16_t *post_feedforward_layernorm; /* [hidden_size] BF16 */
    } layers[GEMMA3_NUM_LAYERS];

    /* Final norm */
    const uint16_t *norm;  /* [hidden_size] BF16 */
} gemma3_weights_t;

/* Load a single weight tensor by name - returns raw BF16 pointer */
static const uint16_t *load_weight_bf16(st_context *st, const char *name) {
    st_tensor_info *info = st_find_tensor(st, name);
    if (!info) {
        fprintf(stderr, "Warning: tensor '%s' not found\n", name);
        return NULL;
    }
    if (info->dtype != ST_DTYPE_BF16) {
        fprintf(stderr, "Warning: tensor '%s' is not BF16 (dtype=%d)\n", name, info->dtype);
    }
    return (const uint16_t *)st_get_tensor_data(st, info);
}

/* Load all Gemma 3 weights from SafeTensors context - uses BF16 mmap'd data */
gemma3_weights_t *gemma3_load_weights(st_context *st) {
    gemma3_weights_t *w = (gemma3_weights_t *)calloc(1, sizeof(gemma3_weights_t));
    if (!w) return NULL;

    // Load embeddings (multimodal model uses language_model.model. prefix)
    w->embed_tokens = load_weight_bf16(st, "language_model.model.embed_tokens.weight");
    if (!w->embed_tokens) {
        free(w);
        return NULL;
    }

    // Load per-layer weights
    for (int l = 0; l < GEMMA3_NUM_LAYERS; l++) {
        char name[256];

        snprintf(name, sizeof(name), "language_model.model.layers.%d.input_layernorm.weight", l);
        w->layers[l].input_layernorm = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.q_proj.weight", l);
        w->layers[l].q_proj = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.k_proj.weight", l);
        w->layers[l].k_proj = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.v_proj.weight", l);
        w->layers[l].v_proj = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.o_proj.weight", l);
        w->layers[l].o_proj = load_weight_bf16(st, name);

        // QK normalization weights (Gemma 3 specific)
        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.q_norm.weight", l);
        w->layers[l].q_norm = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.self_attn.k_norm.weight", l);
        w->layers[l].k_norm = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.post_attention_layernorm.weight", l);
        w->layers[l].post_attention_layernorm = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.gate_proj.weight", l);
        w->layers[l].gate_proj = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.up_proj.weight", l);
        w->layers[l].up_proj = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.mlp.down_proj.weight", l);
        w->layers[l].down_proj = load_weight_bf16(st, name);

        // Gemma 3 has additional layernorms
        snprintf(name, sizeof(name), "language_model.model.layers.%d.pre_feedforward_layernorm.weight", l);
        w->layers[l].pre_feedforward_layernorm = load_weight_bf16(st, name);

        snprintf(name, sizeof(name), "language_model.model.layers.%d.post_feedforward_layernorm.weight", l);
        w->layers[l].post_feedforward_layernorm = load_weight_bf16(st, name);
    }

    // Load final norm
    w->norm = load_weight_bf16(st, "language_model.model.norm.weight");

    return w;
}

/* Free loaded weights - only frees the structure, not the mmap'd data */
void gemma3_free_weights(gemma3_weights_t *w) {
    if (!w) return;
    // Note: weight pointers point to mmap'd data, not allocated memory
    // The mmap'd data is freed when st_free() is called
    free(w);
}
