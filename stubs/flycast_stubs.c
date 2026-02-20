#include <string.h>
#include <stdlib.h>

/* fill_short_pathname_representation
 * Flycast calls this, but it internally calls fill_pathname which has a
 * void vs size_t return signature mismatch between Flycast and RetroArch.
 * In WASM, signature mismatches are instant unreachable traps — not warnings.
 * This stub provides a safe implementation that avoids the problematic call chain.
 */
void fill_short_pathname_representation(char* out_rep, const char *in_path, size_t size) {
    const char *last_slash = in_path;
    const char *p;
    for (p = in_path; *p; p++) {
        if (*p == '/' || *p == '\\')
            last_slash = p + 1;
    }
    strncpy(out_rep, last_slash, size - 1);
    out_rep[size - 1] = '\0';
}

void fill_short_pathname_representation_noext(char* out_rep, const char *in_path, size_t size) {
    char *dot;
    fill_short_pathname_representation(out_rep, in_path, size);
    dot = strrchr(out_rep, '.');
    if (dot)
        *dot = '\0';
}

/* fill_pathname — must match RetroArch's signature (returns size_t, not void) */
size_t fill_pathname(char *out_path, const char *in_path, const char *replace, size_t size) {
    strncpy(out_path, in_path, size - 1);
    out_path[size - 1] = '\0';
    char *dot = strrchr(out_path, '.');
    if (dot)
        *dot = '\0';
    if (replace) {
        size_t len = strlen(out_path);
        strncpy(out_path + len, replace, size - len - 1);
        out_path[size - 1] = '\0';
    }
    return strlen(out_path);
}
