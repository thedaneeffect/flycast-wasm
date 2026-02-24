#!/bin/bash
# Flycast WASM Upstream — PRODUCTION Build, Link, Package, Deploy
# Stripped of all JIT profiling/logging for true performance baseline.
# Uses separate build directory (build-wasm-prod) to avoid touching dev build.
#
# Run from WSL: bash "/mnt/c/DEV Projects/flycast-wasm/upstream/build-and-deploy-prod.sh"
#
# Steps:
#   1. CMake configure (if needed) with -DJIT_PROD_BUILD=1
#   2. Incremental CMake build
#   3. Copy archive + strip conflicting objects
#   4. Link with EmulatorJS RetroArch
#   5. Package as EmulatorJS .data (7z)
#   6. Deploy to demo/data/cores/
#
# Exit codes: 0 = success, 1 = build failed, 2 = link failed, 3 = package failed

set -e

PROJECT_DIR="/mnt/c/DEV Projects/flycast-wasm"
SOURCE_DIR="$PROJECT_DIR/upstream/source"
BUILD_DIR="$SOURCE_DIR/build-wasm-prod"
WSL_DIR="/home/ghost/flycast-wasm"
DEMO_CORES="$PROJECT_DIR/demo/data/cores"
PATCHES_DIR="$PROJECT_DIR/upstream/patches"

# Init emsdk
source /home/ghost/.emsdk/emsdk_env.sh 2>/dev/null

echo "========================================="
echo "  PRODUCTION BUILD (JIT_PROD_BUILD=1)"
echo "  All profiling/logging stripped"
echo "========================================="
echo ""

echo "=== STEP 0: Auto-Regenerate Patches ==="
cd "$SOURCE_DIR"
git diff -- CMakeLists.txt core/ shell/ > "$PATCHES_DIR/wasm-jit-phase1-modified.patch" 2>/dev/null || true
cp core/rec-wasm/rec_wasm.cpp "$PATCHES_DIR/rec_wasm.cpp"
[ -f core/rec-wasm/wasm_module_builder.h ] && cp core/rec-wasm/wasm_module_builder.h "$PATCHES_DIR/"
[ -f core/rec-wasm/wasm_emit.h ] && cp core/rec-wasm/wasm_emit.h "$PATCHES_DIR/"
echo "Patches synced from source"

echo ""
echo "=== STEP 1: CMake Configure (if needed) ==="
if [ ! -f "$BUILD_DIR/Makefile" ]; then
    echo "First-time setup: configuring CMake for prod build..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    emcmake cmake "$SOURCE_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBRETRO=ON \
        -DCMAKE_CXX_FLAGS="-DJIT_PROD_BUILD=1" \
        -DCMAKE_C_FLAGS="-DJIT_PROD_BUILD=1" \
        2>&1 | tail -10
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FAILED: CMake configure"
        exit 1
    fi
    echo "CMake configured for prod build"
else
    echo "Build directory exists, skipping configure"
fi

echo ""
echo "=== STEP 2: Incremental CMake Build ==="
cd "$BUILD_DIR"
emmake make -j$(nproc) 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: CMake build"
    exit 1
fi

echo ""
echo "=== STEP 3: Strip Conflicting Objects ==="
cp libflycast_libretro_emscripten.a libflycast_libretro_emscripten_stripped.a
emar d libflycast_libretro_emscripten_stripped.a \
    CMakeFiles/flycast_libretro.dir/core/deps/libretro-common/file/file_path.c.o 2>/dev/null || true
echo "Archive objects: $(emar t libflycast_libretro_emscripten_stripped.a | wc -l)"

echo ""
echo "=== STEP 4: Link ==="
cd "$WSL_DIR"

# Same link flags as dev build, but using prod archives
UPSTREAM_ARCHIVE="$BUILD_DIR/libflycast_libretro_emscripten_stripped.a"
RESOURCES_ARCHIVE="$BUILD_DIR/libflycast-resources.a"

if [ ! -f "$UPSTREAM_ARCHIVE" ]; then
    echo "ERROR: Prod archive not found: $UPSTREAM_ARCHIVE"
    exit 2
fi
if [ ! -f "$RESOURCES_ARCHIVE" ]; then
    echo "ERROR: Prod resources archive not found: $RESOURCES_ARCHIVE"
    exit 2
fi

echo "Core archive: $(ls -lh "$UPSTREAM_ARCHIVE" | awk '{print $5}')"
echo "Resources archive: $(ls -lh "$RESOURCES_ARCHIVE" | awk '{print $5}')"

RA_OBJS=$(find EJS-RetroArch/obj-emscripten -name "*.o" -type f | grep -vE "libchdr_chd|libchdr_cdrom|libchdr_lzma|libchdr_bitstream|libchdr_huffman|libchdr_zlib|libchdr_flac|chd_stream|LzmaEnc|LzmaDec|Lzma2Dec|Lzma86Dec|flycast_stubs|glsym_es3" | sort)

echo "RetroArch objects: $(echo "$RA_OBJS" | wc -l)"
echo ""
echo "Linking..."

emcc -O3 -flto \
  -s WASM=1 \
  -s WASM_BIGINT \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=EJS_Runtime \
  -s EXPORTED_FUNCTIONS='["_main","_malloc","_free","_system_restart","_save_state_info","_load_state","_cmd_take_screenshot","_simulate_input","_toggleMainLoop","_get_core_options","_ejs_set_variable","_set_cheat","_reset_cheat","_shader_enable","_get_disk_count","_get_current_disk","_set_current_disk","_save_file_path","_cmd_savefiles","_supports_states","_refresh_save_files","_toggle_fastforward","_set_ff_ratio","_toggle_rewind","_set_rewind_granularity","_toggle_slow_motion","_set_sm_ratio","_get_current_frame_count","_set_vsync","_set_video_rotation","_get_video_dimensions","_ejs_set_keyboard_enabled","_wasm_mem_read8","_wasm_mem_read16","_wasm_mem_read32","_wasm_mem_write8","_wasm_mem_write16","_wasm_mem_write32","_wasm_exec_ifb","_wasm_exec_shil_fb"]' \
  -s EXPORTED_RUNTIME_METHODS='["callMain","ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","setValue","getValue","writeArrayToMemory","addRunDependency","removeRunDependency","FS","abort","AL","wasmExports"]' \
  -s INITIAL_MEMORY=268435456 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ALLOW_TABLE_GROWTH \
  -s ASYNCIFY=1 \
  -s ASYNCIFY_STACK_SIZE=65536 \
  -s 'ASYNCIFY_REMOVE=["Sh4Interpreter::*","i0*","i1*","addrspace::*","mmu_*","aica::*","Pvr*","pvr*","*ReadMem*","*WriteMem*","sh4_sched_tick*","*TA_*Param*"]' \
  -s EXIT_RUNTIME=0 \
  -s FORCE_FILESYSTEM=1 \
  -s WARN_ON_UNDEFINED_SYMBOLS=0 \
  -s ASSERTIONS=0 \
  -s DISABLE_EXCEPTION_CATCHING=0 \
  -fexceptions \
  -Wl,--wrap=glGetString -Wl,--allow-undefined \
  -s FULL_ES3=1 \
  -s MIN_WEBGL_VERSION=2 \
  -s MAX_WEBGL_VERSION=2 \
  -lopenal \
  -lidbfs.js \
  --js-library EJS-RetroArch/emscripten/library_platform_emscripten.js \
  --js-library EJS-RetroArch/emscripten/library_rwebaudio.js \
  --js-library EJS-RetroArch/emscripten/library_rwebcam.js \
  flycast_stubs.o \
  $RA_OBJS \
  "$UPSTREAM_ARCHIVE" \
  "$RESOURCES_ARCHIVE" \
  "$BUILD_DIR/core/deps/tinygettext/libtinygettext.a" \
  "$BUILD_DIR/core/deps/nowide/libnowide.a" \
  "$BUILD_DIR/core/deps/libchdr/libchdr-static.a" \
  "$BUILD_DIR/core/deps/libchdr/deps/lzma-24.05/liblzma.a" \
  "$BUILD_DIR/core/deps/libchdr/deps/zstd-1.5.6/build/cmake/lib/libzstd.a" \
  "$BUILD_DIR/core/deps/libchdr/deps/zlib-1.3.1/libz.a" \
  "$BUILD_DIR/core/deps/xxHash/cmake_unofficial/libxxhash.a" \
  -o flycast_libretro_upstream.js \
  --js-library /home/ghost/flycast-wasm/gl_override.js \
  --pre-js EJS-RetroArch/emscripten/pre.js

echo ""
echo "=== STEP 5: Package ==="
mkdir -p pkg_upstream
cp flycast_libretro_upstream.js pkg_upstream/flycast_libretro.js
cp flycast_libretro_upstream.wasm pkg_upstream/flycast_libretro.wasm
echo '{"name":"flycast","extensions":["cdi","gdi","chd","cue","iso","elf","bin","lst","zip","7z","dat"],"options":{}}' > pkg_upstream/core.json
echo '{"version":"1.0.0","minimumEJSVersion":"4.0"}' > pkg_upstream/build.json
cd pkg_upstream
7z a -t7z -mx=9 flycast-wasm.data flycast_libretro.js flycast_libretro.wasm core.json build.json 2>&1 | tail -3
if [ $? -ne 0 ]; then
    echo "FAILED: Package"
    exit 3
fi

echo ""
echo "=== STEP 6: Deploy ==="
cp flycast-wasm.data "$DEMO_CORES/flycast-wasm.data"
echo "Deployed: $(ls -lh "$DEMO_CORES/flycast-wasm.data" | awk '{print $5}')"

echo ""
echo "========================================="
echo "  PRODUCTION BUILD + DEPLOY COMPLETE"
echo "  JIT_PROD_BUILD=1 — zero logging"
echo "========================================="
echo "Run test: node upstream/flycast-wasm-test.js"
