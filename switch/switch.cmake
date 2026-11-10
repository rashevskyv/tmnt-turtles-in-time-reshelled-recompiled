# Nintendo Switch build (devkitA64 + libnx + mesa-switch NVK), included from CMakeLists.txt.
# Host layer from cvhd-recomp (switch/host, switch/CVHD_COMMIT); the SDK is ReXGlue v0.10.0 + patches/rexglue-sdk
# (common + .linux) + switch/sdk-edits.py + switch/sdk-edits-tmnt.py. switch/build.sh configures and builds tmnt.nro.

# The Horizon replacements for the SDK's POSIX sources:
set(TMNT_SWITCH_HOST "${CMAKE_CURRENT_SOURCE_DIR}/switch/host")
set(REXGLUE_SWITCH_INCLUDE_DIR "${TMNT_SWITCH_HOST}/include")
set(REXGLUE_SWITCH_CORE_SOURCES
    ${TMNT_SWITCH_HOST}/memory_nx.cpp
    ${TMNT_SWITCH_HOST}/mapped_memory_nx.cpp
    ${TMNT_SWITCH_HOST}/fiber_nx.cpp
    ${TMNT_SWITCH_HOST}/exception_handler_nx.cpp
    ${TMNT_SWITCH_HOST}/seh_nx.cpp)
set(REXGLUE_SWITCH_UI_SOURCES
    ${TMNT_SWITCH_HOST}/ui/window_nx.cpp
    ${TMNT_SWITCH_HOST}/ui/windowed_app_context_nx.cpp)
set(REXGLUE_SWITCH_MAIN_SOURCE ${TMNT_SWITCH_HOST}/ui/windowed_app_main_nx.cpp)
set(REXGLUE_SWITCH_INPUT_SOURCES ${TMNT_SWITCH_HOST}/input/nx_input_driver.cpp)
# newlib hides POSIX declarations (fileno, fdopen, tm_gmtoff) under strict -std=c++23.
add_compile_definitions(_GNU_SOURCE)
# Horizon has no time zones (newlib has no tm_gmtoff): spdlog logs in UTC.
add_compile_definitions(SPDLOG_NO_TZ_OFFSET)
# vulkan.hpp has no dlopen for this platform; the driver is linked statically.
add_compile_definitions(VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=0)
# newlib/libnx gaps, in every translation unit.
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-include>"
                    "$<$<COMPILE_LANGUAGE:C,CXX>:${REXGLUE_SWITCH_INCLUDE_DIR}/switch_compat.h>")
include_directories(${REXGLUE_SWITCH_INCLUDE_DIR})
# devkitPro makes Release "-g -O2": the recompiled guest code gets -O3 like the SDK.
foreach(lang C CXX)
    string(REPLACE "-O2" "-O3" CMAKE_${lang}_FLAGS_RELEASE "${CMAKE_${lang}_FLAGS_RELEASE}")
endforeach()
# generated/rexglue.cmake runs the rexglue CLI as a build step: the cross build takes the host-built one.
set(REXGLUE_HOST_CLI "${REXSDK_DIR}/out/linux-amd64/Release/rexglue" CACHE FILEPATH "Host rexglue CLI")
add_executable(rex::rexglue IMPORTED GLOBAL)
set_target_properties(rex::rexglue PROPERTIES IMPORTED_LOCATION "${REXGLUE_HOST_CLI}")
# The executable is tmnt (tmnt.nro), not the project name the generated module targets default to.
set(REXGLUE_HOST_TARGET tmnt)
# GPU backends compiled into the SDK's GPU plugin (sdk-edits.py adds the hook), shared with cvhd-recomp:
# the trace backend (--gpu_backend=trace: every draw's state logged, shaders dumped, nothing drawn) and
# the deko3d renderer (selected by the file sdmc:/switch/tmnt/deko3d).
set(REXGPU_XENOS_EXTRA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/trace_backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/deko3d_backend.cpp")
# TMNT_r.xex (manifest [[modules]]) is linked into the executable: Horizon has no dlopen
# (switch/sdk-edits.py "Guest DLL modules"). Its base 0x87500000 is above the main image's function table:
# tools/dlc_modules.py gives it a packed table slot (generated/rex_dll_tables.h).
set(REXGLUE_STATIC_MODULES ON)
set(REXGLUE_MODULE_KEEP_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/switch/tmnt_nx_main.cpp")
execute_process(COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/dlc_modules.py" tables
    "${CMAKE_CURRENT_SOURCE_DIR}/generated" RESULT_VARIABLE _tmnt_tables_rc)
if(NOT _tmnt_tables_rc EQUAL 0)
    message(FATAL_ERROR "tools/dlc_modules.py tables failed")
endif()

include(generated/rexglue.cmake)

# The game hooks of the PC build (CMakeLists.txt), without the launcher and online play: the FIRE menu without
# Xbox Live entries, the Live stat fixes, the languages (pt-BR text and font), the saved title music level.
add_executable(tmnt switch/tmnt_nx_main.cpp
    src/fire_menu_hook.cpp src/fire_movie_patch.cpp src/live_fix.cpp src/live_stats.cpp src/local_stats.cpp
    src/localization/font_patch.cpp src/localization/game_text.cpp src/localization/game_text_hook.cpp
    src/localization/pt_br_game_text.cpp src/localization/pt_br_ui_text.cpp src/localization/ui_text.cpp
    src/title_bgm_level.cpp)
# TMNT_r.xex hooks (skip intro, title music, subtitles), as on the PC; live_hooks_tmnt_r.cpp is online play.
# In the joined module object (ld -r) these strong bodies win over the generated weak aliases.
if(TARGET tmnt_turtles_in_time_reshelled_TMNT_r)
    target_sources(tmnt_turtles_in_time_reshelled_TMNT_r PRIVATE src/title_bgm_level_tmnt_r.cpp
        src/skip_intro_tmnt_r.cpp src/localization/subtitle_default_tmnt_r.cpp)
    target_include_directories(tmnt_turtles_in_time_reshelled_TMNT_r PRIVATE src)
endif()
rexglue_setup_target(tmnt GPU_PLUGINS xenos)
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/generated/default/module_registry.cpp" APPEND PROPERTY
             COMPILE_DEFINITIONS REX_STATIC_MODULES=1)
target_sources(tmnt PRIVATE ${REXGLUE_STATIC_MODULE_OBJECTS})
# The SDK's UI headers include imgui.h (headers only: ImGui is compiled into rexruntime).
target_include_directories(tmnt PRIVATE "${REXSDK_DIR}/thirdparty/imgui" src)

set(NVK_SDK "" CACHE PATH "mesa-switch NVK install (lib/libvulkan.a)")
target_sources(tmnt PRIVATE
    ${TMNT_SWITCH_HOST}/switch_nx.cpp
    ${TMNT_SWITCH_HOST}/nx_menu.cpp
    ${TMNT_SWITCH_HOST}/nx_glyphs.cpp
    ${TMNT_SWITCH_HOST}/frame_recorder.cpp
    ${TMNT_SWITCH_HOST}/membench_nx.cpp
    ${TMNT_SWITCH_HOST}/switch_compat.c
)
set(TMNT_VERSION "0.0.unknown" CACHE STRING "NACP display version, from git (switch/build.sh)")
# tmnt_version.h is written by switch/build.sh into the build dir on every build; lz4.h (frame_recorder.cpp).
target_include_directories(tmnt PRIVATE ${TMNT_SWITCH_HOST}/include ${CMAKE_BINARY_DIR} "$ENV{DEVKITPRO}/portlibs/switch/include")
target_link_options(tmnt PRIVATE
    -Wl,--wrap=__syscall_thread_create
    -Wl,--wrap=rmutexLock
    -Wl,--wrap=rmutexTryLock
    -Wl,--wrap=rmutexUnlock
    -Wl,--wrap=nvGpuChannelKickoff
    -Wl,--wrap=nvGpuChannelAppendEntry
    -Wl,-u,vk_icdGetInstanceProcAddr
)
# zstd, expat and zlib from devkitPro portlibs (the SDK links them on Linux through the system).
# deko3dd (the debug library) reports every error through the device's debug callback before it aborts; the release
# library aborts silently (console 0.67: User Break 0x367 = DkResult_Fail with no message).
set(TMNT_DEKO3D_LIB "deko3d" CACHE STRING "deko3d library to link: deko3d (release) or deko3dd (debug, error messages)")
if(TMNT_DEKO3D_LIB STREQUAL "deko3dd")
    target_compile_definitions(tmnt PRIVATE TMNT_DEKO3D_DEBUG_LIB=1)
endif()
target_link_libraries(tmnt PRIVATE "${NVK_SDK}/lib/libvulkan.a" ${TMNT_DEKO3D_LIB} lz4 zstd expat z nx m stdc++)
# nx_create_nro ignores NAME/AUTHOR/VERSION: the NACP is generated here and handed to it.
nx_generate_nacp(tmnt.nacp NAME "TMNT Re-Shelled" AUTHOR "tmnt-recomp" VERSION "${TMNT_VERSION}")
# The hbmenu icon: the logo and title thumbnail from the player's own game files (tools/game_icon.py, switch/build.sh).
# Game art: it stays outside git and lives only inside the NRO; without it the devkitPro default icon.
set(TMNT_ICON "" CACHE FILEPATH "256x256 JPEG NRO icon (outside git)")
if(TMNT_ICON AND EXISTS "${TMNT_ICON}")
    nx_create_nro(tmnt NACP tmnt.nacp ICON "${TMNT_ICON}")
else()
    nx_create_nro(tmnt NACP tmnt.nacp)
endif()
