cmake_minimum_required(VERSION 3.20)
project(xboxrecomp VERSION 0.9.0 LANGUAGES C)

# Xbox Static Recompilation Runtime Library
# Provides link-time libraries for statically recompiled Xbox games.
# Pattern: extract emulator subsystems (xemu) into reusable static libraries.
#
# Libraries:
#   xbox_kernel    - Xbox kernel replacement layer (147 imports → Win32)
#   xbox_d3d8      - D3D8 → D3D11 graphics compatibility
#   xbox_dsound    - DirectSound → software mixer compatibility
#   xbox_input     - Xbox gamepad → XInput compatibility
#   xbox_apu       - MCPX APU audio emulation (extracted from xemu)
#   xbox_nv2a      - NV2A GPU emulation (extracted from xemu)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common definitions for all targets
add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# MinGW's headers default to an older Windows than the SDK MSVC builds with,
# and hide the Windows 10 API (SetProcessDpiAwarenessContext): same floor.
if(MINGW)
    add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

# Option to build examples

# ── Runtime Libraries ─────────────────────────────────────────

# Platform: Win32/NT type vocabulary + POSIX implementation of generic
# Win32 host primitives. Carries no Xbox semantics; provides the foundation
# the kernel HLE and the graphics/audio/input layers build on.
add_subdirectory(src/platform)

# Core: Xbox kernel replacement (memory layout, file I/O, threading, sync)
add_subdirectory(src/kernel)

# Graphics: D3D8 → D3D11 compatibility layer
add_subdirectory(src/d3d)

# L'emulation de l'APU (xemu ; son coeur DSP, src/apu/dsp, est sous GPL) et la
# couche xbox_dsound qui s'appuie sur son mixeur. Un titre servi par
# src/dsound_hle (DirectSound en haut niveau) n'en a pas besoin : il pose
# XBOXRECOMP_APU a OFF avant add_subdirectory, et rien de GPL n'est lie.
option(XBOXRECOMP_APU "Emulation de l'APU (xbox_apu, xbox_dsound)" ON)

if(XBOXRECOMP_APU)
    # Audio: DirectSound → software mixer compatibility
    add_subdirectory(src/audio)

    # Audio: MCPX APU hardware emulation (voice processor, DSP)
    add_subdirectory(src/apu)
endif()

# DirectSound de l'XDK en haut niveau, sans APU
add_subdirectory(src/dsound_hle)

# GPU: NV2A hardware emulation (register handlers, MMIO, push buffer)
add_subdirectory(src/nv2a)

# USB: MCPX OHCI host controllers. A title's XAPI is statically linked and
# drives these registers directly, so a pad only exists if they do.
add_subdirectory(src/usb)

# Input: Xbox gamepad → XInput
add_subdirectory(src/input)

# Shared FMV player (Media Foundation -> D3D8 texture)
add_subdirectory(src/video)

# ── Umbrella target ───────────────────────────────────────────

# Convenience target that links all runtime libraries
add_library(xboxrecomp INTERFACE)
target_link_libraries(xboxrecomp INTERFACE
    xbox_kernel
    xbox_d3d8
    xbox_input
    xbox_usb
    xbox_dsound_hle
    xbox_nv2a
    xbox_video
)
if(XBOXRECOMP_APU)
    target_link_libraries(xboxrecomp INTERFACE xbox_dsound xbox_apu)
else()
    target_compile_definitions(xboxrecomp INTERFACE XBOXRECOMP_SANS_APU)
endif()

