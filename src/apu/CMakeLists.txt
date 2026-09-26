# MCPX APU Emulation Library
# Extracted from xemu (https://github.com/xemu-project/xemu)
#
# Voice Processor (VP): ADPCM/PCM decode, sample mixing, envelopes, filters
# Global Processor (GP): DSP56300, effects and mixing
# Encode Processor (EP): DSP56300, final encode
#
# NOTE ON LICENSING: dsp/ is GPL-2.0-or-later (it descends from Hatari's
# DSP M56001 emulation), while the rest of this directory is LGPL-2.1-or-later
# and the toolkit around it is MIT. Linking dsp/ in makes the resulting
# binary GPL-2.0-or-later. See the NOTICE file.

add_library(xbox_apu STATIC
    apu_core.c
    apu_vp.c
    apu_dsp.c
    apu_mmio_hook.c
    apu_xaudio2.c
    dsp/dsp.c
    dsp/dsp_cpu.c
    dsp/dsp_dma.c
)

target_include_directories(xbox_apu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# qemu_shim.h lives in nv2a/ - shared QEMU type abstraction
target_include_directories(xbox_apu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nv2a
)

if(MSVC)
    target_compile_options(xbox_apu PRIVATE
        /W3
        /wd4244  # conversion warnings (float/double/int)
        /wd4146  # unary minus on unsigned
        /wd4267  # size_t to int conversion
    )
    target_compile_definitions(xbox_apu PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
    )
endif()

if(WIN32)
    target_link_libraries(xbox_apu PRIVATE platform kernel32 winmm)
else()
    # Linux: platform layer (Win32 primitives on POSIX) + SDL2 audio + libm
    find_package(SDL2 REQUIRED)
    target_link_libraries(xbox_apu PRIVATE platform SDL2::SDL2 m)
endif()

# MSVC links XAudio2 through the #pragma comment(lib) in apu_xaudio2.c. MinGW
# has no xaudio2.lib, only versioned import libraries; 2.9 is Windows 10's.
if(MINGW)
    target_link_libraries(xbox_apu PRIVATE xaudio2_9)
endif()
