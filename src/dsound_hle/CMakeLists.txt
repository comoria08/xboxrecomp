# DirectSound de l'XDK en haut niveau (sans emulation de l'APU, voir
# XBOXRECOMP_APU dans le CMakeLists.txt racine).
# Code ecrit pour ce toolkit, a partir de mesures sur Def Jam (defjam_recomp,
# GUIDE_FR.md point 166 et suivants) : MIT comme le reste du toolkit.

add_library(xbox_dsound_hle STATIC
    dsound_hle.c
    dsound_hle_sortie.c
)

target_include_directories(xbox_dsound_hle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# platform porte le chemin src/ (kernel/xbox_memory_layout.h).
target_link_libraries(xbox_dsound_hle PRIVATE platform)

if(MSVC)
    target_compile_options(xbox_dsound_hle PRIVATE /W3)
    target_compile_definitions(xbox_dsound_hle PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(xbox_dsound_hle PRIVATE -Wall -O2)
endif()

# MinGW n'a pas xaudio2.lib, seulement la bibliotheque d'import versionnee.
if(MINGW)
    target_link_libraries(xbox_dsound_hle PRIVATE xaudio2_9 ole32)
endif()
