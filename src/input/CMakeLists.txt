# Xbox Input → XInput Compatibility Layer

add_library(xbox_input STATIC
    xinput_device.c
)

target_include_directories(xbox_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    # MinGW's libxinput.a imports XINPUT1_3.dll (DirectX SDK), absent from a
    # bare Windows 10; MSVC's xinput.lib imports XINPUT1_4.dll, which ships.
    if(MINGW)
        target_link_libraries(xbox_input PUBLIC platform xinput1_4)
    else()
        target_link_libraries(xbox_input PUBLIC platform xinput)
    endif()
else()
    find_package(SDL2 REQUIRED)
    target_link_libraries(xbox_input PUBLIC platform SDL2::SDL2)
endif()

if(MSVC)
    target_compile_options(xbox_input PRIVATE /W4 /O2)
else()
    target_compile_options(xbox_input PRIVATE -Wall -Wextra -O2)
endif()
