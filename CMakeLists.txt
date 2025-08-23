cmake_minimum_required(VERSION 3.20)
project(ffmpeg_player VERSION 1.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fix runtime library conflicts on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Use dynamic runtime library for both C++ and CUDA to avoid LIBCMT conflicts
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Ensure CUDA uses the same runtime library
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    # Explicitly set C++20 flag for MSVC C++ compiler only
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++20>)
endif()

# Enable Unicode support for Windows
if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

# Renderer selection
option(USE_OPENGL_RENDERER "Use OpenGL renderer instead of DirectX 11" OFF)
if(USE_OPENGL_RENDERER)
    message(STATUS "Using OpenGL renderer")
    add_definitions(-DUSE_OPENGL_RENDERER=1)
else()
    message(STATUS "Using DirectX 11 renderer")
    add_definitions(-DUSE_OPENGL_RENDERER=0)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# FFmpeg download configuration using FetchContent
include(FetchContent)

set(FFMPEG_VERSION "7.1.1")
set(FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-lgpl-shared-7.1.zip")

FetchContent_Declare(
    FFmpeg
    URL ${FFMPEG_URL}
    SOURCE_DIR "${CMAKE_BINARY_DIR}/ffmpeg"
)

FetchContent_MakeAvailable(FFmpeg)

# Set FFmpeg paths
set(FFMPEG_DIR "${CMAKE_BINARY_DIR}/ffmpeg")
set(FFMPEG_INCLUDE_DIR "${FFMPEG_DIR}/include")
set(FFMPEG_LIB_DIR "${FFMPEG_DIR}/lib")
set(FFMPEG_BIN_DIR "${FFMPEG_DIR}/bin")

# Find FFmpeg libraries
find_library(AVCODEC_LIB avcodec PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(AVFORMAT_LIB avformat PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(AVUTIL_LIB avutil PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(SWSCALE_LIB swscale PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)
find_library(SWRESAMPLE_LIB swresample PATHS ${FFMPEG_LIB_DIR} NO_DEFAULT_PATH)

# Verify FFmpeg libraries were found
if(NOT AVCODEC_LIB OR NOT AVFORMAT_LIB OR NOT AVUTIL_LIB OR NOT SWSCALE_LIB OR NOT SWRESAMPLE_LIB)
    message(FATAL_ERROR "Could not find required FFmpeg libraries")
endif()

# Create FFmpeg target
add_library(FFmpeg INTERFACE)
target_include_directories(FFmpeg INTERFACE ${FFMPEG_INCLUDE_DIR})
target_link_libraries(FFmpeg INTERFACE 
    ${AVCODEC_LIB}
    ${AVFORMAT_LIB} 
    ${AVUTIL_LIB}
    ${SWSCALE_LIB}
    ${SWRESAMPLE_LIB}
)

# Optional CUDA support
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    set(HAVE_CUDA TRUE)
    target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=1)
    target_link_libraries(FFmpeg INTERFACE 
        CUDA::cuda_driver
        CUDA::cudart
    )
    # Add OpenGL interop support when using OpenGL renderer
    if(USE_OPENGL_RENDERER)
        target_link_libraries(FFmpeg INTERFACE CUDA::cudart)
    endif()
else()
    message(STATUS "CUDA not found - hardware decoding will use software fallback")
    set(HAVE_CUDA FALSE)
    target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=0)
endif()

# GLAD OpenGL loader (for modern OpenGL)
if(USE_OPENGL_RENDERER)
    include(FetchContent)
    
    # Fetch GLAD from official repository
    message(STATUS "Fetching GLAD OpenGL loader...")
    FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v2.0.6
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    
    FetchContent_GetProperties(glad)
    if(NOT glad_POPULATED)
        FetchContent_MakeAvailable(glad)
        
        # Add GLAD cmake subdirectory
        add_subdirectory("${glad_SOURCE_DIR}/cmake" glad_cmake)
        
        # Generate GLAD library for OpenGL 4.6 Core Profile
        glad_add_library(glad_gl_core_46 
            REPRODUCIBLE 
            EXCLUDE_FROM_ALL 
            LOADER 
            API gl:core=4.6
        )
        
        # Alias for compatibility
        add_library(glad ALIAS glad_gl_core_46)
        
        message(STATUS "GLAD OpenGL 4.6 Core Profile loader generated successfully")
    endif()
endif()

# Windows specific libraries
if(WIN32)
    # Common Windows system libraries
    target_link_libraries(FFmpeg INTERFACE
        user32
        kernel32
        gdi32
        winspool
        comdlg32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
        odbc32
        odbccp32
    )
    
    # Renderer-specific libraries
    if(USE_OPENGL_RENDERER)
        target_link_libraries(FFmpeg INTERFACE opengl32)
    else()
        target_link_libraries(FFmpeg INTERFACE
            d3d11
            dxgi
            d3dcompiler
        )
    endif()
endif()

# Main executable - conditional renderer selection
set(RENDERER_SOURCES)
if(USE_OPENGL_RENDERER)
    set(RENDERER_SOURCES src/OpenGLRenderer.cpp)
    if(HAVE_CUDA)
        list(APPEND RENDERER_SOURCES src/CudaOpenGLInterop.cpp)
        list(APPEND RENDERER_SOURCES src/CudaYuvConversion.cu)
    endif()
else()
    set(RENDERER_SOURCES src/D3D11Renderer.cpp)
endif()

add_executable(ffmpeg_player
    src/main.cpp
    src/Window.cpp
    src/CommandLineParser.cpp
    src/VideoValidator.cpp
    src/HardwareDecoder.cpp
    src/VideoDemuxer.cpp
    src/VideoDecoder.cpp
    ${RENDERER_SOURCES}
    src/VideoManager.cpp
    src/Logger.cpp
    src/FFmpegInitializer.cpp
    src/VideoSwitchingStrategy.cpp
    src/ImmediateSwitchStrategy.cpp
    src/PredecodedSwitchStrategy.cpp
    src/KeyframeSwitchStrategy.cpp
)

target_link_libraries(ffmpeg_player PRIVATE FFmpeg)

# Link GLAD for OpenGL renderer
if(USE_OPENGL_RENDERER)
    target_link_libraries(ffmpeg_player PRIVATE glad)
    # Link CUDA libraries if CUDA is available
    if(HAVE_CUDA)
        target_link_libraries(ffmpeg_player PRIVATE CUDA::cudart)
    endif()
endif()

# Copy FFmpeg DLLs to output directory on Windows
if(WIN32)
    file(GLOB FFMPEG_DLLS "${FFMPEG_BIN_DIR}/*.dll")
    foreach(DLL ${FFMPEG_DLLS})
        add_custom_command(TARGET ffmpeg_player POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}" $<TARGET_FILE_DIR:ffmpeg_player>
        )
    endforeach()
endif()

# Set target properties
set_target_properties(ffmpeg_player PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Additional linker flags to resolve runtime library conflicts on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Exclude conflicting static runtime libraries
    target_link_options(ffmpeg_player PRIVATE 
        "$<$<CONFIG:Debug>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:Debug>:/NODEFAULTLIB:LIBCMTD>"
        "$<$<CONFIG:Release>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:RelWithDebInfo>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:MinSizeRel>:/NODEFAULTLIB:LIBCMT>"
    )
endif()