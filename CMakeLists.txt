cmake_minimum_required(VERSION 3.20)
project(ffmpeg_player VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

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

# FFmpeg download configuration
set(FFMPEG_VERSION "7.1.1")
set(FFMPEG_URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-lgpl-shared-7.1.zip")
set(FFMPEG_DIR "${CMAKE_BINARY_DIR}/ffmpeg")

# Download and extract FFmpeg
if(NOT EXISTS "${FFMPEG_DIR}")
    message(STATUS "Downloading FFmpeg ${FFMPEG_VERSION}...")
    
    # Download FFmpeg archive
    file(DOWNLOAD 
        ${FFMPEG_URL}
        "${CMAKE_BINARY_DIR}/ffmpeg-dev.zip"
        SHOW_PROGRESS
        TIMEOUT 300
    )
    
    # Create temp directory and extract archive using CMake's tar command (works with zip files)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ffmpeg_temp")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/ffmpeg-dev.zip"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/ffmpeg_temp"
        RESULT_VARIABLE EXTRACT_RESULT
    )
    
    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract FFmpeg archive")
    endif()
    
    # Find the extracted directory and rename it
    file(GLOB FFMPEG_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/ffmpeg_temp/ffmpeg-*")
    if(FFMPEG_EXTRACTED_DIR)
        list(GET FFMPEG_EXTRACTED_DIR 0 FFMPEG_EXTRACTED_DIR)
        file(RENAME "${FFMPEG_EXTRACTED_DIR}" "${FFMPEG_DIR}")
        file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/ffmpeg_temp")
    else()
        message(FATAL_ERROR "Could not find extracted FFmpeg directory")
    endif()
    
    # Clean up archive
    file(REMOVE "${CMAKE_BINARY_DIR}/ffmpeg-dev.zip")
    
    message(STATUS "FFmpeg ${FFMPEG_VERSION} downloaded and extracted successfully")
endif()

# Set FFmpeg paths
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
    target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=1)
    target_link_libraries(FFmpeg INTERFACE CUDA::cuda_driver)
else()
    message(STATUS "CUDA not found - hardware decoding will use software fallback")
    target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=0)
endif()

# GLAD OpenGL loader (for modern OpenGL)
if(USE_OPENGL_RENDERER)
    # Use a more reliable approach - download pre-generated GLAD files
    set(GLAD_SOURCES_DIR "${CMAKE_BINARY_DIR}/glad_generated")
    if(NOT EXISTS "${GLAD_SOURCES_DIR}")
        message(STATUS "Setting up GLAD OpenGL loader...")
        file(MAKE_DIRECTORY "${GLAD_SOURCES_DIR}")
        file(MAKE_DIRECTORY "${GLAD_SOURCES_DIR}/include")
        file(MAKE_DIRECTORY "${GLAD_SOURCES_DIR}/include/glad")
        file(MAKE_DIRECTORY "${GLAD_SOURCES_DIR}/include/KHR")
        file(MAKE_DIRECTORY "${GLAD_SOURCES_DIR}/src")
        
        # Download pre-generated GLAD files for OpenGL 4.6 Core Profile
        # This is a more reliable approach than generating at build time
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/Dav1dde/glad/master/glad/files/glad_46_core.h"
            "${GLAD_SOURCES_DIR}/include/glad/glad.h"
            TIMEOUT 30
        )
        
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/Dav1dde/glad/master/glad/files/glad_46_core.c"
            "${GLAD_SOURCES_DIR}/src/glad.c" 
            TIMEOUT 30
        )
        
        file(DOWNLOAD 
            "https://raw.githubusercontent.com/Dav1dde/glad/master/glad/files/khrplatform.h"
            "${GLAD_SOURCES_DIR}/include/KHR/khrplatform.h"
            TIMEOUT 30
        )
        
        # Fallback: Create basic GLAD files if download fails
        if(NOT EXISTS "${GLAD_SOURCES_DIR}/include/glad/glad.h")
            message(STATUS "Using fallback GLAD implementation")
            # Create a basic GLAD header with essential OpenGL 4.6 definitions
            file(WRITE "${GLAD_SOURCES_DIR}/include/glad/glad.h" [=[
#pragma once
#include <windows.h>
#include <GL/gl.h>

#define GLAD_GL_VERSION_4_6 1

// OpenGL API function pointers will be loaded by gladLoadGL()
extern int gladLoadGL(void);

// Define constants for modern OpenGL that might be missing
#ifndef GL_DEBUG_OUTPUT
#define GL_DEBUG_OUTPUT 0x92E0
#define GL_DEBUG_OUTPUT_SYNCHRONOUS 0x8242
#define GL_DEBUG_SEVERITY_HIGH 0x9146
#define GL_DEBUG_SEVERITY_MEDIUM 0x9147
#define GL_DEBUG_SEVERITY_LOW 0x9148
#define GL_DEBUG_SEVERITY_NOTIFICATION 0x826B
#define GL_DEBUG_TYPE_ERROR 0x824C
#define GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR 0x824D
#define GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR 0x824E
#define GL_DEBUG_TYPE_PORTABILITY 0x824F
#define GL_DEBUG_TYPE_PERFORMANCE 0x8250
#define GL_DEBUG_TYPE_OTHER 0x8251
#endif

#ifndef GL_CONTEXT_FLAGS
#define GL_CONTEXT_FLAGS 0x821E
#define GL_CONTEXT_FLAG_DEBUG_BIT 0x00000002
#define GL_CONTEXT_PROFILE_MASK 0x9126
#define GL_CONTEXT_CORE_PROFILE_BIT 0x00000001
#endif

// Debug callback function type
typedef void (APIENTRY *GLDEBUGPROC)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam);
]=])
            
            file(WRITE "${GLAD_SOURCES_DIR}/include/KHR/khrplatform.h" "#pragma once\n")
            
            file(WRITE "${GLAD_SOURCES_DIR}/src/glad.c" [=[
#include <glad/glad.h>
#include <windows.h>

// Fallback GLAD implementation that loads core OpenGL functions
// This is a minimal implementation for when GLAD files can't be downloaded

// Function pointers for core OpenGL functions
PFNGLGETSTRINGPROC glad_glGetString = NULL;
PFNGLGETINTEGERVPROC glad_glGetIntegerv = NULL;
PFNGLVIEWPORTPROC glad_glViewport = NULL;
PFNGLCLEARPROC glad_glClear = NULL;
PFNGLCLEARCOLORPROC glad_glClearColor = NULL;

int gladLoadGL(void) {
    // Load core OpenGL 1.1 functions that should be available
    glad_glGetString = (PFNGLGETSTRINGPROC)wglGetProcAddress("glGetString");
    if (!glad_glGetString) glad_glGetString = glGetString;
    
    glad_glGetIntegerv = (PFNGLGETINTEGERVPROC)wglGetProcAddress("glGetIntegerv");  
    if (!glad_glGetIntegerv) glad_glGetIntegerv = glGetIntegerv;
    
    glad_glViewport = (PFNGLVIEWPORTPROC)wglGetProcAddress("glViewport");
    if (!glad_glViewport) glad_glViewport = glViewport;
    
    glad_glClear = (PFNGLCLEARPROC)wglGetProcAddress("glClear");
    if (!glad_glClear) glad_glClear = glClear;
    
    glad_glClearColor = (PFNGLCLEARCOLORPROC)wglGetProcAddress("glClearColor");
    if (!glad_glClearColor) glad_glClearColor = glClearColor;
    
    return 1; // Always succeed for fallback
}

// Map GLAD function names to our pointers (basic functions only)
#define glGetString glad_glGetString
#define glGetIntegerv glad_glGetIntegerv
#define glViewport glad_glViewport
#define glClear glad_glClear
#define glClearColor glad_glClearColor
]=])
        endif()
        
        message(STATUS "GLAD OpenGL loader setup complete")
    endif()
    
    # Create GLAD library target
    add_library(glad STATIC 
        "${GLAD_SOURCES_DIR}/src/glad.c"
    )
    target_include_directories(glad PUBLIC "${GLAD_SOURCES_DIR}/include")
    
    # Explicitly set the language to C
    set_target_properties(glad PROPERTIES
        LINKER_LANGUAGE C
        C_STANDARD 99
    )
    
    if(WIN32)
        target_link_libraries(glad PUBLIC opengl32)
    endif()
endif()

# Windows specific libraries
if(WIN32)
    if(USE_OPENGL_RENDERER)
        target_link_libraries(FFmpeg INTERFACE
            opengl32
            gdi32
            user32
            kernel32
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
    else()
        target_link_libraries(FFmpeg INTERFACE
            d3d11
            dxgi
            d3dcompiler
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
    endif()
endif()

# Main executable - conditional renderer selection
set(RENDERER_SOURCES)
if(USE_OPENGL_RENDERER)
    set(RENDERER_SOURCES src/OpenGLRenderer.cpp)
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