cmake_minimum_required(VERSION 3.20)
project(ffmpeg_player VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Unicode support for Windows
if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
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

# Windows specific libraries
if(WIN32)
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

# Main executable
add_executable(ffmpeg_player
    src/main.cpp
    src/Window.cpp
    src/CommandLineParser.cpp
    src/VideoValidator.cpp
    src/HardwareDecoder.cpp
    src/VideoDemuxer.cpp
    src/VideoDecoder.cpp
    src/D3D11Renderer.cpp
    src/VideoManager.cpp
    src/Logger.cpp
    src/FFmpegInitializer.cpp
)

target_link_libraries(ffmpeg_player PRIVATE FFmpeg)

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