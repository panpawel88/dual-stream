cmake_minimum_required(VERSION 3.24)
project(dual_stream VERSION 2.0.0 LANGUAGES C CXX CUDA)

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
    
    # Build performance optimizations for MSVC
    add_compile_options(
        /MP           # Enable parallel compilation
        /bigobj       # Increase object file capacity for template-heavy code
        $<$<CONFIG:Debug>:/Od>      # No optimization in Debug
        $<$<CONFIG:Debug>:/Zi>      # Debug info
        $<$<CONFIG:Release>:/O2>    # Maximum speed optimization
        $<$<CONFIG:Release>:/GL>    # Whole program optimization
    )
    
    # Linker optimizations
    add_link_options(
        $<$<CONFIG:Release>:/LTCG>  # Link-time code generation
        $<$<CONFIG:Release>:/OPT:REF>  # Remove unreferenced code
        $<$<CONFIG:Release>:/OPT:ICF>  # Remove duplicate code
    )
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

# Camera support configuration
option(ENABLE_CAMERA_SUPPORT "Enable camera capture support" ON)
if(ENABLE_CAMERA_SUPPORT)
    message(STATUS "Camera support enabled")
    
    # OpenCV support (required for camera processing)
    find_package(OpenCV QUIET COMPONENTS core imgproc objdetect videoio highgui)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        set(HAVE_OPENCV TRUE)
        add_definitions(-DHAVE_OPENCV=1)
        
        # Check for DNN module support (required for YuNet)
        find_package(OpenCV QUIET COMPONENTS dnn)
        if(TARGET opencv_dnn OR opencv_dnn_FOUND)
            message(STATUS "OpenCV DNN module found - YuNet face detection enabled")
            set(HAVE_OPENCV_DNN TRUE)
            add_definitions(-DHAVE_OPENCV_DNN=1)
        else()
            message(STATUS "OpenCV DNN module not found - YuNet face detection disabled")
            message(STATUS "Only Haar cascade face detection will be available")
            set(HAVE_OPENCV_DNN FALSE)
            add_definitions(-DHAVE_OPENCV_DNN=0)
        endif()
        
        # Create OpenCV target for linking
        add_library(CameraOpenCV INTERFACE)
        target_link_libraries(CameraOpenCV INTERFACE ${OpenCV_LIBS})
        target_include_directories(CameraOpenCV INTERFACE ${OpenCV_INCLUDE_DIRS})
        
    else()
        message(WARNING "OpenCV not found - camera support will be limited")
        set(HAVE_OPENCV FALSE)
        add_definitions(-DHAVE_OPENCV=0)
    endif()
    
    # Intel RealSense support (optional - install via vcpkg)
    find_package(realsense2 QUIET)
    if(realsense2_FOUND)
        message(STATUS "Intel RealSense SDK found: ${realsense2_VERSION}")
        set(HAVE_REALSENSE TRUE)
        add_definitions(-DHAVE_REALSENSE=1)
        
        # Create RealSense target for linking
        add_library(CameraRealSense INTERFACE)
        target_link_libraries(CameraRealSense INTERFACE ${realsense2_LIBRARY})
        target_include_directories(CameraRealSense INTERFACE ${realsense2_INCLUDE_DIR})
        
    else()
        message(STATUS "Intel RealSense SDK not found")
        message(STATUS "To enable RealSense support, install via vcpkg:")
        message(STATUS "  vcpkg install librealsense2")
        message(STATUS "  Or manually: https://github.com/IntelRealSense/librealsense")
        set(HAVE_REALSENSE FALSE)
        add_definitions(-DHAVE_REALSENSE=0)
    endif()
    
else()
    message(STATUS "Camera support disabled")
    set(HAVE_OPENCV FALSE)
    set(HAVE_REALSENSE FALSE)
    add_definitions(-DHAVE_OPENCV=0 -DHAVE_REALSENSE=0)
endif()

# Face detection model downloading (OpenCV Haar cascades)
option(DOWNLOAD_FACE_DETECTION_MODELS "Download OpenCV Haar cascade models at build time" ON)

if(ENABLE_CAMERA_SUPPORT AND HAVE_OPENCV AND DOWNLOAD_FACE_DETECTION_MODELS)
    message(STATUS "Downloading face detection models...")
    
    # Create data directory structure
    set(HAAR_MODELS_DIR "${CMAKE_BINARY_DIR}/data/haarcascades")
    set(YUNET_MODELS_DIR "${CMAKE_BINARY_DIR}/data/yunet")
    file(MAKE_DIRECTORY "${HAAR_MODELS_DIR}")
    file(MAKE_DIRECTORY "${YUNET_MODELS_DIR}")
    
    # Define Haar cascade model files to download
    set(OPENCV_MODELS_BASE_URL "https://github.com/opencv/opencv/raw/master/data/haarcascades")
    set(HAAR_CASCADE_MODELS
        "haarcascade_frontalface_alt.xml"
        "haarcascade_frontalface_default.xml"
        "haarcascade_frontalface_alt2.xml"
    )
    
    # Define YuNet model files to download
    set(YUNET_MODELS_BASE_URL "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet")
    set(YUNET_MODELS
        "face_detection_yunet_2023mar.onnx"
    )
    
    # Download Haar cascade models
    foreach(MODEL_FILE ${HAAR_CASCADE_MODELS})
        set(MODEL_URL "${OPENCV_MODELS_BASE_URL}/${MODEL_FILE}")
        set(MODEL_PATH "${HAAR_MODELS_DIR}/${MODEL_FILE}")
        
        if(NOT EXISTS "${MODEL_PATH}")
            message(STATUS "Downloading ${MODEL_FILE}...")
            file(DOWNLOAD
                "${MODEL_URL}"
                "${MODEL_PATH}"
                TIMEOUT 30
                STATUS DOWNLOAD_STATUS
                LOG DOWNLOAD_LOG
            )
            
            list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
            if(NOT DOWNLOAD_RESULT EQUAL 0)
                list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
                message(WARNING "Failed to download ${MODEL_FILE}: ${DOWNLOAD_ERROR}")
                message(STATUS "Download log: ${DOWNLOAD_LOG}")
            else()
                message(STATUS "Successfully downloaded ${MODEL_FILE}")
            endif()
        else()
            message(STATUS "${MODEL_FILE} already exists, skipping download")
        endif()
    endforeach()
    
    # Download YuNet models
    foreach(MODEL_FILE ${YUNET_MODELS})
        set(MODEL_URL "${YUNET_MODELS_BASE_URL}/${MODEL_FILE}")
        set(MODEL_PATH "${YUNET_MODELS_DIR}/${MODEL_FILE}")
        
        if(NOT EXISTS "${MODEL_PATH}")
            message(STATUS "Downloading ${MODEL_FILE}...")
            file(DOWNLOAD
                "${MODEL_URL}"
                "${MODEL_PATH}"
                TIMEOUT 60
                STATUS DOWNLOAD_STATUS
                LOG DOWNLOAD_LOG
                SHOW_PROGRESS
            )
            
            list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
            if(NOT DOWNLOAD_RESULT EQUAL 0)
                list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
                message(WARNING "Failed to download ${MODEL_FILE}: ${DOWNLOAD_ERROR}")
                message(STATUS "Download log: ${DOWNLOAD_LOG}")
                # Remove empty/failed file
                if(EXISTS "${MODEL_PATH}")
                    file(REMOVE "${MODEL_PATH}")
                endif()
            else()
                # Validate that the downloaded file is not empty
                file(SIZE "${MODEL_PATH}" MODEL_SIZE)
                if(MODEL_SIZE EQUAL 0)
                    message(WARNING "Downloaded ${MODEL_FILE} is empty (0 bytes). Removing invalid file.")
                    file(REMOVE "${MODEL_PATH}")
                else()
                    message(STATUS "Successfully downloaded ${MODEL_FILE} (${MODEL_SIZE} bytes)")
                endif()
            endif()
        else()
            # Check if existing file is empty and re-download if necessary
            file(SIZE "${MODEL_PATH}" EXISTING_SIZE)
            if(EXISTING_SIZE EQUAL 0)
                message(STATUS "Existing ${MODEL_FILE} is empty, re-downloading...")
                file(REMOVE "${MODEL_PATH}")
                file(DOWNLOAD
                    "${MODEL_URL}"
                    "${MODEL_PATH}"
                    TIMEOUT 60
                    STATUS DOWNLOAD_STATUS
                    LOG DOWNLOAD_LOG
                    SHOW_PROGRESS
                )
                
                list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
                if(NOT DOWNLOAD_RESULT EQUAL 0)
                    list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
                    message(WARNING "Failed to re-download ${MODEL_FILE}: ${DOWNLOAD_ERROR}")
                    file(REMOVE "${MODEL_PATH}")
                else()
                    file(SIZE "${MODEL_PATH}" MODEL_SIZE)
                    if(MODEL_SIZE EQUAL 0)
                        message(WARNING "Re-downloaded ${MODEL_FILE} is still empty. Removing invalid file.")
                        file(REMOVE "${MODEL_PATH}")
                    else()
                        message(STATUS "Successfully re-downloaded ${MODEL_FILE} (${MODEL_SIZE} bytes)")
                    endif()
                endif()
            else()
                message(STATUS "${MODEL_FILE} already exists (${EXISTING_SIZE} bytes), skipping download")
            endif()
        endif()
    endforeach()
    
    # Store face models directories for later use in executable post-build
    set(HAAR_MODELS_DOWNLOAD_DIR "${HAAR_MODELS_DIR}" CACHE INTERNAL "Haar cascade models directory")
    set(YUNET_MODELS_DOWNLOAD_DIR "${YUNET_MODELS_DIR}" CACHE INTERNAL "YuNet models directory")
    
else()
    if(ENABLE_CAMERA_SUPPORT AND HAVE_OPENCV AND NOT DOWNLOAD_FACE_DETECTION_MODELS)
        message(STATUS "Face detection model downloading disabled (DOWNLOAD_FACE_DETECTION_MODELS=OFF)")
    endif()
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
    set(RENDERER_SOURCES src/rendering/OpenGLRenderer.cpp)
    if(HAVE_CUDA)
        list(APPEND RENDERER_SOURCES src/rendering/CudaOpenGLInterop.cpp)
        list(APPEND RENDERER_SOURCES src/rendering/CudaYuvConversion.cu)
    endif()
else()
    set(RENDERER_SOURCES src/rendering/D3D11Renderer.cpp)
endif()

# Camera source files (conditional compilation)
set(CAMERA_SOURCES)
if(ENABLE_CAMERA_SUPPORT)
    list(APPEND CAMERA_SOURCES
        src/camera/sources/CameraSourceFactory.cpp
    )
    
    # Add OpenCV camera source if OpenCV is available
    if(HAVE_OPENCV)
        list(APPEND CAMERA_SOURCES
            src/camera/sources/OpenCVCameraSource.cpp
        )
    endif()
    
    # Add RealSense camera source if RealSense is available
    if(HAVE_REALSENSE)
        list(APPEND CAMERA_SOURCES
            src/camera/sources/RealSenseCameraSource.cpp
        )
    endif()
    
    # Add camera processing and management sources
    list(APPEND CAMERA_SOURCES
        src/camera/CameraFrame.cpp
        src/camera/processing/CameraFramePublisher.cpp
        src/camera/processing/FaceDetectionSwitchingTrigger.cpp
        src/camera/CameraManager.cpp
    )
endif()

add_executable(dual_stream
    src/pch.cpp  # Precompiled header source
    src/main.cpp
    src/ui/Window.cpp
    src/core/CommandLineParser.cpp
    src/video/VideoValidator.cpp
    src/video/decode/HardwareDecoder.cpp
    src/video/demux/VideoDemuxer.cpp
    src/video/decode/VideoDecoder.cpp
    ${RENDERER_SOURCES}
    src/rendering/RendererFactory.cpp
    src/rendering/TextureConverter.cpp
    src/video/VideoManager.cpp
    src/core/Logger.cpp
    src/core/FFmpegInitializer.cpp
    src/core/Config.cpp
    src/video/switching/VideoSwitchingStrategy.cpp
    src/video/switching/experimental/ImmediateSwitchStrategy.cpp
    src/video/switching/experimental/PredecodedSwitchStrategy.cpp
    src/video/switching/KeyframeSwitchStrategy.cpp
    src/video/triggers/KeyboardSwitchingTrigger.cpp
    src/video/triggers/SwitchingTriggerFactory.cpp
    ${CAMERA_SOURCES}
)

# Enable Unity Build for faster compilation
set_target_properties(dual_stream PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 8  # Compile 8 source files together
)

# Configure precompiled headers
target_precompile_headers(dual_stream
    PRIVATE
    src/pch.h
)

target_link_libraries(dual_stream PRIVATE FFmpeg)

# Add src directory to include path for absolute includes
target_include_directories(dual_stream PRIVATE src)

# Link GLAD for OpenGL renderer
if(USE_OPENGL_RENDERER)
    target_link_libraries(dual_stream PRIVATE glad)
    # Link CUDA libraries if CUDA is available
    if(HAVE_CUDA)
        target_link_libraries(dual_stream PRIVATE CUDA::cudart)
    endif()
endif()

# Link camera support libraries
if(ENABLE_CAMERA_SUPPORT)
    if(HAVE_OPENCV)
        target_link_libraries(dual_stream PRIVATE CameraOpenCV)
    endif()
    
    if(HAVE_REALSENSE)
        target_link_libraries(dual_stream PRIVATE CameraRealSense)
    endif()
endif()

# Copy FFmpeg DLLs to output directory on Windows
if(WIN32)
    file(GLOB FFMPEG_DLLS "${FFMPEG_BIN_DIR}/*.dll")
    foreach(DLL ${FFMPEG_DLLS})
        add_custom_command(TARGET dual_stream POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}" $<TARGET_FILE_DIR:dual_stream>
        )
    endforeach()
    
    # Copy OpenCV DLLs if OpenCV is found and camera support is enabled
    if(ENABLE_CAMERA_SUPPORT AND HAVE_OPENCV AND OpenCV_SHARED)
        get_filename_component(OpenCV_BIN_DIR "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
        if(EXISTS "${OpenCV_BIN_DIR}")
            file(GLOB OpenCV_DLLS "${OpenCV_BIN_DIR}/opencv_*${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}*.dll")
            foreach(DLL ${OpenCV_DLLS})
                if(EXISTS "${DLL}")
                    add_custom_command(TARGET dual_stream POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${DLL}" $<TARGET_FILE_DIR:dual_stream>
                    )
                endif()
            endforeach()
        endif()
    endif()
    
endif()

# Copy face detection models to runtime directory (all platforms)
if(ENABLE_CAMERA_SUPPORT AND HAVE_OPENCV AND DOWNLOAD_FACE_DETECTION_MODELS)
    # Copy Haar cascade models
    if(DEFINED HAAR_MODELS_DOWNLOAD_DIR)
        add_custom_command(TARGET dual_stream POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:dual_stream>/data/haarcascades"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${HAAR_MODELS_DOWNLOAD_DIR}" "$<TARGET_FILE_DIR:dual_stream>/data/haarcascades"
            COMMENT "Copying Haar cascade models to runtime directory"
            VERBATIM
        )
    endif()
    
    # Copy YuNet models
    if(DEFINED YUNET_MODELS_DOWNLOAD_DIR)
        add_custom_command(TARGET dual_stream POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:dual_stream>/data/yunet"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${YUNET_MODELS_DOWNLOAD_DIR}" "$<TARGET_FILE_DIR:dual_stream>/data/yunet"
            COMMENT "Copying YuNet models to runtime directory"
            VERBATIM
        )
    endif()
endif()

# Set target properties
set_target_properties(dual_stream PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    # Additional performance properties
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON  # Enable LTO for Release builds
)

# Additional linker flags to resolve runtime library conflicts on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Exclude conflicting static runtime libraries
    target_link_options(dual_stream PRIVATE 
        "$<$<CONFIG:Debug>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:Debug>:/NODEFAULTLIB:LIBCMTD>"
        "$<$<CONFIG:Release>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:RelWithDebInfo>:/NODEFAULTLIB:LIBCMT>"
        "$<$<CONFIG:MinSizeRel>:/NODEFAULTLIB:LIBCMT>"
    )
endif()