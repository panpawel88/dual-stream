cmake_minimum_required(VERSION 3.24)
project(dual_stream VERSION 2.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Fix runtime library conflicts on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Use dynamic runtime library for both C++ and CUDA to avoid LIBCMT conflicts
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Ensure CUDA uses the same runtime library
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    # Explicitly set C++20 flag for MSVC C++ compiler only
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/std:c++20>)
    
    # Build performance optimizations for MSVC - C++ only
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>         # Enable parallel compilation (C++ only)
        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>     # Increase object file capacity (C++ only)
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od>       # No optimization in Debug
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Zi>       # Debug info
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>     # Maximum speed optimization
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>     # Whole program optimization
    )
    
    # CUDA-specific compiler options (nvcc compatible)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MP>          # Parallel compilation for host code
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>      # Object file capacity for host code
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Od>    # Debug no optimization
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Zi>    # Debug info
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>                # CUDA debug info
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/O2>  # Release optimization
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

# Both renderers are now always available at runtime
message(STATUS "Both OpenGL and DirectX 11 renderers available")

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
option(ENABLE_CUDA "Enable CUDA support for OpenGL renderer acceleration" ON)

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        set(HAVE_CUDA TRUE)
        
        # Ensure CUDA uses the same runtime library as C++ on Windows
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
        endif()
        
        target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=1)
        target_link_libraries(FFmpeg INTERFACE 
            CUDA::cuda_driver
            CUDA::cudart
        )
    else()
        message(STATUS "CUDA not found - hardware decoding will use software fallback")
        set(HAVE_CUDA FALSE)
        target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=0)
    endif()
else()
    message(STATUS "CUDA support disabled by user")
    set(HAVE_CUDA FALSE)
    target_compile_definitions(FFmpeg INTERFACE HAVE_CUDA=0)
endif()

# ImGui Configuration (using git submodule)
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_dx11.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# ImGui-Notify Configuration (using git submodule) - Header-only library
set(IMGUI_NOTIFY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-notify")

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

# GLAD OpenGL loader (for modern OpenGL - always needed for runtime switching)
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
    
    # Link both renderer libraries for runtime selection
    target_link_libraries(FFmpeg INTERFACE
        # OpenGL libraries
        opengl32
        # DirectX 11 libraries  
        d3d11
        dxgi
        d3dcompiler
    )
endif()

# Main executable - compile both renderers for runtime selection
set(RENDERER_SOURCES
    src/rendering/OpenGLRenderer.cpp
    src/rendering/D3D11Renderer.cpp
    src/rendering/renderpass/RenderPassConfig.cpp
    src/rendering/renderpass/RenderPassPipeline.cpp
    src/rendering/renderpass/RenderPassConfigLoader.cpp
    src/rendering/renderpass/RenderPassConfigLoaderOpenGL.cpp
    # D3D11 render pass core files
    src/rendering/renderpass/d3d11/D3D11SimpleRenderPass.cpp
    src/rendering/renderpass/d3d11/D3D11RenderPassResources.cpp
    src/rendering/renderpass/d3d11/ShaderLibrary.cpp
    # D3D11 render pass implementations
    src/rendering/renderpass/d3d11/passes/SharpenPass.cpp
    src/rendering/renderpass/d3d11/passes/BloomPass.cpp
    src/rendering/renderpass/d3d11/passes/VignettePass.cpp
    # OpenGL render pass core files
    src/rendering/renderpass/opengl/OpenGLRenderPassPipeline.cpp
    src/rendering/renderpass/opengl/OpenGLRenderPassResources.cpp
    src/rendering/renderpass/opengl/OpenGLSimpleRenderPass.cpp
    # OpenGL render pass implementations
    src/rendering/renderpass/opengl/passes/PassthroughPass.cpp
    src/rendering/renderpass/opengl/passes/YUVToRGBRenderPass.cpp
    src/rendering/renderpass/opengl/passes/MotionBlurPass.cpp
    src/rendering/renderpass/opengl/passes/VignettePass.cpp
    src/rendering/renderpass/opengl/passes/SharpenPass.cpp
    src/rendering/renderpass/opengl/passes/BloomPass.cpp
)

# Add CUDA sources for OpenGL renderer if CUDA is available
if(HAVE_CUDA)
    list(APPEND RENDERER_SOURCES 
        src/rendering/CudaOpenGLInterop.cpp
        src/rendering/CudaYuvConversion.cu
    )
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
    src/core/PerformanceStatistics.cpp
    src/video/switching/VideoSwitchingStrategy.cpp
    src/video/switching/experimental/ImmediateSwitchStrategy.cpp
    src/video/switching/experimental/PredecodedSwitchStrategy.cpp
    src/video/switching/KeyframeSwitchStrategy.cpp
    src/video/triggers/KeyboardSwitchingTrigger.cpp
    src/video/triggers/SwitchingTriggerFactory.cpp
    ${CAMERA_SOURCES}
)

# Unity Build disabled due to OpenGL header conflicts with windows.h in pch.h
# The issue: pch.h includes windows.h which includes GL/gl.h, conflicting with GLAD
set_target_properties(dual_stream PROPERTIES
    UNITY_BUILD OFF
)

# Exclude OpenGL-related files from using precompiled headers to prevent header conflicts
set_source_files_properties(
    src/rendering/OpenGLRenderer.cpp
    src/rendering/OpenGLToastRenderer.cpp
    src/rendering/CudaOpenGLInterop.cpp
    src/rendering/RendererFactory.cpp
    src/rendering/renderpass/RenderPassConfigLoaderOpenGL.cpp
    # OpenGL render pass core files
    src/rendering/renderpass/opengl/OpenGLRenderPassPipeline.cpp
    src/rendering/renderpass/opengl/OpenGLRenderPassResources.cpp
    src/rendering/renderpass/opengl/OpenGLSimpleRenderPass.cpp
    # OpenGL render pass implementations
    src/rendering/renderpass/opengl/passes/PassthroughPass.cpp
    src/rendering/renderpass/opengl/passes/YUVToRGBRenderPass.cpp
    src/rendering/renderpass/opengl/passes/MotionBlurPass.cpp
    src/rendering/renderpass/opengl/passes/VignettePass.cpp
    src/rendering/renderpass/opengl/passes/SharpenPass.cpp
    src/rendering/renderpass/opengl/passes/BloomPass.cpp
    src/video/VideoManager.cpp
    src/video/decode/VideoDecoder.cpp
    PROPERTIES
    SKIP_PRECOMPILE_HEADERS ON
)

# Configure precompiled headers
target_precompile_headers(dual_stream
    PRIVATE
    src/pch.h
)

target_link_libraries(dual_stream PRIVATE FFmpeg)

# Add src directory to include path for absolute includes
target_include_directories(dual_stream PRIVATE src)

# Add ImGui include directories and sources
target_include_directories(dual_stream PRIVATE ${IMGUI_DIR})
target_include_directories(dual_stream PRIVATE ${IMGUI_DIR}/backends)
target_sources(dual_stream PRIVATE ${IMGUI_SOURCES})

# Add ImGui-Notify include directory (header-only library)
target_include_directories(dual_stream PRIVATE ${IMGUI_NOTIFY_DIR}/win32Example/backends)
target_include_directories(dual_stream PRIVATE ${IMGUI_NOTIFY_DIR}/win32Example/fonts)

# Configure ImGui-Notify to render outside main window
target_compile_definitions(dual_stream PRIVATE NOTIFY_RENDER_OUTSIDE_MAIN_WINDOW=false)

# Add overlay system source files
target_sources(dual_stream PRIVATE
    src/ui/ImGuiManager.cpp
    src/ui/NotificationManager.cpp
    src/ui/UIRegistry.cpp
    src/ui/GlobalInputHandler.cpp
    src/ui/OverlayManager.cpp
    src/rendering/renderpass/OverlayRenderPass.cpp
    src/rendering/renderpass/d3d11/passes/D3D11OverlayRenderPass.cpp
    src/rendering/renderpass/opengl/passes/OpenGLOverlayRenderPass.cpp
)

# Always link GLAD for OpenGL runtime support
target_link_libraries(dual_stream PRIVATE glad)

# Link CUDA libraries if CUDA is available (needed for OpenGL CUDA interop)
if(HAVE_CUDA)
    target_link_libraries(dual_stream PRIVATE CUDA::cudart)
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

# Copy configuration file to output directory
add_custom_command(TARGET dual_stream POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/config/default.ini" $<TARGET_FILE_DIR:dual_stream>/config/default.ini
    COMMENT "Copying configuration file to runtime directory"
    VERBATIM
)

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

# Test system
option(ENABLE_TESTS "Enable test system" ON)

if(ENABLE_TESTS)
    message(STATUS "Test system enabled")
    
    # Add test subdirectory
    add_subdirectory(tests)
    
    # Create convenience targets for running tests
    add_custom_target(test_videos
        COMMAND cmd //c generate_test_videos.bat
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating test videos"
    )
    
    # Make test runner depend on main executable being built  
    add_dependencies(test_runner dual_stream)
    
    message(STATUS "Test targets available:")
    message(STATUS "  make test_videos       - Generate test videos")
    message(STATUS "  make test_runner       - Build test runner executable")
    message(STATUS "  make run_basic_tests   - Run basic frame validation tests") 
    message(STATUS "  make run_all_tests     - Run all test suites")
    message(STATUS "  make run_performance_tests - Run performance benchmarks")
else()
    message(STATUS "Test system disabled")
endif()