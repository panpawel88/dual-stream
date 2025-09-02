#pragma once

// Windows headers
#ifdef _WIN32
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <windowsx.h>
    #include <dwmapi.h>
#endif

// Standard library headers
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <optional>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include <filesystem>

// FFmpeg headers
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/pixdesc.h>
    #include <libswscale/swscale.h>
}

// DirectX headers (conditional)
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

// CUDA headers (conditional) - OpenGL interop moved to specific files to avoid conflicts
#ifdef HAVE_CUDA
    #include <cuda_runtime.h>
    // Note: cuda_gl_interop.h moved to CudaOpenGLInterop.cpp to avoid GLAD conflicts
#endif

// OpenCV headers (conditional)
#ifdef HAVE_OPENCV
    #include <opencv2/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include <opencv2/videoio.hpp>
    #include <opencv2/objdetect.hpp>
    #include <opencv2/highgui.hpp>
    #ifdef HAVE_OPENCV_DNN
        #include <opencv2/dnn.hpp>
    #endif
#endif

// Intel RealSense headers (conditional)
#ifdef HAVE_REALSENSE
    #include <librealsense2/rs.hpp>
#endif