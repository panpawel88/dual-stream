#pragma once

#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

/**
 * Camera frame format enumeration for computer vision processing
 */
enum class CameraFormat {
    BGR8,       // 8-bit BGR (OpenCV default)
    RGB8,       // 8-bit RGB
    BGRA8,      // 8-bit BGRA with alpha
    RGBA8,      // 8-bit RGBA with alpha
    GRAY8,      // 8-bit grayscale
    DEPTH16     // 16-bit depth data (RealSense)
};

/**
 * Camera frame type for processing optimization
 */
enum class CameraFrameType {
    CPU_BUFFER,     // Standard CPU memory buffer
    OPENCV_MAT,     // Direct OpenCV Mat integration
    REALSENSE_FRAME // RealSense native frame format
};

/**
 * Reference-counted frame data for safe multi-consumer access
 */
struct FrameData {
    std::unique_ptr<uint8_t[]> data;
    size_t dataSize;
    int refCount;
    std::mutex refMutex;
    
    FrameData(size_t size) : dataSize(size), refCount(1) {
        data = std::make_unique<uint8_t[]>(size);
    }
    
    void AddRef() {
        std::lock_guard<std::mutex> lock(refMutex);
        refCount++;
    }
    
    bool Release() {
        std::lock_guard<std::mutex> lock(refMutex);
        return --refCount == 0;
    }
};

/**
 * Camera frame abstraction optimized for computer vision processing.
 * Designed to work efficiently with OpenCV and RealSense processing pipelines.
 */
struct CameraFrame {
    CameraFrameType type;
    CameraFormat format;
    int width;
    int height;
    std::chrono::steady_clock::time_point timestamp;
    
    // CPU buffer data
    struct {
        const uint8_t* data;
        int pitch;          // Bytes per row (may include padding)
        cv::Mat mat;        // Direct OpenCV Mat for zero-copy processing
    } cpu;
    
    // RealSense specific data
    struct {
        void* rsFrame;      // rs2::frame pointer (void* to avoid header dependency)
        const uint8_t* depthData;  // Depth data pointer (if available)
        int depthWidth, depthHeight;
    } realsense;
    
    // Reference counting for multi-consumer safety
    std::shared_ptr<FrameData> frameData;
    
    /**
     * Default constructor creates invalid frame
     */
    CameraFrame() : type(CameraFrameType::CPU_BUFFER), format(CameraFormat::BGR8), 
                   width(0), height(0), timestamp(std::chrono::steady_clock::now()) {
        cpu.data = nullptr;
        cpu.pitch = 0;
        realsense.rsFrame = nullptr;
        realsense.depthData = nullptr;
        realsense.depthWidth = realsense.depthHeight = 0;
    }
    
    /**
     * Create CPU buffer frame with OpenCV Mat integration
     */
    static CameraFrame CreateCPUFrame(int width, int height, CameraFormat format, 
                                     const uint8_t* data, int pitch);
    
    /**
     * Create frame directly from OpenCV Mat (zero-copy when possible)
     */
    static CameraFrame CreateFromMat(const cv::Mat& mat, CameraFormat format);
    
    /**
     * Check if frame contains valid data
     */
    bool IsValid() const;
    
    /**
     * Get bytes per pixel for the frame format
     */
    int GetBytesPerPixel() const;
    
    /**
     * Convert camera format to OpenCV Mat type
     */
    static int GetOpenCVType(CameraFormat format);
    
    /**
     * Get format name for debugging
     */
    const char* GetFormatName() const;
};