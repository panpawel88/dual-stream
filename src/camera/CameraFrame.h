#pragma once

#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>

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
 * Camera frame abstraction optimized for computer vision processing.
 * Thread-safe via shared_ptr, non-copyable for safety.
 * All frame data stored as cv::Mat for uniform processing.
 */
struct CameraFrame {
    // Core metadata
    CameraFormat format;
    int width;
    int height;
    std::chrono::steady_clock::time_point timestamp;

    // Frame data - always stored as cv::Mat
    cv::Mat mat;                              // Primary frame data (color/gray/IR)
    std::optional<cv::Mat> depthMat;          // Optional depth channel

    /**
     * Default constructor creates invalid frame
     */
    CameraFrame() : format(CameraFormat::BGR8), width(0), height(0),
                   timestamp(std::chrono::steady_clock::now()) {}

    // Make non-copyable, only moveable
    ~CameraFrame() = default;
    CameraFrame(const CameraFrame&) = delete;
    CameraFrame& operator=(const CameraFrame&) = delete;
    CameraFrame(CameraFrame&&) = default;
    CameraFrame& operator=(CameraFrame&&) = default;

    /**
     * Create frame from OpenCV Mat with optional depth
     */
    static std::shared_ptr<CameraFrame> CreateFromMat(
        cv::Mat mat,
        CameraFormat format,
        std::optional<cv::Mat> depth = std::nullopt);

    /**
     * Create frame from RealSense data
     * Forward declaration to avoid rs2 header dependency
     */
    template<typename VideoFrame, typename DepthFrame>
    static std::shared_ptr<CameraFrame> CreateFromRealSense(
        const VideoFrame& colorFrame,
        const DepthFrame* depthFrame = nullptr);

    /**
     * Check if frame contains valid data
     */
    bool IsValid() const { return !mat.empty(); }

    /**
     * Check if frame has depth data
     */
    bool HasDepth() const { return depthMat.has_value(); }

    /**
     * Convert camera format to OpenCV Mat type
     */
    static int GetOpenCVType(CameraFormat format);

    /**
     * Get format name for debugging
     */
    const char* GetFormatName() const;
};