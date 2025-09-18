#include "CameraFrame.h"
#include <cstring>

std::shared_ptr<CameraFrame> CameraFrame::CreateFromMat(
    cv::Mat mat,
    CameraFormat format,
    std::optional<cv::Mat> depth) {

    auto frame = std::make_shared<CameraFrame>();
    frame->format = format;
    frame->timestamp = std::chrono::steady_clock::now();

    // Clone mat to ensure we own the data
    frame->mat = mat.clone();

    // Clone depth if provided
    if (depth.has_value()) {
        frame->depthMat = depth->clone();
    }

    return frame;
}


int CameraFrame::GetOpenCVType(CameraFormat format) {
    switch (format) {
        case CameraFormat::BGR8:
            return CV_8UC3;
        case CameraFormat::RGB8:
            return CV_8UC3;
        case CameraFormat::BGRA8:
            return CV_8UC4;
        case CameraFormat::RGBA8:
            return CV_8UC4;
        case CameraFormat::GRAY8:
            return CV_8UC1;
        case CameraFormat::DEPTH16:
            return CV_16UC1;
        default:
            return CV_8UC3;
    }
}

const char* CameraFrame::GetFormatName() const {
    switch (format) {
        case CameraFormat::BGR8: return "BGR8";
        case CameraFormat::RGB8: return "RGB8";
        case CameraFormat::BGRA8: return "BGRA8";
        case CameraFormat::RGBA8: return "RGBA8";
        case CameraFormat::GRAY8: return "GRAY8";
        case CameraFormat::DEPTH16: return "DEPTH16";
        default: return "UNKNOWN";
    }
}

