#include "CameraFrame.h"
#include <cstring>

// Include RealSense headers only in implementation
#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

std::shared_ptr<CameraFrame> CameraFrame::CreateFromMat(
    cv::Mat mat,
    CameraFormat format,
    std::optional<cv::Mat> depth) {

    auto frame = std::make_shared<CameraFrame>();
    frame->format = format;
    frame->width = mat.cols;
    frame->height = mat.rows;
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

#ifdef HAVE_REALSENSE
// Template specialization for RealSense frames
template<>
std::shared_ptr<CameraFrame> CameraFrame::CreateFromRealSense<rs2::video_frame, rs2::depth_frame>(
    const rs2::video_frame& colorFrame,
    const rs2::depth_frame* depthFrame) {

    auto frame = std::make_shared<CameraFrame>();
    frame->timestamp = std::chrono::steady_clock::now();

    // Extract color/IR data into cv::Mat
    int w = colorFrame.get_width();
    int h = colorFrame.get_height();
    auto profile = colorFrame.get_profile().as<rs2::video_stream_profile>();

    frame->width = w;
    frame->height = h;

    // Handle different RealSense formats
    switch (profile.format()) {
        case RS2_FORMAT_BGR8:
            frame->format = CameraFormat::BGR8;
            frame->mat = cv::Mat(h, w, CV_8UC3, (void*)colorFrame.get_data()).clone();
            break;
        case RS2_FORMAT_RGB8:
            frame->format = CameraFormat::RGB8;
            frame->mat = cv::Mat(h, w, CV_8UC3, (void*)colorFrame.get_data()).clone();
            break;
        case RS2_FORMAT_Y8:
            frame->format = CameraFormat::GRAY8;
            frame->mat = cv::Mat(h, w, CV_8UC1, (void*)colorFrame.get_data()).clone();
            break;
        default:
            // Default to BGR8 and convert if needed
            frame->format = CameraFormat::BGR8;
            frame->mat = cv::Mat(h, w, CV_8UC3, (void*)colorFrame.get_data()).clone();
            break;
    }

    // Extract depth if provided
    if (depthFrame) {
        int dw = depthFrame->get_width();
        int dh = depthFrame->get_height();
        frame->depthMat = cv::Mat(dh, dw, CV_16UC1, (void*)depthFrame->get_data()).clone();
    }

    return frame;
}
#endif