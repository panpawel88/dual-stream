#include "CameraFrame.h"
#include <cstring>

CameraFrame CameraFrame::CreateCPUFrame(int width, int height, CameraFormat format, 
                                       const uint8_t* data, int pitch) {
    CameraFrame frame;
    frame.type = CameraFrameType::OPENCV_MAT;
    frame.format = format;
    frame.width = width;
    frame.height = height;
    frame.timestamp = std::chrono::steady_clock::now();
    
    // Calculate data size and create reference-counted buffer
    size_t dataSize = pitch * height;
    frame.frameData = std::make_shared<FrameData>(dataSize);
    std::memcpy(frame.frameData->data.get(), data, dataSize);
    
    // Set CPU buffer pointers
    frame.cpu.data = frame.frameData->data.get();
    frame.cpu.pitch = pitch;
    
    // Create OpenCV Mat for zero-copy processing
    int cvType = GetOpenCVType(format);
    frame.cpu.mat = cv::Mat(height, width, cvType, 
                           const_cast<uint8_t*>(frame.cpu.data), pitch);
    
    return frame;
}

CameraFrame CameraFrame::CreateFromMat(const cv::Mat& mat, CameraFormat format) {
    CameraFrame frame;
    frame.type = CameraFrameType::OPENCV_MAT;
    frame.format = format;
    frame.width = mat.cols;
    frame.height = mat.rows;
    frame.timestamp = std::chrono::steady_clock::now();
    
    // Use Mat's data directly if possible (continuous memory)
    if (mat.isContinuous()) {
        size_t dataSize = mat.total() * mat.elemSize();
        frame.frameData = std::make_shared<FrameData>(dataSize);
        std::memcpy(frame.frameData->data.get(), mat.data, dataSize);
        
        frame.cpu.data = frame.frameData->data.get();
        frame.cpu.pitch = static_cast<int>(mat.step[0]);
        frame.cpu.mat = cv::Mat(frame.height, frame.width, mat.type(), 
                               const_cast<uint8_t*>(frame.cpu.data), frame.cpu.pitch);
    } else {
        // Copy to continuous buffer
        cv::Mat continuous;
        mat.copyTo(continuous);
        return CreateFromMat(continuous, format);
    }
    
    return frame;
}

bool CameraFrame::IsValid() const {
    return width > 0 && height > 0 && cpu.data != nullptr;
}

int CameraFrame::GetBytesPerPixel() const {
    switch (format) {
        case CameraFormat::BGR8:
        case CameraFormat::RGB8:
            return 3;
        case CameraFormat::BGRA8:
        case CameraFormat::RGBA8:
            return 4;
        case CameraFormat::GRAY8:
            return 1;
        case CameraFormat::DEPTH16:
            return 2;
        default:
            return 0;
    }
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