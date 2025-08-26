#include "CameraSourceFactory.h"
#include <algorithm>
#include <cctype>

std::unique_ptr<ICameraSource> CameraSourceFactory::Create(CameraSourceType type) {
    switch (type) {
        case CameraSourceType::OPENCV_WEBCAM:
        case CameraSourceType::OPENCV_VIDEO_FILE:
            return std::make_unique<OpenCVCameraSource>();
            
        case CameraSourceType::REALSENSE_DEVICE:
            return std::make_unique<RealSenseCameraSource>();
            
        case CameraSourceType::MOCK_CAMERA:
            // Could implement MockCameraSource for testing
            return nullptr;
            
        default:
            return nullptr;
    }
}

std::unique_ptr<ICameraSource> CameraSourceFactory::CreateForDevice(const CameraDeviceInfo& deviceInfo) {
    auto source = Create(deviceInfo.type);
    if (!source) {
        return nullptr;
    }
    
    // Check if the source type is available before returning
    if (!source->IsAvailable()) {
        return nullptr;
    }
    
    return source;
}

std::vector<CameraDeviceInfo> CameraSourceFactory::EnumerateAllDevices() {
    std::vector<CameraDeviceInfo> allDevices;
    
    // Enumerate OpenCV devices
    if (IsSourceTypeAvailable(CameraSourceType::OPENCV_WEBCAM)) {
        auto opencvDevices = OpenCVCameraSource::EnumerateDevices();
        allDevices.insert(allDevices.end(), opencvDevices.begin(), opencvDevices.end());
    }
    
    // Enumerate RealSense devices
    if (IsSourceTypeAvailable(CameraSourceType::REALSENSE_DEVICE)) {
        auto realsenseDevices = RealSenseCameraSource::EnumerateDevices();
        allDevices.insert(allDevices.end(), realsenseDevices.begin(), realsenseDevices.end());
    }
    
    return allDevices;
}

std::vector<CameraDeviceInfo> CameraSourceFactory::EnumerateDevices(CameraSourceType type) {
    switch (type) {
        case CameraSourceType::OPENCV_WEBCAM:
        case CameraSourceType::OPENCV_VIDEO_FILE:
            if (IsSourceTypeAvailable(type)) {
                return OpenCVCameraSource::EnumerateDevices();
            }
            break;
            
        case CameraSourceType::REALSENSE_DEVICE:
            if (IsSourceTypeAvailable(type)) {
                return RealSenseCameraSource::EnumerateDevices();
            }
            break;
            
        case CameraSourceType::MOCK_CAMERA:
            // Could return mock devices for testing
            break;
    }
    
    return {};
}

CameraSourceType CameraSourceFactory::ParseSourceType(const std::string& typeName) {
    std::string lowerName = typeName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    if (lowerName == "opencv" || lowerName == "webcam" || lowerName == "opencv_webcam") {
        return CameraSourceType::OPENCV_WEBCAM;
    } else if (lowerName == "opencv_file" || lowerName == "video_file" || lowerName == "file") {
        return CameraSourceType::OPENCV_VIDEO_FILE;
    } else if (lowerName == "realsense" || lowerName == "realsense_device" || lowerName == "depth") {
        return CameraSourceType::REALSENSE_DEVICE;
    } else if (lowerName == "mock" || lowerName == "test" || lowerName == "mock_camera") {
        return CameraSourceType::MOCK_CAMERA;
    }
    
    // Default to OpenCV webcam
    return CameraSourceType::OPENCV_WEBCAM;
}

std::string CameraSourceFactory::GetSourceTypeName(CameraSourceType type) {
    switch (type) {
        case CameraSourceType::OPENCV_WEBCAM:
            return "OpenCV Webcam";
        case CameraSourceType::OPENCV_VIDEO_FILE:
            return "OpenCV Video File";
        case CameraSourceType::REALSENSE_DEVICE:
            return "Intel RealSense";
        case CameraSourceType::MOCK_CAMERA:
            return "Mock Camera";
        default:
            return "Unknown";
    }
}

bool CameraSourceFactory::IsSourceTypeAvailable(CameraSourceType type) {
    switch (type) {
        case CameraSourceType::OPENCV_WEBCAM:
        case CameraSourceType::OPENCV_VIDEO_FILE: {
            auto source = std::make_unique<OpenCVCameraSource>();
            return source->IsAvailable();
        }
        
        case CameraSourceType::REALSENSE_DEVICE: {
            auto source = std::make_unique<RealSenseCameraSource>();
            return source->IsAvailable();
        }
        
        case CameraSourceType::MOCK_CAMERA:
            return false; // Not implemented yet
            
        default:
            return false;
    }
}

CameraSourceType CameraSourceFactory::GetBestAvailableSourceType() {
    // Priority order: RealSense (best features) -> OpenCV Webcam (widely available)
    if (IsSourceTypeAvailable(CameraSourceType::REALSENSE_DEVICE)) {
        return CameraSourceType::REALSENSE_DEVICE;
    }
    
    if (IsSourceTypeAvailable(CameraSourceType::OPENCV_WEBCAM)) {
        return CameraSourceType::OPENCV_WEBCAM;
    }
    
    // Fallback to OpenCV (might not work, but it's the most basic)
    return CameraSourceType::OPENCV_WEBCAM;
}

CameraDeviceInfo CameraSourceFactory::CreateDefaultDevice(CameraSourceType type) {
    switch (type) {
        case CameraSourceType::OPENCV_WEBCAM:
            return OpenCVCameraSource::CreateWebcamDevice(0);  // First webcam
            
        case CameraSourceType::OPENCV_VIDEO_FILE:
            return OpenCVCameraSource::CreateVideoFileDevice("test_video.mp4");
            
        case CameraSourceType::REALSENSE_DEVICE: {
            CameraDeviceInfo info;
            info.deviceIndex = 0;
            info.deviceName = "RealSense Device";
            info.serialNumber = "realsense_default";
            info.type = CameraSourceType::REALSENSE_DEVICE;
            info.maxWidth = 640;
            info.maxHeight = 480;
            info.maxFrameRate = 30.0;
            info.supportsDepth = true;
            return info;
        }
        
        case CameraSourceType::MOCK_CAMERA: {
            CameraDeviceInfo info;
            info.deviceIndex = 0;
            info.deviceName = "Mock Camera";
            info.serialNumber = "mock_device";
            info.type = CameraSourceType::MOCK_CAMERA;
            info.maxWidth = 640;
            info.maxHeight = 480;
            info.maxFrameRate = 30.0;
            info.supportsDepth = false;
            return info;
        }
        
        default: {
            // Return invalid device info
            CameraDeviceInfo info;
            return info;
        }
    }
}