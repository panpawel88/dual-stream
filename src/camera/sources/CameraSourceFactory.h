#pragma once

#include "ICameraSource.h"
#include "OpenCVCameraSource.h"
#include "RealSenseCameraSource.h"
#include <memory>
#include <vector>
#include <string>

/**
 * Factory class for creating camera source instances.
 * Follows the same pattern as RendererFactory and VideoSwitchingStrategyFactory.
 */
class CameraSourceFactory {
public:
    /**
     * Create a camera source of the specified type.
     * @param type The type of camera source to create
     * @return Unique pointer to camera source instance, or nullptr on failure
     */
    static std::unique_ptr<ICameraSource> Create(CameraSourceType type);
    
    /**
     * Create camera source for a specific device.
     * @param deviceInfo Device information obtained from EnumerateAllDevices()
     * @return Unique pointer to camera source instance, or nullptr on failure
     */
    static std::unique_ptr<ICameraSource> CreateForDevice(const CameraDeviceInfo& deviceInfo);
    
    /**
     * Enumerate all available camera devices from all source types.
     * @return Vector of all available camera devices
     */
    static std::vector<CameraDeviceInfo> EnumerateAllDevices();
    
    /**
     * Enumerate devices for a specific source type.
     * @param type Source type to enumerate
     * @return Vector of available devices for the specified type
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices(CameraSourceType type);
    
    /**
     * Parse camera source type from string name.
     * @param typeName String representation of camera source type
     * @return Parsed camera source type, or OPENCV_WEBCAM if unknown
     */
    static CameraSourceType ParseSourceType(const std::string& typeName);
    
    /**
     * Get human-readable name for camera source type.
     * @param type Camera source type
     * @return Human-readable name
     */
    static std::string GetSourceTypeName(CameraSourceType type);
    
    /**
     * Check if a specific camera source type is available on this system.
     * @param type Camera source type to check
     * @return true if the source type is available
     */
    static bool IsSourceTypeAvailable(CameraSourceType type);
    
    /**
     * Get the best available camera source type for general use.
     * Priority: RealSense -> OpenCV Webcam -> OpenCV Video File
     * @return Best available camera source type
     */
    static CameraSourceType GetBestAvailableSourceType();
    
    /**
     * Create a default camera device info for quick testing.
     * @param type Type of device to create default info for
     * @return Default device info structure
     */
    static CameraDeviceInfo CreateDefaultDevice(CameraSourceType type);

private:
    // Private constructor to prevent instantiation
    CameraSourceFactory() = default;
};

