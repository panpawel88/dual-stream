#pragma once

#include "CameraFrame.h"
#include "sources/ICameraSource.h"
#include "sources/CameraSourceFactory.h"
#include "processing/ICameraFrameListener.h"
#include "processing/CameraFramePublisher.h"
#include <memory>
#include <string>
#include <mutex>

/**
 * Camera manager state enumeration
 */
enum class CameraManagerState {
    UNINITIALIZED,      // Not initialized
    INITIALIZED,        // Initialized but not capturing
    CAPTURING,          // Currently capturing frames
    ERROR_STATE         // Error state
};

/**
 * Central camera management class that coordinates camera sources and frame delivery.
 * Follows the same pattern as VideoManager for consistency with existing architecture.
 */
class CameraManager {
public:
    CameraManager();
    ~CameraManager();
    
    /**
     * Create a CameraConfig populated from the global configuration system
     * @return CameraConfig with values from Config::GetInstance()
     */
    static CameraConfig CreateCameraConfigFromGlobal();
    
    /**
     * Create a PublisherConfig populated from the global configuration system
     * @return PublisherConfig with values from Config::GetInstance()
     */
    static PublisherConfig CreatePublisherConfigFromGlobal();
    
    /**
     * Initialize camera manager with specified source type and configuration.
     * 
     * @param sourceType Type of camera source to use
     * @param config Camera configuration
     * @param publisherConfig Publisher configuration for frame delivery
     * @return true if initialization successful
     */
    bool Initialize(CameraSourceType sourceType, 
                   const CameraConfig& config = CameraConfig{},
                   const PublisherConfig& publisherConfig = PublisherConfig{});
    
    /**
     * Initialize with specific device.
     * 
     * @param deviceInfo Device information from enumeration
     * @param config Camera configuration
     * @param publisherConfig Publisher configuration
     * @return true if initialization successful
     */
    bool Initialize(const CameraDeviceInfo& deviceInfo,
                   const CameraConfig& config = CameraConfig{},
                   const PublisherConfig& publisherConfig = PublisherConfig{});
    
    /**
     * Initialize with automatic device selection.
     * Uses the best available camera source.
     * 
     * @param config Camera configuration
     * @param publisherConfig Publisher configuration
     * @return true if initialization successful
     */
    bool InitializeAuto(const CameraConfig& config = CameraConfig{},
                       const PublisherConfig& publisherConfig = PublisherConfig{});
    
    /**
     * Start camera capture and frame delivery.
     * 
     * @return true if capture started successfully
     */
    bool StartCapture();
    
    /**
     * Stop camera capture and frame delivery.
     */
    void StopCapture();
    
    /**
     * Clean up all resources.
     */
    void Cleanup();
    
    /**
     * Check if camera is currently capturing.
     * 
     * @return true if camera is capturing
     */
    bool IsCapturing() const;
    
    /**
     * Get current camera manager state.
     * 
     * @return Current state
     */
    CameraManagerState GetState() const;
    
    /**
     * Register a frame processing listener.
     * 
     * @param listener Listener to register
     * @return true if registration successful
     */
    bool RegisterFrameListener(CameraFrameListenerPtr listener);
    
    /**
     * Unregister a frame processing listener.
     * 
     * @param listenerId ID of listener to unregister
     * @return true if listener was found and removed
     */
    bool UnregisterFrameListener(const std::string& listenerId);
    
    /**
     * Unregister listener by pointer.
     * 
     * @param listener Listener to unregister
     * @return true if listener was found and removed
     */
    bool UnregisterFrameListener(CameraFrameListenerPtr listener);
    
    /**
     * Get all registered listeners.
     * 
     * @return Vector of registered listeners
     */
    std::vector<CameraFrameListenerPtr> GetFrameListeners() const;
    
    /**
     * Enable/disable a specific listener.
     * 
     * @param listenerId ID of listener to modify
     * @param enabled true to enable, false to disable
     * @return true if listener was found and modified
     */
    bool SetListenerEnabled(const std::string& listenerId, bool enabled);
    
    /**
     * Capture a single frame synchronously (for testing/debugging).
     * 
     * @param frame Output frame
     * @return true if frame captured successfully
     */
    std::shared_ptr<CameraFrame> CaptureFrame();
    
    /**
     * Get current camera configuration.
     * 
     * @return Camera configuration
     */
    CameraConfig GetCameraConfig() const;
    
    /**
     * Update camera configuration.
     * 
     * @param config New camera configuration
     * @return true if configuration updated successfully
     */
    bool UpdateCameraConfig(const CameraConfig& config);
    
    /**
     * Get current publisher configuration.
     * 
     * @return Publisher configuration
     */
    PublisherConfig GetPublisherConfig() const;
    
    /**
     * Update publisher configuration.
     * 
     * @param config New publisher configuration
     * @return true if configuration updated successfully
     */
    bool UpdatePublisherConfig(const PublisherConfig& config);
    
    /**
     * Get camera device information.
     * 
     * @return Device information
     */
    CameraDeviceInfo GetDeviceInfo() const;
    
    /**
     * Get camera source statistics.
     * 
     * @return Camera statistics
     */
    CameraStats GetCameraStats() const;
    
    /**
     * Get publisher statistics.
     * 
     * @return Publisher statistics
     */
    PublisherStats GetPublisherStats() const;
    
    /**
     * Reset all statistics.
     */
    void ResetStats();
    
    /**
     * Get last error message.
     * 
     * @return Last error message or empty string
     */
    std::string GetLastError() const;
    
    /**
     * Check if camera manager is initialized.
     * 
     * @return true if initialized
     */
    bool IsInitialized() const;
    
    /**
     * Enumerate all available camera devices.
     * 
     * @return Vector of available camera devices
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices();
    
    /**
     * Enumerate devices for specific source type.
     *
     * @param sourceType Source type to enumerate
     * @return Vector of available devices
     */
    static std::vector<CameraDeviceInfo> EnumerateDevices(CameraSourceType sourceType);

    /**
     * Set a camera property at runtime.
     *
     * @param property Property type to set
     * @param value Property value (typically 0-100 range)
     * @return true if property was set successfully
     */
    bool SetCameraProperty(CameraPropertyType property, int value);

    /**
     * Get current value of a camera property.
     *
     * @param property Property type to get
     * @param value Output parameter for property value
     * @return true if property was retrieved successfully
     */
    bool GetCameraProperty(CameraPropertyType property, int& value) const;

    /**
     * Set multiple camera properties at once.
     *
     * @param properties Structure containing properties to set
     * @return true if all properties were set successfully
     */
    bool SetCameraProperties(const CameraProperties& properties);

    /**
     * Get all current camera properties.
     *
     * @return Structure containing all current property values
     */
    CameraProperties GetAllCameraProperties() const;

    /**
     * Get property range information.
     *
     * @param property Property type to query
     * @return Range information for the property
     */
    CameraPropertyRange GetPropertyRange(CameraPropertyType property) const;

private:
    mutable std::mutex m_mutex;
    CameraManagerState m_state;
    std::string m_lastError;
    
    // Core components
    std::unique_ptr<ICameraSource> m_cameraSource;
    std::unique_ptr<CameraFramePublisher> m_publisher;
    
    // Configuration
    CameraDeviceInfo m_deviceInfo;
    CameraConfig m_cameraConfig;
    PublisherConfig m_publisherConfig;
    
    // Frame callback for camera source
    void OnCameraFrame(std::shared_ptr<const CameraFrame> frame);
    
    // Internal helpers
    bool InitializeInternal(const CameraDeviceInfo& deviceInfo,
                           const CameraConfig& cameraConfig,
                           const PublisherConfig& publisherConfig);
    void UpdateLastError(const std::string& error);
    void SetState(CameraManagerState state);
};































