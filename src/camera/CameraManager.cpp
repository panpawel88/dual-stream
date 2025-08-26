#include "CameraManager.h"

CameraManager::CameraManager() 
    : m_state(CameraManagerState::UNINITIALIZED) {
}

CameraManager::~CameraManager() {
    Cleanup();
}

bool CameraManager::Initialize(CameraSourceType sourceType, 
                             const CameraConfig& config,
                             const PublisherConfig& publisherConfig) {
    CameraDeviceInfo defaultDevice = CameraSourceFactory::CreateDefaultDevice(sourceType);
    return InitializeInternal(defaultDevice, config, publisherConfig);
}

bool CameraManager::Initialize(const CameraDeviceInfo& deviceInfo,
                             const CameraConfig& config,
                             const PublisherConfig& publisherConfig) {
    return InitializeInternal(deviceInfo, config, publisherConfig);
}

bool CameraManager::InitializeAuto(const CameraConfig& config,
                                 const PublisherConfig& publisherConfig) {
    auto devices = EnumerateDevices();
    if (devices.empty()) {
        UpdateLastError("No camera devices found");
        return false;
    }
    
    // Use first available device
    return InitializeInternal(devices[0], config, publisherConfig);
}

bool CameraManager::InitializeInternal(const CameraDeviceInfo& deviceInfo,
                                     const CameraConfig& cameraConfig,
                                     const PublisherConfig& publisherConfig) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_state != CameraManagerState::UNINITIALIZED) {
        Cleanup();
    }
    
    // Create camera source
    m_cameraSource = CameraSourceFactory::CreateForDevice(deviceInfo);
    if (!m_cameraSource) {
        UpdateLastError("Failed to create camera source for device: " + deviceInfo.deviceName);
        SetState(CameraManagerState::ERROR);
        return false;
    }
    
    // Initialize camera source
    if (!m_cameraSource->Initialize(deviceInfo, cameraConfig)) {
        UpdateLastError("Failed to initialize camera source: " + m_cameraSource->GetLastError());
        SetState(CameraManagerState::ERROR);
        return false;
    }
    
    // Create and start publisher
    m_publisher = std::make_unique<CameraFramePublisher>(publisherConfig);
    if (!m_publisher->Start()) {
        UpdateLastError("Failed to start camera frame publisher");
        SetState(CameraManagerState::ERROR);
        return false;
    }
    
    // Set up camera frame callback
    m_cameraSource->SetFrameCallback([this](const CameraFrame& frame) {
        OnCameraFrame(frame);
    });
    
    // Store configuration
    m_deviceInfo = deviceInfo;
    m_cameraConfig = cameraConfig;
    m_publisherConfig = publisherConfig;
    
    SetState(CameraManagerState::INITIALIZED);
    return true;
}

bool CameraManager::StartCapture() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_state != CameraManagerState::INITIALIZED) {
        UpdateLastError("Camera manager not initialized");
        return false;
    }
    
    if (!m_cameraSource->StartCapture()) {
        UpdateLastError("Failed to start camera capture: " + m_cameraSource->GetLastError());
        SetState(CameraManagerState::ERROR);
        return false;
    }
    
    SetState(CameraManagerState::CAPTURING);
    return true;
}

void CameraManager::StopCapture() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_state == CameraManagerState::CAPTURING && m_cameraSource) {
        m_cameraSource->StopCapture();
        SetState(CameraManagerState::INITIALIZED);
    }
}

void CameraManager::Cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_cameraSource) {
        m_cameraSource->StopCapture();
        m_cameraSource.reset();
    }
    
    if (m_publisher) {
        m_publisher->Stop();
        m_publisher.reset();
    }
    
    SetState(CameraManagerState::UNINITIALIZED);
}

bool CameraManager::IsCapturing() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state == CameraManagerState::CAPTURING;
}

CameraManagerState CameraManager::GetState() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
}

bool CameraManager::RegisterFrameListener(CameraFrameListenerPtr listener) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        UpdateLastError("Publisher not initialized");
        return false;
    }
    
    return m_publisher->RegisterListener(listener);
}

bool CameraManager::UnregisterFrameListener(const std::string& listenerId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        return false;
    }
    
    return m_publisher->UnregisterListener(listenerId);
}

bool CameraManager::UnregisterFrameListener(CameraFrameListenerPtr listener) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        return false;
    }
    
    return m_publisher->UnregisterListener(listener);
}

std::vector<CameraFrameListenerPtr> CameraManager::GetFrameListeners() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        return {};
    }
    
    return m_publisher->GetListeners();
}

bool CameraManager::SetListenerEnabled(const std::string& listenerId, bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        return false;
    }
    
    return m_publisher->SetListenerEnabled(listenerId, enabled);
}

bool CameraManager::CaptureFrame(CameraFrame& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_cameraSource) {
        UpdateLastError("Camera source not initialized");
        return false;
    }
    
    return m_cameraSource->CaptureFrame(frame);
}

CameraConfig CameraManager::GetCameraConfig() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_cameraSource) {
        return m_cameraSource->GetConfig();
    }
    
    return m_cameraConfig;
}

bool CameraManager::UpdateCameraConfig(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_cameraSource) {
        UpdateLastError("Camera source not initialized");
        return false;
    }
    
    if (m_cameraSource->UpdateConfig(config)) {
        m_cameraConfig = config;
        
        // Notify listeners of configuration change
        if (m_publisher) {
            m_publisher->NotifyConfigChange(config);
        }
        
        return true;
    }
    
    UpdateLastError("Failed to update camera configuration: " + m_cameraSource->GetLastError());
    return false;
}

PublisherConfig CameraManager::GetPublisherConfig() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_publisher) {
        return m_publisher->GetConfig();
    }
    
    return m_publisherConfig;
}

bool CameraManager::UpdatePublisherConfig(const PublisherConfig& config) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_publisher) {
        UpdateLastError("Publisher not initialized");
        return false;
    }
    
    if (m_publisher->UpdateConfig(config)) {
        m_publisherConfig = config;
        return true;
    }
    
    UpdateLastError("Failed to update publisher configuration");
    return false;
}

CameraDeviceInfo CameraManager::GetDeviceInfo() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_cameraSource) {
        return m_cameraSource->GetDeviceInfo();
    }
    
    return m_deviceInfo;
}

CameraStats CameraManager::GetCameraStats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_cameraSource) {
        return m_cameraSource->GetStats();
    }
    
    return CameraStats{};
}

PublisherStats CameraManager::GetPublisherStats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_publisher) {
        return m_publisher->GetStats();
    }
    
    return PublisherStats{};
}

void CameraManager::ResetStats() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_cameraSource) {
        m_cameraSource->ResetStats();
    }
    
    if (m_publisher) {
        m_publisher->ResetStats();
    }
}

std::string CameraManager::GetLastError() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_lastError;
}

bool CameraManager::IsInitialized() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state != CameraManagerState::UNINITIALIZED;
}

std::vector<CameraDeviceInfo> CameraManager::EnumerateDevices() {
    return CameraSourceFactory::EnumerateAllDevices();
}

std::vector<CameraDeviceInfo> CameraManager::EnumerateDevices(CameraSourceType sourceType) {
    return CameraSourceFactory::EnumerateDevices(sourceType);
}

void CameraManager::OnCameraFrame(const CameraFrame& frame) {
    // This callback is called from camera source thread
    if (m_publisher && frame.IsValid()) {
        m_publisher->PublishFrame(frame);
    }
}

void CameraManager::UpdateLastError(const std::string& error) {
    m_lastError = error;
}

void CameraManager::SetState(CameraManagerState state) {
    m_state = state;
}