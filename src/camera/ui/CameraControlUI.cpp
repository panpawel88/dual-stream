#include "CameraControlUI.h"
#include "../../core/Logger.h"
#include "../../core/Config.h"
#include <imgui.h>
#include <algorithm>
#include <cmath>

CameraControlUI::CameraControlUI()
    : m_cameraManager(nullptr)
    , m_renderer(nullptr)
    , m_frameTexture(nullptr)
    , m_previewEnabled(true)
    , m_brightness(50)
    , m_contrast(50)
    , m_saturation(50)
    , m_gain(50)
    , m_currentFrame(nullptr)
    , m_hasNewFrame(false)
    , m_previewFPS(10.0)
    , m_maxPreviewFPS(10.0)
    , m_initialized(false)
    , m_cameraAvailable(false)
    , m_frameWidth(0)
    , m_frameHeight(0)
    , m_currentFPS(0.0)
{
    m_lastPreviewUpdate = std::chrono::steady_clock::now();
}

CameraControlUI::~CameraControlUI() {
    Cleanup();
}

bool CameraControlUI::Initialize(CameraManager* cameraManager, IRenderer* renderer) {
    if (!cameraManager || !renderer) {
        LOG_ERROR("CameraControlUI::Initialize: null parameters");
        return false;
    }

    m_cameraManager = cameraManager;
    m_renderer = renderer;

    // Create frame texture converter
    m_frameTexture = std::make_unique<CameraFrameTexture>();
    if (!m_frameTexture->Initialize(renderer)) {
        LOG_ERROR("CameraControlUI::Initialize: Failed to initialize frame texture");
        return false;
    }

    // Load configuration
    LoadConfigurationSettings();

    // Register as frame listener
    if (m_cameraManager->IsInitialized()) {
        m_cameraManager->RegisterFrameListener(this);
        m_cameraAvailable = true;

        // Sync UI with current camera properties
        SyncUIWithCameraProperties();
    }

    m_initialized = true;
    LOG_INFO("CameraControlUI: Initialized successfully");
    return true;
}

void CameraControlUI::Cleanup() {
    if (m_cameraManager && m_initialized) {
        m_cameraManager->UnregisterFrameListener(this);
    }

    if (m_frameTexture) {
        m_frameTexture->Cleanup();
        m_frameTexture.reset();
    }

    SaveConfigurationSettings();

    m_initialized = false;
    m_cameraAvailable = false;
    m_cameraManager = nullptr;
    m_renderer = nullptr;
}

void CameraControlUI::DrawUI() {
    if (!m_initialized) {
        ImGui::Text("Camera Control UI not initialized");
        return;
    }

    ImGui::PushID("CameraControlUI");

    // Check camera availability
    m_cameraAvailable = m_cameraManager && m_cameraManager->IsInitialized() && m_cameraManager->IsCapturing();

    if (!m_cameraAvailable) {
        ImGui::Text("Camera not available");
        if (ImGui::Button("Refresh Camera Status")) {
            if (m_cameraManager && m_cameraManager->IsInitialized()) {
                m_cameraManager->RegisterFrameListener(this);
                SyncUIWithCameraProperties();
            }
        }
        ImGui::PopID();
        return;
    }

    // Draw camera control sections
    DrawCameraInfo();
    ImGui::Separator();

    DrawCameraProperties();
    ImGui::Separator();

    DrawFramePreview();

    ImGui::PopID();
}

void CameraControlUI::DrawCameraInfo() {
    ImGui::Text("Camera Information");

    if (m_cameraManager) {
        auto deviceInfo = m_cameraManager->GetDeviceInfo();
        auto stats = m_cameraManager->GetCameraStats();

        ImGui::Text("Device: %s", deviceInfo.deviceName.c_str());
        ImGui::Text("Resolution: %dx%d", m_frameWidth, m_frameHeight);
        ImGui::Text("FPS: %.1f", stats.averageFrameRate);
        ImGui::Text("Frames: %llu received, %llu dropped",
                   static_cast<unsigned long long>(stats.framesReceived),
                   static_cast<unsigned long long>(stats.framesDropped));
    }
}

void CameraControlUI::DrawCameraProperties() {
    ImGui::Text("Camera Properties");

    bool propertyChanged = false;

    // Brightness
    if (ImGui::SliderInt("Brightness", &m_brightness, 0, 100)) {
        UpdateCameraProperty(CameraPropertyType::BRIGHTNESS, m_brightness);
        propertyChanged = true;
    }

    // Contrast
    if (ImGui::SliderInt("Contrast", &m_contrast, 0, 100)) {
        UpdateCameraProperty(CameraPropertyType::CONTRAST, m_contrast);
        propertyChanged = true;
    }


    // Saturation
    if (ImGui::SliderInt("Saturation", &m_saturation, 0, 100)) {
        UpdateCameraProperty(CameraPropertyType::SATURATION, m_saturation);
        propertyChanged = true;
    }

    // Gain
    if (ImGui::SliderInt("Gain", &m_gain, 0, 100)) {
        UpdateCameraProperty(CameraPropertyType::GAIN, m_gain);
        propertyChanged = true;
    }


    // Reset button
    if (ImGui::Button("Reset to Defaults")) {
        ResetPropertiesToDefaults();
        propertyChanged = true;
    }

    if (propertyChanged) {
        SaveConfigurationSettings();
    }
}

void CameraControlUI::DrawFramePreview() {
    ImGui::Text("Frame Preview");

    // Preview toggle
    if (ImGui::Checkbox("Enable Preview", &m_previewEnabled)) {
        SaveConfigurationSettings();
    }

    if (!m_previewEnabled) {
        return;
    }

    // Preview FPS slider
    float previewFPS = static_cast<float>(m_maxPreviewFPS);
    if (ImGui::SliderFloat("Preview FPS", &previewFPS, 1.0f, 30.0f, "%.1f")) {
        m_maxPreviewFPS = static_cast<double>(previewFPS);
        SaveConfigurationSettings();
    }

    // Update frame texture if needed
    if (m_hasNewFrame && ShouldUpdatePreview()) {
        UpdateFrameTexture();
    }

    // Display frame
    if (m_frameTexture && m_frameTexture->IsValid()) {
        int textureWidth, textureHeight;
        m_frameTexture->GetTextureDimensions(textureWidth, textureHeight);

        if (textureWidth > 0 && textureHeight > 0) {
            void* textureID = m_frameTexture->GetImGuiTextureID();
            if (textureID) {
                // Calculate display size (fit within a reasonable area)
                float maxDisplayWidth = 400.0f;
                float maxDisplayHeight = 300.0f;

                float aspectRatio = static_cast<float>(textureWidth) / textureHeight;
                float displayWidth = std::min(maxDisplayWidth, static_cast<float>(textureWidth));
                float displayHeight = displayWidth / aspectRatio;

                if (displayHeight > maxDisplayHeight) {
                    displayHeight = maxDisplayHeight;
                    displayWidth = displayHeight * aspectRatio;
                }

                ImGui::Image(textureID, ImVec2(displayWidth, displayHeight));

                // Show preview info
                ImGui::Text("Preview: %dx%d @ %.1f FPS", textureWidth, textureHeight, m_previewFPS);
            }
        }
    } else {
        ImGui::Text("No frame available");
    }
}

FrameProcessingResult CameraControlUI::ProcessFrame(std::shared_ptr<const CameraFrame> frame) {
    auto start = std::chrono::steady_clock::now();

    m_stats.framesReceived++;

    if (!frame || !m_previewEnabled) {
        m_stats.framesSkipped++;
        return FrameProcessingResult::SUCCESS;
    }

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_currentFrame = frame;
        m_hasNewFrame = true;

        // Update frame info
        m_frameWidth = frame->mat.cols;
        m_frameHeight = frame->mat.rows;
    }

    m_stats.framesProcessed++;

    // Update processing time stats
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double processingTimeMs = elapsed.count() / 1000.0;

    m_stats.maxProcessingTimeMs = std::max(m_stats.maxProcessingTimeMs, processingTimeMs);

    // Simple moving average for processing time
    const double alpha = 0.1;
    m_stats.averageProcessingTimeMs = m_stats.averageProcessingTimeMs * (1.0 - alpha) + processingTimeMs * alpha;

    m_stats.lastFrameTime = end;

    return FrameProcessingResult::SUCCESS;
}

bool CameraControlUI::CanProcessFormat(CameraFormat format) const {
    // We can process all common camera formats
    switch (format) {
        case CameraFormat::BGR8:
        case CameraFormat::RGB8:
        case CameraFormat::GRAY8:
        case CameraFormat::BGRA8:
        case CameraFormat::RGBA8:
            return true;
        default:
            return false;
    }
}

void CameraControlUI::UpdateCameraProperty(CameraPropertyType property, int value) {
    if (!m_cameraManager) {
        return;
    }

    // Convert UI percentage (0-100) to normalized value (0.0-1.0)
    double normalizedValue = value / 100.0;

    LOG_DEBUG("CameraControlUI: Updating property ", static_cast<int>(property), " to ", value, "% (normalized: ", normalizedValue, ")");

    if (m_cameraManager->SetCameraProperty(property, normalizedValue)) {
        LOG_DEBUG("CameraControlUI: Successfully updated property ", static_cast<int>(property), " to ", value);

        // For property changes that affect UI state
        if (false) { // Simplified since no exposure control
            // Small delay to allow camera to process the change
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            SyncUIWithCameraProperties();
        }
    } else {
        LOG_WARNING("CameraControlUI: Failed to update property ", static_cast<int>(property), " to ", value);
        // Sync UI back with actual camera values
        SyncUIWithCameraProperties();
    }
}

void CameraControlUI::ResetPropertiesToDefaults() {
    if (!m_cameraManager) {
        return;
    }

    // Set UI values to 50% (reasonable default)
    m_brightness = 50;
    m_contrast = 50;
    m_saturation = 50;
    m_gain = 50;

    LOG_INFO("Resetting camera properties to defaults: All properties set to 50%");

    // Apply the default properties (convert to normalized values)
    CameraProperties defaultProps;
    defaultProps.brightness = 0.5;  // 50% = 0.5 normalized
    defaultProps.contrast = 0.5;
    defaultProps.saturation = 0.5;
    defaultProps.gain = 0.5;


    if (!m_cameraManager->SetCameraProperties(defaultProps)) {
        LOG_WARNING("Failed to reset some camera properties to defaults");
        // Sync UI with actual camera values
        SyncUIWithCameraProperties();
    }
}

void CameraControlUI::SyncUIWithCameraProperties() {
    if (!m_cameraManager) {
        return;
    }

    auto properties = m_cameraManager->GetAllCameraProperties();

    // Convert normalized values (0.0-1.0) to UI percentage (0-100)
    // Only update UI values if they're valid (not NaN)
    if (!std::isnan(properties.brightness)) {
        m_brightness = static_cast<int>(properties.brightness * 100.0);
        LOG_DEBUG("Synced brightness to ", m_brightness, "% (normalized: ", properties.brightness, ")");
    }
    if (!std::isnan(properties.contrast)) {
        m_contrast = static_cast<int>(properties.contrast * 100.0);
        LOG_DEBUG("Synced contrast to ", m_contrast, "% (normalized: ", properties.contrast, ")");
    }
    if (!std::isnan(properties.saturation)) {
        m_saturation = static_cast<int>(properties.saturation * 100.0);
        LOG_DEBUG("Synced saturation to ", m_saturation, "% (normalized: ", properties.saturation, ")");
    }
    if (!std::isnan(properties.gain)) {
        m_gain = static_cast<int>(properties.gain * 100.0);
        LOG_DEBUG("Synced gain to ", m_gain, "% (normalized: ", properties.gain, ")");
    }


}

bool CameraControlUI::ShouldUpdatePreview() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastPreviewUpdate);
    double targetInterval = 1000.0 / m_maxPreviewFPS;

    return elapsed.count() >= targetInterval;
}

void CameraControlUI::UpdateFrameTexture() {
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (m_currentFrame && m_frameTexture) {
        if (m_frameTexture->UpdateTexture(m_currentFrame)) {
            m_hasNewFrame = false;
            m_lastPreviewUpdate = std::chrono::steady_clock::now();

            // Update preview FPS calculation
            static auto lastFpsUpdate = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsUpdate);
            if (elapsed.count() > 0) {
                m_previewFPS = 1000.0 / elapsed.count();
                lastFpsUpdate = now;
            }
        }
    }
}

void CameraControlUI::LoadConfigurationSettings() {
    auto* config = Config::GetInstance();

    m_previewEnabled = config->GetBool("camera_ui.preview_enabled", true);
    m_maxPreviewFPS = config->GetDouble("camera_ui.preview_fps", 10.0);

    // Set max dimensions on frame texture
    int maxWidth = config->GetInt("camera_ui.preview_max_width", 640);
    int maxHeight = config->GetInt("camera_ui.preview_max_height", 480);

    if (m_frameTexture) {
        m_frameTexture->SetMaxDimensions(maxWidth, maxHeight);
    }
}

void CameraControlUI::SaveConfigurationSettings() {
    auto* config = Config::GetInstance();

    config->SetBool("camera_ui.preview_enabled", m_previewEnabled);
    config->SetDouble("camera_ui.preview_fps", m_maxPreviewFPS);
}

FrameProcessingStats CameraControlUI::GetStats() const {
    return m_stats;
}

void CameraControlUI::ResetStats() {
    m_stats.Reset();
}

const char* CameraControlUI::FormatToString(CameraFormat format) const {
    switch (format) {
        case CameraFormat::BGR8: return "BGR8";
        case CameraFormat::RGB8: return "RGB8";
        case CameraFormat::GRAY8: return "GRAY8";
        case CameraFormat::BGRA8: return "BGRA8";
        case CameraFormat::RGBA8: return "RGBA8";
        case CameraFormat::DEPTH16: return "DEPTH16";
        default: return "Unknown";
    }
}