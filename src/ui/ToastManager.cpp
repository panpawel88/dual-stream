#include "ToastManager.h"
#include "core/Config.h"
#include "core/Logger.h"
#include "rendering/IToastRenderer.h"
#include <algorithm>
#include <sstream>

ToastManager& ToastManager::GetInstance() {
    static ToastManager instance;
    return instance;
}

void ToastManager::Initialize() {
    LoadConfig();
    LOG_DEBUG("ToastManager initialized. Enabled: ", m_config.enabled ? "true" : "false");
}

void ToastManager::SetRenderer(std::unique_ptr<IToastRenderer> renderer) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_renderer = std::move(renderer);
    
    if (m_renderer && m_config.enabled) {
        if (!m_renderer->Initialize(m_config)) {
            LOG_ERROR("Failed to initialize toast renderer");
            m_renderer.reset();
        } else {
            LOG_DEBUG("Toast renderer initialized successfully");
        }
    }
}

void ToastManager::ShowToast(const std::string& message, ToastType type) {
    if (!m_config.enabled || message.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Remove any existing message with the same text to avoid duplicates
    m_messages.erase(
        std::remove_if(m_messages.begin(), m_messages.end(),
            [&message](const ToastMessage& toast) { return toast.text == message; }),
        m_messages.end());
    
    ToastMessage toast;
    toast.text = message;
    toast.type = type;
    toast.duration = m_config.duration;
    toast.fadeInTime = m_config.fadeInTime;
    toast.fadeOutTime = m_config.fadeOutTime;
    toast.currentAlpha = 0.0f;
    toast.startTime = std::chrono::steady_clock::now();
    toast.state = ToastState::FADE_IN;
    
    m_messages.push_back(toast);
    
    LOG_DEBUG("Toast added: ", message);
}

void ToastManager::ShowTriggerEvent(const std::string& triggerName, int targetVideoIndex) {
    if (!m_config.showTriggerEvents) {
        return;
    }
    
    std::string message = triggerName + ": Switching to Video " + std::to_string(targetVideoIndex + 1);
    ShowToast(message, ToastType::INFO);
}

void ToastManager::ShowStrategyEvent(const std::string& strategyName, int videoIndex, const std::string& action) {
    if (!m_config.showStrategyEvents) {
        return;
    }
    
    std::string message = strategyName + ": " + action + " Video " + std::to_string(videoIndex + 1);
    ShowToast(message, ToastType::INFO);
}

void ToastManager::ShowKeyframeScheduled(int targetVideoIndex) {
    if (!m_config.showKeyframeEvents) {
        return;
    }
    
    std::string message = "Keyframe sync: Switch scheduled to Video " + std::to_string(targetVideoIndex + 1);
    ShowToast(message, ToastType::DEBUG);
}

void ToastManager::ShowKeyframeCompleted(int videoIndex) {
    if (!m_config.showKeyframeEvents) {
        return;
    }
    
    std::string message = "Keyframe sync: Switched to Video " + std::to_string(videoIndex + 1);
    ShowToast(message, ToastType::INFO);
}

void ToastManager::Update() {
    if (!m_config.enabled || !m_renderer) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Update all active toast animations
    for (auto& toast : m_messages) {
        UpdateToastAnimation(toast);
    }
    
    // Remove completed toasts
    RemoveCompletedToasts();
}

void ToastManager::Render() {
    if (!m_config.enabled || !m_renderer || m_messages.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Render all active toasts
    for (const auto& toast : m_messages) {
        if (toast.currentAlpha > 0.0f) {
            m_renderer->RenderToast(toast);
        }
    }
}

void ToastManager::Cleanup() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    m_messages.clear();
    if (m_renderer) {
        m_renderer->Cleanup();
        m_renderer.reset();
    }
    
    LOG_DEBUG("ToastManager cleaned up");
}

void ToastManager::LoadConfig() {
    Config* config = Config::GetInstance();
    
    m_config.enabled = config->GetBool("toast.enabled", false);
    m_config.duration = config->GetFloat("toast.duration_ms", 2000.0f) / 1000.0f;  // Convert to seconds
    m_config.fadeInTime = config->GetFloat("toast.fade_in_ms", 200.0f) / 1000.0f;  // Convert to seconds
    m_config.fadeOutTime = config->GetFloat("toast.fade_out_ms", 300.0f) / 1000.0f;  // Convert to seconds
    
    m_config.position = ParsePosition(config->GetString("toast.position", "bottom_center"));
    m_config.offsetX = config->GetInt("toast.offset_x", 0);
    m_config.offsetY = config->GetInt("toast.offset_y", 100);
    m_config.maxWidth = config->GetInt("toast.max_width", 400);
    m_config.fontSize = config->GetInt("toast.font_size", 14);
    m_config.cornerRadius = config->GetInt("toast.corner_radius", 8);
    m_config.padding = config->GetInt("toast.padding", 12);
    
    m_config.backgroundColor = ParseColor(config->GetString("toast.background_color", "0,0,0,180"), ToastConfig::Color(0, 0, 0, 180));
    m_config.textColor = ParseColor(config->GetString("toast.text_color", "255,255,255,255"), ToastConfig::Color(255, 255, 255, 255));
    
    m_config.showTriggerEvents = config->GetBool("toast.show_trigger_events", true);
    m_config.showStrategyEvents = config->GetBool("toast.show_strategy_events", true);
    m_config.showKeyframeEvents = config->GetBool("toast.show_keyframe_events", true);
}

ToastConfig::Color ToastManager::ParseColor(const std::string& colorStr, const ToastConfig::Color& defaultColor) {
    std::istringstream iss(colorStr);
    std::string token;
    std::vector<int> values;
    
    while (std::getline(iss, token, ',')) {
        try {
            int value = std::stoi(token);
            values.push_back(std::max(0, std::min(255, value)));  // Clamp to 0-255
        } catch (const std::exception&) {
            LOG_WARNING("Invalid color component in: ", colorStr);
            return defaultColor;
        }
    }
    
    if (values.size() != 4) {
        LOG_WARNING("Color must have 4 components (RGBA): ", colorStr);
        return defaultColor;
    }
    
    return ToastConfig::Color(
        static_cast<uint8_t>(values[0]),
        static_cast<uint8_t>(values[1]),
        static_cast<uint8_t>(values[2]),
        static_cast<uint8_t>(values[3])
    );
}

ToastPosition ToastManager::ParsePosition(const std::string& positionStr) {
    if (positionStr == "top_left") return ToastPosition::TOP_LEFT;
    if (positionStr == "top_center") return ToastPosition::TOP_CENTER;
    if (positionStr == "top_right") return ToastPosition::TOP_RIGHT;
    if (positionStr == "center_left") return ToastPosition::CENTER_LEFT;
    if (positionStr == "center") return ToastPosition::CENTER;
    if (positionStr == "center_right") return ToastPosition::CENTER_RIGHT;
    if (positionStr == "bottom_left") return ToastPosition::BOTTOM_LEFT;
    if (positionStr == "bottom_center") return ToastPosition::BOTTOM_CENTER;
    if (positionStr == "bottom_right") return ToastPosition::BOTTOM_RIGHT;
    
    LOG_WARNING("Unknown toast position: ", positionStr, ", using bottom_center");
    return ToastPosition::BOTTOM_CENTER;
}

void ToastManager::UpdateToastAnimation(ToastMessage& toast) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - toast.startTime).count() / 1000000.0f;
    
    switch (toast.state) {
        case ToastState::FADE_IN:
            if (elapsed < toast.fadeInTime) {
                toast.currentAlpha = elapsed / toast.fadeInTime;
            } else {
                toast.currentAlpha = 1.0f;
                toast.state = ToastState::VISIBLE;
            }
            break;
            
        case ToastState::VISIBLE:
            if (elapsed >= toast.fadeInTime + toast.duration) {
                toast.state = ToastState::FADE_OUT;
            }
            // Keep alpha at 1.0 during visible phase
            break;
            
        case ToastState::FADE_OUT:
            {
                float fadeStartTime = toast.fadeInTime + toast.duration;
                float fadeElapsed = elapsed - fadeStartTime;
                
                if (fadeElapsed < toast.fadeOutTime) {
                    toast.currentAlpha = 1.0f - (fadeElapsed / toast.fadeOutTime);
                } else {
                    toast.currentAlpha = 0.0f;
                    toast.state = ToastState::DONE;
                }
            }
            break;
            
        case ToastState::DONE:
            toast.currentAlpha = 0.0f;
            break;
    }
    
    // Ensure alpha stays in valid range
    toast.currentAlpha = std::max(0.0f, std::min(1.0f, toast.currentAlpha));
}

void ToastManager::RemoveCompletedToasts() {
    m_messages.erase(
        std::remove_if(m_messages.begin(), m_messages.end(),
            [](const ToastMessage& toast) { return toast.state == ToastState::DONE; }),
        m_messages.end());
}