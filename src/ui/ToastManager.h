#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <mutex>

enum class ToastType {
    INFO,
    DEBUG,
    WARNING
};

enum class ToastState {
    FADE_IN,
    VISIBLE,
    FADE_OUT,
    DONE
};

enum class ToastPosition {
    TOP_LEFT,
    TOP_CENTER,
    TOP_RIGHT,
    CENTER_LEFT,
    CENTER,
    CENTER_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_CENTER,
    BOTTOM_RIGHT
};

struct ToastMessage {
    std::string text;
    ToastType type;
    float duration;           // Display time in seconds
    float fadeInTime;         // Fade in animation duration in seconds
    float fadeOutTime;        // Fade out animation duration in seconds
    float currentAlpha;       // Current transparency (0-1)
    std::chrono::steady_clock::time_point startTime;
    ToastState state;
    
    ToastMessage() 
        : type(ToastType::INFO)
        , duration(2.0f)
        , fadeInTime(0.2f)
        , fadeOutTime(0.3f)
        , currentAlpha(0.0f)
        , state(ToastState::FADE_IN) {}
};

struct ToastConfig {
    bool enabled;
    float duration;           // in seconds
    float fadeInTime;         // in seconds
    float fadeOutTime;        // in seconds
    ToastPosition position;
    int offsetX;
    int offsetY;
    int maxWidth;
    int fontSize;
    int cornerRadius;
    int padding;
    
    // Colors (RGBA 0-255)
    struct Color {
        uint8_t r, g, b, a;
        Color(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 255)
            : r(r), g(g), b(b), a(a) {}
    };
    
    Color backgroundColor;
    Color textColor;
    
    // Message filtering
    bool showTriggerEvents;
    bool showStrategyEvents;
    bool showKeyframeEvents;
    
    ToastConfig()
        : enabled(false)
        , duration(2.0f)
        , fadeInTime(0.2f)
        , fadeOutTime(0.3f)
        , position(ToastPosition::BOTTOM_CENTER)
        , offsetX(0)
        , offsetY(100)
        , maxWidth(400)
        , fontSize(14)
        , cornerRadius(8)
        , padding(12)
        , backgroundColor(0, 0, 0, 180)
        , textColor(255, 255, 255, 255)
        , showTriggerEvents(true)
        , showStrategyEvents(true)
        , showKeyframeEvents(true) {}
};

// Forward declaration
class IToastRenderer;

/**
 * Singleton class for managing toast notifications.
 * Handles message queue, timing, animations, and integration with renderers.
 */
class ToastManager {
public:
    /**
     * Get the singleton instance
     */
    static ToastManager& GetInstance();
    
    /**
     * Initialize the toast system with configuration
     */
    void Initialize();
    
    /**
     * Set the toast renderer for the current graphics backend
     */
    void SetRenderer(std::unique_ptr<IToastRenderer> renderer);
    
    /**
     * Add a new toast message to the queue
     */
    void ShowToast(const std::string& message, ToastType type = ToastType::INFO);
    
    /**
     * Show trigger-related toast message (if enabled in config)
     */
    void ShowTriggerEvent(const std::string& triggerName, int targetVideoIndex);
    
    /**
     * Show strategy-related toast message (if enabled in config)
     */
    void ShowStrategyEvent(const std::string& strategyName, int videoIndex, const std::string& action = "switched to");
    
    /**
     * Show keyframe sync scheduling message (if enabled in config)
     */
    void ShowKeyframeScheduled(int targetVideoIndex);
    
    /**
     * Show keyframe sync completion message (if enabled in config)
     */
    void ShowKeyframeCompleted(int videoIndex);
    
    /**
     * Update all active toast messages (called each frame)
     */
    void Update();
    
    /**
     * Render all active toast messages
     */
    void Render();
    
    /**
     * Check if toast system is enabled
     */
    bool IsEnabled() const { return m_config.enabled; }
    
    /**
     * Get current configuration
     */
    const ToastConfig& GetConfig() const { return m_config; }
    
    /**
     * Get current active messages (for renderer)
     */
    const std::vector<ToastMessage>& GetActiveMessages() const { return m_messages; }
    
    /**
     * Clean up resources
     */
    void Cleanup();

private:
    ToastManager() = default;
    ~ToastManager() = default;
    
    // Prevent copying
    ToastManager(const ToastManager&) = delete;
    ToastManager& operator=(const ToastManager&) = delete;
    
    /**
     * Load configuration from Config singleton
     */
    void LoadConfig();
    
    /**
     * Parse color string from config (format: "r,g,b,a")
     */
    ToastConfig::Color ParseColor(const std::string& colorStr, const ToastConfig::Color& defaultColor);
    
    /**
     * Parse position string from config
     */
    ToastPosition ParsePosition(const std::string& positionStr);
    
    /**
     * Update animation state for a toast message
     */
    void UpdateToastAnimation(ToastMessage& toast);
    
    /**
     * Remove completed toast messages
     */
    void RemoveCompletedToasts();
    
    ToastConfig m_config;
    std::vector<ToastMessage> m_messages;
    std::unique_ptr<IToastRenderer> m_renderer;
    mutable std::mutex m_mutex;  // For thread safety
};