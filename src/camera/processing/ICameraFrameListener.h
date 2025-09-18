#pragma once

#include "../CameraFrame.h"
#include "camera/sources/ICameraSource.h"
#include <string>
#include <memory>

// Forward declarations
enum class OverflowPolicy;

/**
 * Camera frame listener priority levels for processing order
 */
enum class ListenerPriority {
    LOW = 0,        // Background processing, non-critical
    NORMAL = 10,    // Standard processing priority
    HIGH = 20,      // Time-sensitive processing
    CRITICAL = 30   // Real-time processing (e.g., switching triggers)
};

/**
 * Frame processing result to provide feedback to the publisher
 */
enum class FrameProcessingResult {
    SUCCESS,            // Frame processed successfully
    SKIPPED,           // Frame intentionally skipped (e.g., not enough change)
    FAILED,            // Processing failed but not critical
    CRITICAL_ERROR     // Critical error, listener should be disabled
};

/**
 * Frame processing statistics for monitoring listener performance
 */
struct FrameProcessingStats {
    uint64_t framesReceived = 0;        // Total frames received
    uint64_t framesProcessed = 0;       // Frames successfully processed
    uint64_t framesSkipped = 0;         // Frames intentionally skipped
    uint64_t framesFailed = 0;          // Failed frame processing attempts
    double averageProcessingTimeMs = 0.0;  // Average processing time in ms
    double maxProcessingTimeMs = 0.0;   // Maximum processing time seen
    std::chrono::steady_clock::time_point lastFrameTime; // Last frame received
    
    void Reset() {
        framesReceived = 0;
        framesProcessed = 0;
        framesSkipped = 0;
        framesFailed = 0;
        averageProcessingTimeMs = 0.0;
        maxProcessingTimeMs = 0.0;
        lastFrameTime = std::chrono::steady_clock::now();
    }
    
    double GetSuccessRate() const {
        return framesReceived > 0 ? (double)framesProcessed / framesReceived : 0.0;
    }
};

/**
 * Abstract interface for camera frame listeners.
 * Listeners implement computer vision processing and other frame analysis tasks.
 */
class ICameraFrameListener {
public:
    virtual ~ICameraFrameListener() = default;
    
    /**
     * Process a camera frame.
     * This method will be called from a background thread to avoid blocking camera capture.
     * 
     * @param frame Camera frame to process
     * @return Processing result indicating success/failure/skip status
     */
    virtual FrameProcessingResult ProcessFrame(const CameraFrame& frame) = 0;
    
    /**
     * Get the listener's priority for processing order.
     * Higher priority listeners receive frames first.
     * 
     * @return Priority level for this listener
     */
    virtual ListenerPriority GetPriority() const = 0;
    
    /**
     * Get unique identifier for this listener.
     * Used for logging, debugging, and managing subscriptions.
     * 
     * @return Unique listener identifier
     */
    virtual std::string GetListenerId() const = 0;
    
    /**
     * Get human-readable name for this listener.
     * 
     * @return Descriptive name for logging and UI
     */
    virtual std::string GetListenerName() const = 0;
    
    /**
     * Check if this listener can process frames of the given format.
     * Allows the publisher to skip sending incompatible frames.
     * 
     * @param format Camera frame format to check
     * @return true if this listener can process the format
     */
    virtual bool CanProcessFormat(CameraFormat format) const = 0;
    
    /**
     * Check if this listener requires depth data.
     * Used by the publisher to determine if depth capture is needed.
     * 
     * @return true if depth data is required
     */
    virtual bool RequiresDepthData() const { return false; }
    
    /**
     * Get current processing statistics.
     * 
     * @return Statistics structure with current performance data
     */
    virtual FrameProcessingStats GetStats() const = 0;
    
    /**
     * Reset processing statistics.
     */
    virtual void ResetStats() = 0;
    
    /**
     * Called when the listener is registered with a publisher.
     * Can be used for initialization that requires publisher context.
     */
    virtual void OnRegistered() {}
    
    /**
     * Called when the listener is unregistered from a publisher.
     * Should clean up any resources or ongoing processing.
     */
    virtual void OnUnregistered() {}
    
    /**
     * Called when camera configuration changes.
     * Allows listeners to adapt to new frame sizes, formats, etc.
     * 
     * @param config New camera configuration
     */
    virtual void OnCameraConfigChanged(const CameraConfig& config) {}
    
    /**
     * Enable or disable this listener.
     * Disabled listeners will not receive frames but remain registered.
     * 
     * @param enabled true to enable, false to disable
     */
    virtual void SetEnabled(bool enabled) { m_enabled = enabled; }
    
    /**
     * Check if this listener is currently enabled.
     * 
     * @return true if listener is enabled
     */
    virtual bool IsEnabled() const { return m_enabled; }
    
    /**
     * Get the maximum processing time this listener should be allowed.
     * If processing takes longer, the frame may be skipped or processing interrupted.
     * 
     * @return Maximum processing time in milliseconds (0 = no limit)
     */
    virtual double GetMaxProcessingTimeMs() const { return 0.0; }
    
    /**
     * Check if this listener supports frame skipping under load.
     * If true, the publisher may skip frames when the listener is falling behind.
     *
     * @return true if frame skipping is acceptable
     */
    virtual bool SupportsFrameSkipping() const { return true; }

    /**
     * Check if this listener has custom queue configuration preferences.
     * If true, the publisher will call GetCustomQueueConfig() for settings.
     *
     * @return true if listener has custom preferences
     */
    virtual bool HasCustomQueueConfig() const { return false; }

    /**
     * Get preferred queue size for this listener.
     * Used if GetPreferredQueueConfig() returns nullptr.
     *
     * @return Preferred queue size (0 = use default)
     */
    virtual size_t GetPreferredQueueSize() const { return 0; }

    /**
     * Get preferred overflow policy for this listener.
     * Used if GetPreferredQueueConfig() returns nullptr.
     *
     * @return Preferred overflow policy
     */
    virtual OverflowPolicy GetPreferredOverflowPolicy() const;

    /**
     * Get preferred maximum frame age for this listener.
     * Frames older than this will be dropped before processing.
     *
     * @return Maximum frame age in milliseconds (0 = no limit)
     */
    virtual double GetPreferredMaxFrameAgeMs() const { return 100.0; }

protected:
    bool m_enabled = true;
    mutable FrameProcessingStats m_stats;
    
    /**
     * Update processing statistics.
     * Call this from ProcessFrame implementations to maintain accurate stats.
     */
    void UpdateStats(FrameProcessingResult result, double processingTimeMs) {
        m_stats.framesReceived++;
        m_stats.lastFrameTime = std::chrono::steady_clock::now();
        
        switch (result) {
            case FrameProcessingResult::SUCCESS:
                m_stats.framesProcessed++;
                break;
            case FrameProcessingResult::SKIPPED:
                m_stats.framesSkipped++;
                break;
            case FrameProcessingResult::FAILED:
            case FrameProcessingResult::CRITICAL_ERROR:
                m_stats.framesFailed++;
                break;
        }
        
        // Update processing time statistics
        if (processingTimeMs > 0) {
            m_stats.maxProcessingTimeMs = std::max(m_stats.maxProcessingTimeMs, processingTimeMs);
            
            // Simple moving average for processing time
            if (m_stats.framesProcessed > 1) {
                m_stats.averageProcessingTimeMs = (m_stats.averageProcessingTimeMs * 0.9) + 
                                                 (processingTimeMs * 0.1);
            } else {
                m_stats.averageProcessingTimeMs = processingTimeMs;
            }
        }
    }
};

/**
 * Shared pointer type for camera frame listeners to enable safe multi-threaded access
 */
using CameraFrameListenerPtr = std::shared_ptr<ICameraFrameListener>;

// Forward declare OverflowPolicy enum - defined in CircularBuffer.h
// Default implementation for GetPreferredOverflowPolicy - defined in cpp file if needed