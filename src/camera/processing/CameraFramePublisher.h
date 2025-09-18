#pragma once

#include "ICameraFrameListener.h"
#include "ListenerProcessor.h"
#include "../CameraFrame.h"
#include "camera/sources/ICameraSource.h"
#include <vector>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <memory>
#include <optional>

/**
 * Publisher configuration for frame delivery behavior
 */
struct PublisherConfig {
    // Default listener processor configuration
    ListenerProcessorConfig defaultListenerConfig;

    // Global publisher settings
    bool useListenerPreferences = true;    // Use listener-specific queue preferences
};

/**
 * Publisher statistics for monitoring performance
 */
struct PublisherStats {
    uint64_t framesPublished = 0;          // Total frames published
    uint64_t totalFrameEnqueues = 0;       // Total frame enqueue operations
    uint64_t successfulEnqueues = 0;       // Successful frame enqueues
    uint64_t failedEnqueues = 0;           // Failed frame enqueues
    double averagePublishTimeMs = 0.0;     // Average time to publish to all listeners
    double maxPublishTimeMs = 0.0;         // Maximum publish time seen
    int activeListeners = 0;               // Number of active listeners
    int enabledListeners = 0;              // Number of enabled listeners
    int runningProcessors = 0;             // Number of running processors

    void Reset() {
        framesPublished = 0;
        totalFrameEnqueues = 0;
        successfulEnqueues = 0;
        failedEnqueues = 0;
        averagePublishTimeMs = 0.0;
        maxPublishTimeMs = 0.0;
    }
};


/**
 * Thread-safe camera frame publisher with per-listener processing.
 * Each listener gets its own dedicated thread and circular buffer queue.
 * This ensures that slow listeners don't block other listeners.
 */
class CameraFramePublisher {
public:
    explicit CameraFramePublisher(const PublisherConfig& config = PublisherConfig{});
    ~CameraFramePublisher();

    /**
     * Start the publisher and all listener processors.
     * @return true if started successfully
     */
    bool Start();

    /**
     * Stop the publisher and all listener processors.
     */
    void Stop();

    /**
     * Check if publisher is currently running.
     * @return true if publisher is active
     */
    bool IsRunning() const { return m_running; }
    
    /**
     * Publish a frame to all registered listeners.
     * This method is thread-safe and non-blocking.
     * Each listener processes the frame independently on its own thread.
     *
     * @param frame Frame to publish
     * @return true if frame was queued for delivery to at least one listener
     */
    bool PublishFrame(const CameraFrame& frame);
    
    /**
     * Register a listener for frame delivery.
     * 
     * @param listener Listener to register
     * @return true if listener was registered successfully
     */
    bool RegisterListener(CameraFrameListenerPtr listener);
    
    /**
     * Unregister a listener from frame delivery.
     * 
     * @param listenerId ID of listener to unregister
     * @return true if listener was found and removed
     */
    bool UnregisterListener(const std::string& listenerId);
    
    /**
     * Unregister a listener by pointer.
     * 
     * @param listener Listener pointer to unregister
     * @return true if listener was found and removed
     */
    bool UnregisterListener(CameraFrameListenerPtr listener);
    
    /**
     * Get list of all registered listeners.
     * 
     * @return Vector of registered listener pointers
     */
    std::vector<CameraFrameListenerPtr> GetListeners() const;
    
    /**
     * Get listener by ID.
     * 
     * @param listenerId ID of listener to find
     * @return Listener pointer or nullptr if not found
     */
    CameraFrameListenerPtr GetListener(const std::string& listenerId) const;
    
    /**
     * Enable/disable a specific listener.
     * 
     * @param listenerId ID of listener to modify
     * @param enabled true to enable, false to disable
     * @return true if listener was found and modified
     */
    bool SetListenerEnabled(const std::string& listenerId, bool enabled);
    
    /**
     * Get current publisher configuration.
     * 
     * @return Current configuration
     */
    PublisherConfig GetConfig() const;
    
    /**
     * Update publisher configuration.
     * Some changes may require restart to take effect.
     * 
     * @param config New configuration
     * @return true if configuration was updated
     */
    bool UpdateConfig(const PublisherConfig& config);
    
    /**
     * Get current publisher statistics.
     *
     * @return Statistics structure
     */
    PublisherStats GetStats() const;

    /**
     * Reset publisher statistics.
     */
    void ResetStats();

    /**
     * Get statistics for a specific listener processor.
     *
     * @param listenerId ID of listener to get stats for
     * @return Processor statistics or empty optional if not found
     */
    std::optional<ListenerProcessorStats> GetListenerStats(const std::string& listenerId) const;

    /**
     * Get statistics for all listener processors.
     *
     * @return Map of listener ID to processor statistics
     */
    std::unordered_map<std::string, ListenerProcessorStats> GetAllListenerStats() const;

    /**
     * Get total number of frames queued across all listeners.
     *
     * @return Total queue size
     */
    size_t GetTotalQueueSize() const;

    /**
     * Get queue size for a specific listener.
     *
     * @param listenerId ID of listener
     * @return Queue size or 0 if listener not found
     */
    size_t GetListenerQueueSize(const std::string& listenerId) const;

    /**
     * Clear all frames from all listener queues.
     */
    void ClearAllQueues();

    /**
     * Clear frames from a specific listener queue.
     *
     * @param listenerId ID of listener
     * @return true if listener was found and queue cleared
     */
    bool ClearListenerQueue(const std::string& listenerId);
    
    /**
     * Notify listeners of camera configuration change.
     * 
     * @param config New camera configuration
     */
    void NotifyConfigChange(const CameraConfig& config);

private:
    PublisherConfig m_config;
    mutable std::mutex m_listenersMutex;
    std::vector<CameraFrameListenerPtr> m_listeners;

    // Per-listener processors
    mutable std::mutex m_processorsMutex;
    std::unordered_map<std::string, std::unique_ptr<ListenerProcessor>> m_processors;

    std::atomic<bool> m_running{false};

    // Statistics
    mutable std::mutex m_statsMutex;
    PublisherStats m_stats;

    // Private methods
    std::vector<CameraFrameListenerPtr> GetEnabledListeners() const;
    std::vector<CameraFrameListenerPtr> GetEnabledListenersInternal() const; // No mutex lock - for internal use
    std::vector<CameraFrameListenerPtr> GetListenersForFormat(CameraFormat format) const;
    void UpdatePublishStats(double processingTimeMs);
    void LogPerformanceStats();
    ListenerProcessorConfig CreateProcessorConfig(CameraFrameListenerPtr listener) const;
    bool StartListenerProcessor(CameraFrameListenerPtr listener);
    void StopListenerProcessor(const std::string& listenerId);
    void StopAllProcessors();
    void UpdateStatsCounts();
};


