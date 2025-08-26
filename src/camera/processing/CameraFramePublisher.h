#pragma once

#include "ICameraFrameListener.h"
#include "../CameraFrame.h"
#include "camera/sources/ICameraSource.h"
#include <vector>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <memory>

/**
 * Publisher configuration for frame delivery behavior
 */
struct PublisherConfig {
    size_t maxFrameQueueSize = 5;          // Maximum frames in delivery queue
    size_t maxWorkerThreads = 2;           // Number of worker threads for frame processing
    double maxFrameAgeMs = 100.0;          // Maximum age for frames before dropping (ms)
    bool enableFrameSkipping = true;       // Allow skipping frames under load
    bool enableLoadBalancing = true;       // Distribute listeners across worker threads
    bool enablePriorityProcessing = true;  // Process higher priority listeners first
    
    // Performance monitoring
    double statsReportIntervalMs = 5000.0; // Statistics reporting interval
    bool enablePerformanceLogging = false; // Log performance statistics
};

/**
 * Publisher statistics for monitoring performance
 */
struct PublisherStats {
    uint64_t framesPublished = 0;          // Total frames published
    uint64_t framesDroppedQueue = 0;       // Frames dropped due to full queue
    uint64_t framesDroppedAge = 0;         // Frames dropped due to age limit
    uint64_t totalListenerCalls = 0;       // Total listener processing calls
    double averagePublishTimeMs = 0.0;     // Average time to publish to all listeners
    double maxPublishTimeMs = 0.0;         // Maximum publish time seen
    int activeListeners = 0;               // Number of active listeners
    int enabledListeners = 0;              // Number of enabled listeners
    
    void Reset() {
        framesPublished = 0;
        framesDroppedQueue = 0;
        framesDroppedAge = 0;
        totalListenerCalls = 0;
        averagePublishTimeMs = 0.0;
        maxPublishTimeMs = 0.0;
    }
};

/**
 * Frame delivery task for worker threads
 */
struct FrameDeliveryTask {
    CameraFrame frame;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<CameraFrameListenerPtr> listeners; // Listeners to deliver to
    
    FrameDeliveryTask(const CameraFrame& f, const std::vector<CameraFrameListenerPtr>& l)
        : frame(f), timestamp(std::chrono::steady_clock::now()), listeners(l) {}
    
    double GetAgeMs() const {
        auto now = std::chrono::steady_clock::now();
        auto age = std::chrono::duration_cast<std::chrono::microseconds>(now - timestamp);
        return age.count() / 1000.0;
    }
};

/**
 * Thread-safe camera frame publisher with multi-subscriber support.
 * Delivers frames to registered listeners using background worker threads
 * to avoid blocking the camera capture thread.
 */
class CameraFramePublisher {
public:
    explicit CameraFramePublisher(const PublisherConfig& config = PublisherConfig{});
    ~CameraFramePublisher();
    
    /**
     * Start the publisher and worker threads.
     * @return true if started successfully
     */
    bool Start();
    
    /**
     * Stop the publisher and worker threads.
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
     * 
     * @param frame Frame to publish
     * @return true if frame was queued for delivery
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
     * Get number of frames currently in delivery queue.
     * 
     * @return Queue size
     */
    size_t GetQueueSize() const;
    
    /**
     * Clear all frames from delivery queue.
     */
    void ClearQueue();
    
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
    
    // Worker thread management
    std::vector<std::unique_ptr<std::thread>> m_workerThreads;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};
    
    // Frame delivery queue
    mutable std::mutex m_queueMutex;
    std::queue<FrameDeliveryTask> m_frameQueue;
    std::condition_variable m_queueCondition;
    
    // Statistics
    mutable std::mutex m_statsMutex;
    PublisherStats m_stats;
    
    // Private methods
    void WorkerThreadFunc(int threadId);
    void ProcessDeliveryTask(const FrameDeliveryTask& task, int threadId);
    std::vector<CameraFrameListenerPtr> GetEnabledListeners() const;
    std::vector<CameraFrameListenerPtr> GetListenersForFormat(CameraFormat format) const;
    bool ShouldDropFrame(const FrameDeliveryTask& task) const;
    void UpdatePublishStats(double processingTimeMs);
    void LogPerformanceStats();
    void SortListenersByPriority(std::vector<CameraFrameListenerPtr>& listeners) const;
    void RemoveExpiredFrames();
};


