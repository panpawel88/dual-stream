#pragma once

#include "ICameraFrameListener.h"
#include "CircularBuffer.h"
#include "../CameraFrame.h"
#include <thread>
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>

/**
 * Configuration for individual listener processing
 */
struct ListenerProcessorConfig {
    size_t queueSize = 3;                              // Circular buffer size for frames
    OverflowPolicy overflowPolicy = OverflowPolicy::DROP_OLDEST; // What to do when queue is full
    double maxFrameAgeMs = 100.0;                      // Maximum age before dropping frames
    bool enableFrameAgeCheck = true;                   // Enable frame age checking
    bool enableStatistics = true;                      // Enable detailed statistics
    std::string threadName = "";                       // Thread name for debugging (auto-generated if empty)
};

/**
 * Statistics for individual listener processor
 */
struct ListenerProcessorStats {
    uint64_t framesEnqueued = 0;       // Total frames added to queue
    uint64_t framesProcessed = 0;      // Frames successfully processed
    uint64_t framesDroppedQueue = 0;   // Frames dropped due to full queue
    uint64_t framesDroppedAge = 0;     // Frames dropped due to age limit
    uint64_t framesSkipped = 0;        // Frames skipped by listener
    uint64_t framesFailed = 0;         // Failed processing attempts

    double averageProcessingTimeMs = 0.0;  // Average processing time
    double maxProcessingTimeMs = 0.0;      // Maximum processing time seen
    double averageQueueDepth = 0.0;        // Average queue depth
    double maxQueueDepth = 0.0;            // Maximum queue depth seen

    std::chrono::steady_clock::time_point lastFrameTime; // Last frame processed
    std::chrono::steady_clock::time_point startTime;     // When processor started

    void Reset() {
        framesEnqueued = 0;
        framesProcessed = 0;
        framesDroppedQueue = 0;
        framesDroppedAge = 0;
        framesSkipped = 0;
        framesFailed = 0;
        averageProcessingTimeMs = 0.0;
        maxProcessingTimeMs = 0.0;
        averageQueueDepth = 0.0;
        maxQueueDepth = 0.0;
        lastFrameTime = std::chrono::steady_clock::now();
        startTime = std::chrono::steady_clock::now();
    }

    double GetProcessingRate() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
        return elapsed.count() > 0 ? (double)framesProcessed / elapsed.count() : 0.0;
    }

    double GetSuccessRate() const {
        uint64_t totalAttempts = framesProcessed + framesSkipped + framesFailed;
        return totalAttempts > 0 ? (double)framesProcessed / totalAttempts : 0.0;
    }
};

/**
 * Frame with timestamp for age checking
 */
struct TimestampedFrame {
    CameraFrame frame;
    std::chrono::steady_clock::time_point timestamp;

    // Default constructor needed for CircularBuffer template
    TimestampedFrame()
        : timestamp(std::chrono::steady_clock::now()) {}

    TimestampedFrame(const CameraFrame& f)
        : frame(f), timestamp(std::chrono::steady_clock::now()) {}

    double GetAgeMs() const {
        auto now = std::chrono::steady_clock::now();
        auto age = std::chrono::duration_cast<std::chrono::microseconds>(now - timestamp);
        return age.count() / 1000.0;
    }
};

/**
 * Individual processor for a camera frame listener.
 * Each listener gets its own thread and circular buffer queue.
 */
class ListenerProcessor {
public:
    explicit ListenerProcessor(CameraFrameListenerPtr listener,
                              const ListenerProcessorConfig& config = ListenerProcessorConfig{});
    ~ListenerProcessor();

    /**
     * Start the processor thread.
     * @return true if started successfully
     */
    bool Start();

    /**
     * Stop the processor thread and wait for completion.
     */
    void Stop();

    /**
     * Check if processor is currently running.
     */
    bool IsRunning() const { return m_running; }

    /**
     * Enqueue a frame for processing.
     * This method is non-blocking and thread-safe.
     *
     * @param frame Frame to process
     * @return true if frame was enqueued successfully
     */
    bool EnqueueFrame(const CameraFrame& frame);

    /**
     * Get the associated listener.
     */
    CameraFrameListenerPtr GetListener() const { return m_listener; }

    /**
     * Get listener ID for identification.
     */
    std::string GetListenerId() const {
        return m_listener ? m_listener->GetListenerId() : "";
    }

    /**
     * Check if listener can process the given format.
     */
    bool CanProcessFormat(CameraFormat format) const {
        return m_listener ? m_listener->CanProcessFormat(format) : false;
    }

    /**
     * Check if listener is enabled.
     */
    bool IsEnabled() const {
        return m_listener ? m_listener->IsEnabled() : false;
    }

    /**
     * Get current processor statistics.
     */
    ListenerProcessorStats GetStats() const;

    /**
     * Reset processor statistics.
     */
    void ResetStats();

    /**
     * Get current configuration.
     */
    ListenerProcessorConfig GetConfig() const;

    /**
     * Update configuration.
     * Some changes may require restart to take effect.
     *
     * @param config New configuration
     * @return true if configuration was updated
     */
    bool UpdateConfig(const ListenerProcessorConfig& config);

    /**
     * Get current queue size.
     */
    size_t GetQueueSize() const {
        return m_frameQueue ? m_frameQueue->Size() : 0;
    }

    /**
     * Get queue capacity.
     */
    size_t GetQueueCapacity() const {
        return m_frameQueue ? m_frameQueue->Capacity() : 0;
    }

    /**
     * Clear all frames from the queue.
     */
    void ClearQueue();

    /**
     * Notify listener of camera configuration change.
     */
    void NotifyConfigChange(const CameraConfig& config);

private:
    CameraFrameListenerPtr m_listener;
    ListenerProcessorConfig m_config;

    // Threading
    std::unique_ptr<std::thread> m_processorThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};

    // Frame queue
    std::unique_ptr<CircularBuffer<TimestampedFrame>> m_frameQueue;

    // Statistics
    mutable std::mutex m_statsMutex;
    ListenerProcessorStats m_stats;

    // Private methods
    void ProcessorThreadFunc();
    bool ProcessFrame(const TimestampedFrame& timestampedFrame);
    bool ShouldDropFrame(const TimestampedFrame& timestampedFrame) const;
    void UpdateStats(FrameProcessingResult result, double processingTimeMs, size_t queueDepth);
    void UpdateQueueStats(size_t currentDepth);
    std::string GenerateThreadName() const;
};