#include "ListenerProcessor.h"
#include <sstream>

ListenerProcessor::ListenerProcessor(CameraFrameListenerPtr listener,
                                   const ListenerProcessorConfig& config)
    : m_listener(std::move(listener))
    , m_config(config) {

    m_stats.Reset();

    // Create circular buffer with configured size and policy
    m_frameQueue = std::make_unique<CircularBuffer<TimestampedFrame>>(
        m_config.queueSize, m_config.overflowPolicy);
}

ListenerProcessor::~ListenerProcessor() {
    Stop();
}

bool ListenerProcessor::Start() {
    if (m_running || !m_listener) {
        return false;
    }

    m_shouldStop = false;
    m_running = true;

    // Generate thread name if not provided
    std::string threadName = m_config.threadName;
    if (threadName.empty()) {
        threadName = GenerateThreadName();
    }

    // Create and start processor thread
    m_processorThread = std::make_unique<std::thread>(&ListenerProcessor::ProcessorThreadFunc, this);

    // Set thread name (Windows specific - could be extended for other platforms)
    #ifdef _WIN32
    // Note: SetThreadDescription is available in Windows 10, version 1607 and later
    // For older versions, this will fail silently
    std::wstring wThreadName(threadName.begin(), threadName.end());
    SetThreadDescription(m_processorThread->native_handle(), wThreadName.c_str());
    #endif

    m_stats.startTime = std::chrono::steady_clock::now();

    return true;
}

void ListenerProcessor::Stop() {
    if (!m_running) {
        return;
    }

    m_shouldStop = true;
    m_running = false;

    // Shutdown the queue to wake up the processor thread
    if (m_frameQueue) {
        m_frameQueue->Shutdown();
    }

    // Wait for processor thread to finish
    if (m_processorThread && m_processorThread->joinable()) {
        m_processorThread->join();
    }
    m_processorThread.reset();

    // Clear the queue
    if (m_frameQueue) {
        m_frameQueue->Clear();
    }
}

bool ListenerProcessor::EnqueueFrame(std::shared_ptr<const CameraFrame> frame) {
    if (!m_running || !m_listener || !m_frameQueue || !frame) {
        return false;
    }

    // Check if listener is enabled
    if (!m_listener->IsEnabled()) {
        return true; // Not an error, just skip
    }

    // Check if listener can process this format
    if (!m_listener->CanProcessFormat(frame->format)) {
        return true; // Not an error, just skip
    }

    // Create timestamped frame
    TimestampedFrame timestampedFrame(frame);

    // Try to enqueue the frame
    bool success = m_frameQueue->TryPush(std::move(timestampedFrame));

    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        if (success) {
            m_stats.framesEnqueued++;
            UpdateQueueStats(m_frameQueue->Size());
        } else {
            m_stats.framesDroppedQueue++;
        }
    }

    return success;
}

void ListenerProcessor::ProcessorThreadFunc() {
    while (!m_shouldStop) {
        // Wait for frame with timeout to allow periodic checks
        auto timestampedFrame = m_frameQueue->Pop(100); // 100ms timeout

        if (!timestampedFrame) {
            if (m_frameQueue->IsShutdown()) {
                break;
            }
            continue; // Timeout, check again
        }

        // Check if we should process this frame
        if (ShouldDropFrame(*timestampedFrame)) {
            std::lock_guard<std::mutex> lock(m_statsMutex);
            m_stats.framesDroppedAge++;
            continue;
        }

        // Process the frame
        if (!ProcessFrame(*timestampedFrame)) {
            // Processing failed or was skipped
            continue;
        }
    }
}

bool ListenerProcessor::ProcessFrame(const TimestampedFrame& timestampedFrame) {
    if (!m_listener || !m_listener->IsEnabled()) {
        return false;
    }

    auto startTime = std::chrono::steady_clock::now();
    FrameProcessingResult result = FrameProcessingResult::FAILED;

    try {
        // Process the frame
        result = m_listener->ProcessFrame(timestampedFrame.frame);

        // Handle critical errors by disabling the listener
        if (result == FrameProcessingResult::CRITICAL_ERROR) {
            try {
                m_listener->SetEnabled(false);
            } catch (...) {
                // Ignore errors when trying to disable listener
            }
        }

    } catch (const std::exception& e) {
        // Handle listener exceptions gracefully
        result = FrameProcessingResult::CRITICAL_ERROR;
        try {
            m_listener->SetEnabled(false);
        } catch (...) {
            // Ignore errors when trying to disable listener
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    double processingTimeMs = processingTime.count() / 1000.0;

    // Update statistics
    size_t currentQueueDepth = m_frameQueue ? m_frameQueue->Size() : 0;
    UpdateStats(result, processingTimeMs, currentQueueDepth);

    return result == FrameProcessingResult::SUCCESS || result == FrameProcessingResult::SKIPPED;
}

bool ListenerProcessor::ShouldDropFrame(const TimestampedFrame& timestampedFrame) const {
    if (!m_config.enableFrameAgeCheck) {
        return false;
    }

    return timestampedFrame.GetAgeMs() > m_config.maxFrameAgeMs;
}

void ListenerProcessor::UpdateStats(FrameProcessingResult result, double processingTimeMs, size_t queueDepth) {
    std::lock_guard<std::mutex> lock(m_statsMutex);

    switch (result) {
        case FrameProcessingResult::SUCCESS:
            m_stats.framesProcessed++;
            break;
        case FrameProcessingResult::SKIPPED:
            m_stats.framesSkipped++;
            break;
        case FrameProcessingResult::FAILED:
            m_stats.framesFailed++;
            break;
        case FrameProcessingResult::CRITICAL_ERROR:
            m_stats.framesFailed++;
            break;
    }

    // Update processing time statistics
    if (processingTimeMs > 0) {
        m_stats.maxProcessingTimeMs = std::max(m_stats.maxProcessingTimeMs, processingTimeMs);

        if (m_stats.framesProcessed > 1) {
            m_stats.averageProcessingTimeMs = (m_stats.averageProcessingTimeMs * 0.9) +
                                            (processingTimeMs * 0.1);
        } else {
            m_stats.averageProcessingTimeMs = processingTimeMs;
        }
    }

    // Update queue depth statistics
    UpdateQueueStats(queueDepth);

    m_stats.lastFrameTime = std::chrono::steady_clock::now();
}

void ListenerProcessor::UpdateQueueStats(size_t currentDepth) {
    // This should be called with m_statsMutex already locked
    m_stats.maxQueueDepth = std::max(m_stats.maxQueueDepth, (double)currentDepth);

    // Simple moving average for queue depth
    if (m_stats.framesEnqueued > 1) {
        m_stats.averageQueueDepth = (m_stats.averageQueueDepth * 0.9) + (currentDepth * 0.1);
    } else {
        m_stats.averageQueueDepth = currentDepth;
    }
}

std::string ListenerProcessor::GenerateThreadName() const {
    std::stringstream ss;
    ss << "Listener_";
    if (m_listener) {
        std::string listenerId = m_listener->GetListenerId();
        // Sanitize the listener ID for use as thread name
        for (char& c : listenerId) {
            if (!std::isalnum(c) && c != '_') {
                c = '_';
            }
        }
        ss << listenerId;
    } else {
        ss << "Unknown";
    }
    return ss.str();
}

ListenerProcessorStats ListenerProcessor::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void ListenerProcessor::ResetStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    m_stats.Reset();
}

ListenerProcessorConfig ListenerProcessor::GetConfig() const {
    return m_config;
}

bool ListenerProcessor::UpdateConfig(const ListenerProcessorConfig& config) {
    bool needsRestart = (config.queueSize != m_config.queueSize ||
                        config.overflowPolicy != m_config.overflowPolicy);

    if (needsRestart && m_running) {
        Stop();
        m_config = config;

        // Recreate queue with new settings
        m_frameQueue = std::make_unique<CircularBuffer<TimestampedFrame>>(
            m_config.queueSize, m_config.overflowPolicy);

        return Start();
    } else {
        m_config = config;
        return true;
    }
}

void ListenerProcessor::ClearQueue() {
    if (m_frameQueue) {
        m_frameQueue->Clear();
    }
}

void ListenerProcessor::NotifyConfigChange(const CameraConfig& config) {
    if (m_listener) {
        try {
            m_listener->OnCameraConfigChanged(config);
        } catch (...) {
            // Ignore errors in listener notification
        }
    }
}