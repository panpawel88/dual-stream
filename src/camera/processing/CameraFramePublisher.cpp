#include "CameraFramePublisher.h"

CameraFramePublisher::CameraFramePublisher(const PublisherConfig& config)
    : m_config(config) {
    m_stats.Reset();
}

CameraFramePublisher::~CameraFramePublisher() {
    Stop();
}

bool CameraFramePublisher::Start() {
    if (m_running) {
        return true;
    }

    m_running = true;

    // Start all existing listener processors
    std::lock_guard<std::mutex> processorsLock(m_processorsMutex);
    for (auto& [listenerId, processor] : m_processors) {
        if (processor && !processor->IsRunning()) {
            processor->Start();
        }
    }

    UpdateStatsCounts();
    return true;
}

void CameraFramePublisher::Stop() {
    if (!m_running) {
        return;
    }

    m_running = false;
    StopAllProcessors();
}

bool CameraFramePublisher::PublishFrame(const CameraFrame& frame) {
    if (!m_running || !frame.IsValid()) {
        return false;
    }

    auto startTime = std::chrono::steady_clock::now();

    // Get all processors that can handle this frame format
    std::vector<ListenerProcessor*> targetProcessors;
    {
        std::lock_guard<std::mutex> lock(m_processorsMutex);
        for (auto& [listenerId, processor] : m_processors) {
            if (processor && processor->IsRunning() &&
                processor->IsEnabled() &&
                processor->CanProcessFormat(frame.format)) {
                targetProcessors.push_back(processor.get());
            }
        }
    }

    if (targetProcessors.empty()) {
        return true; // No listeners interested in this format - not an error
    }

    // Enqueue frame to all eligible processors
    size_t successfulEnqueues = 0;
    size_t totalEnqueues = 0;

    for (auto* processor : targetProcessors) {
        totalEnqueues++;
        if (processor->EnqueueFrame(frame)) {
            successfulEnqueues++;
        }
    }

    // Update statistics
    auto endTime = std::chrono::steady_clock::now();
    auto publishTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    UpdatePublishStats(publishTime.count() / 1000.0);

    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.framesPublished++;
        m_stats.totalFrameEnqueues += totalEnqueues;
        m_stats.successfulEnqueues += successfulEnqueues;
        m_stats.failedEnqueues += (totalEnqueues - successfulEnqueues);
    }

    return successfulEnqueues > 0;
}

bool CameraFramePublisher::RegisterListener(CameraFrameListenerPtr listener) {
    if (!listener) {
        return false;
    }

    std::lock_guard<std::mutex> listenersLock(m_listenersMutex);

    // Check if listener already registered
    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listener](const CameraFrameListenerPtr& existing) {
            return existing->GetListenerId() == listener->GetListenerId();
        });

    if (it != m_listeners.end()) {
        return false; // Already registered
    }

    // Add to listeners list
    m_listeners.push_back(listener);
    listener->OnRegistered();

    // Start processor for this listener
    bool processorStarted = StartListenerProcessor(listener);

    // Update statistics
    UpdateStatsCounts();

    return processorStarted;
}

bool CameraFramePublisher::UnregisterListener(const std::string& listenerId) {
    std::lock_guard<std::mutex> listenersLock(m_listenersMutex);

    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listenerId](const CameraFrameListenerPtr& listener) {
            return listener->GetListenerId() == listenerId;
        });

    if (it != m_listeners.end()) {
        (*it)->OnUnregistered();
        m_listeners.erase(it);

        // Stop and remove processor
        StopListenerProcessor(listenerId);

        // Update statistics
        UpdateStatsCounts();

        return true;
    }

    return false;
}

bool CameraFramePublisher::UnregisterListener(CameraFrameListenerPtr listener) {
    if (!listener) {
        return false;
    }
    return UnregisterListener(listener->GetListenerId());
}

std::vector<CameraFrameListenerPtr> CameraFramePublisher::GetListeners() const {
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    return m_listeners;
}

CameraFrameListenerPtr CameraFramePublisher::GetListener(const std::string& listenerId) const {
    std::lock_guard<std::mutex> lock(m_listenersMutex);

    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listenerId](const CameraFrameListenerPtr& listener) {
            return listener->GetListenerId() == listenerId;
        });

    return (it != m_listeners.end()) ? *it : nullptr;
}

bool CameraFramePublisher::SetListenerEnabled(const std::string& listenerId, bool enabled) {
    std::lock_guard<std::mutex> listenersLock(m_listenersMutex);

    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listenerId](const CameraFrameListenerPtr& listener) {
            return listener->GetListenerId() == listenerId;
        });

    if (it != m_listeners.end()) {
        (*it)->SetEnabled(enabled);
        UpdateStatsCounts();
        return true;
    }
    return false;
}

std::vector<CameraFrameListenerPtr> CameraFramePublisher::GetEnabledListeners() const {
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    return GetEnabledListenersInternal();
}

std::vector<CameraFrameListenerPtr> CameraFramePublisher::GetEnabledListenersInternal() const {
    // This method assumes m_listenersMutex is already locked by the caller
    std::vector<CameraFrameListenerPtr> enabledListeners;
    for (const auto& listener : m_listeners) {
        if (!listener) {
            continue;
        }

        try {
            if (listener->IsEnabled()) {
                enabledListeners.push_back(listener);
            }
        } catch (const std::exception& e) {
            // Skip listeners that throw exceptions
            continue;
        }
    }

    return enabledListeners;
}

std::vector<CameraFrameListenerPtr> CameraFramePublisher::GetListenersForFormat(CameraFormat format) const {
    auto enabledListeners = GetEnabledListeners();

    std::vector<CameraFrameListenerPtr> compatibleListeners;
    for (const auto& listener : enabledListeners) {
        if (!listener) {
            continue;
        }

        try {
            if (listener->CanProcessFormat(format)) {
                compatibleListeners.push_back(listener);
            }
        } catch (const std::exception& e) {
            // Skip listeners that throw exceptions during format check
            continue;
        }
    }

    return compatibleListeners;
}

PublisherConfig CameraFramePublisher::GetConfig() const {
    return m_config;
}

bool CameraFramePublisher::UpdateConfig(const PublisherConfig& config) {
    m_config = config;

    // Update all processor configurations if needed
    std::lock_guard<std::mutex> lock(m_processorsMutex);
    for (auto& [listenerId, processor] : m_processors) {
        if (processor) {
            // Create new config for this processor
            auto listener = GetListener(listenerId);
            if (listener) {
                auto newProcessorConfig = CreateProcessorConfig(listener);
                processor->UpdateConfig(newProcessorConfig);
            }
        }
    }

    return true;
}

PublisherStats CameraFramePublisher::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void CameraFramePublisher::ResetStats() {
    std::lock_guard<std::mutex> statsLock(m_statsMutex);
    std::lock_guard<std::mutex> listenersLock(m_listenersMutex);

    m_stats.Reset();
    UpdateStatsCounts();

    // Reset all processor stats
    std::lock_guard<std::mutex> processorsLock(m_processorsMutex);
    for (auto& [listenerId, processor] : m_processors) {
        if (processor) {
            processor->ResetStats();
        }
    }
}

std::optional<ListenerProcessorStats> CameraFramePublisher::GetListenerStats(const std::string& listenerId) const {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    auto it = m_processors.find(listenerId);
    if (it != m_processors.end() && it->second) {
        return it->second->GetStats();
    }

    return std::nullopt;
}

std::unordered_map<std::string, ListenerProcessorStats> CameraFramePublisher::GetAllListenerStats() const {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    std::unordered_map<std::string, ListenerProcessorStats> allStats;
    for (const auto& [listenerId, processor] : m_processors) {
        if (processor) {
            allStats[listenerId] = processor->GetStats();
        }
    }

    return allStats;
}

size_t CameraFramePublisher::GetTotalQueueSize() const {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    size_t totalSize = 0;
    for (const auto& [listenerId, processor] : m_processors) {
        if (processor) {
            totalSize += processor->GetQueueSize();
        }
    }

    return totalSize;
}

size_t CameraFramePublisher::GetListenerQueueSize(const std::string& listenerId) const {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    auto it = m_processors.find(listenerId);
    if (it != m_processors.end() && it->second) {
        return it->second->GetQueueSize();
    }

    return 0;
}

void CameraFramePublisher::ClearAllQueues() {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    for (auto& [listenerId, processor] : m_processors) {
        if (processor) {
            processor->ClearQueue();
        }
    }
}

bool CameraFramePublisher::ClearListenerQueue(const std::string& listenerId) {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    auto it = m_processors.find(listenerId);
    if (it != m_processors.end() && it->second) {
        it->second->ClearQueue();
        return true;
    }

    return false;
}

void CameraFramePublisher::NotifyConfigChange(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    for (auto& [listenerId, processor] : m_processors) {
        if (processor) {
            processor->NotifyConfigChange(config);
        }
    }
}

void CameraFramePublisher::UpdatePublishStats(double processingTimeMs) {
    std::lock_guard<std::mutex> lock(m_statsMutex);

    m_stats.maxPublishTimeMs = std::max(m_stats.maxPublishTimeMs, processingTimeMs);

    if (m_stats.framesPublished > 1) {
        m_stats.averagePublishTimeMs = (m_stats.averagePublishTimeMs * 0.9) +
                                      (processingTimeMs * 0.1);
    } else {
        m_stats.averagePublishTimeMs = processingTimeMs;
    }
}

void CameraFramePublisher::LogPerformanceStats() {
    // Performance logging implementation could go here
    // For now, this is a placeholder
}

ListenerProcessorConfig CameraFramePublisher::CreateProcessorConfig(CameraFrameListenerPtr listener) const {
    ListenerProcessorConfig config;

    if (m_config.useListenerPreferences && listener) {
        // Use individual preference methods
        size_t preferredQueueSize = listener->GetPreferredQueueSize();
        if (preferredQueueSize > 0) {
            config.queueSize = preferredQueueSize;
        } else {
            config.queueSize = m_config.maxFrameQueueSize;
        }

        config.overflowPolicy = listener->GetPreferredOverflowPolicy();
        config.maxFrameAgeMs = listener->GetPreferredMaxFrameAgeMs();
    } else {
        // Use global defaults
        config.queueSize = m_config.maxFrameQueueSize;
        config.maxFrameAgeMs = m_config.maxFrameAgeMs;
        config.overflowPolicy = OverflowPolicy::DROP_OLDEST;
    }

    // Copy global settings
    config.enableStatistics = true;
    config.enableFrameAgeCheck = true;

    return config;
}

bool CameraFramePublisher::StartListenerProcessor(CameraFrameListenerPtr listener) {
    if (!listener) {
        return false;
    }

    std::string listenerId = listener->GetListenerId();

    std::lock_guard<std::mutex> lock(m_processorsMutex);

    // Check if processor already exists
    auto it = m_processors.find(listenerId);
    if (it != m_processors.end()) {
        return it->second->IsRunning();
    }

    // Create processor configuration
    auto config = CreateProcessorConfig(listener);

    // Create and start processor
    auto processor = std::make_unique<ListenerProcessor>(listener, config);
    bool started = false;

    if (m_running) {
        started = processor->Start();
    }

    m_processors[listenerId] = std::move(processor);

    return started || !m_running; // Success if started or publisher not running yet
}

void CameraFramePublisher::StopListenerProcessor(const std::string& listenerId) {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    auto it = m_processors.find(listenerId);
    if (it != m_processors.end()) {
        if (it->second) {
            it->second->Stop();
        }
        m_processors.erase(it);
    }
}

void CameraFramePublisher::StopAllProcessors() {
    std::lock_guard<std::mutex> lock(m_processorsMutex);

    for (auto& [listenerId, processor] : m_processors) {
        if (processor) {
            processor->Stop();
        }
    }
    // Don't clear processors here - they may be restarted
}

void CameraFramePublisher::UpdateStatsCounts() {
    // This should be called with m_listenersMutex already locked
    std::lock_guard<std::mutex> statsLock(m_statsMutex);
    std::lock_guard<std::mutex> processorsLock(m_processorsMutex);

    m_stats.activeListeners = static_cast<int>(m_listeners.size());
    m_stats.enabledListeners = static_cast<int>(GetEnabledListenersInternal().size());

    // Count running processors
    int runningProcessors = 0;
    for (const auto& [listenerId, processor] : m_processors) {
        if (processor && processor->IsRunning()) {
            runningProcessors++;
        }
    }
    m_stats.runningProcessors = runningProcessors;
}