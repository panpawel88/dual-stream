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
    
    m_shouldStop = false;
    m_running = true;
    
    // Create worker threads
    for (size_t i = 0; i < m_config.maxWorkerThreads; ++i) {
        m_workerThreads.push_back(
            std::make_unique<std::thread>(&CameraFramePublisher::WorkerThreadFunc, this, static_cast<int>(i))
        );
    }
    
    return true;
}

void CameraFramePublisher::Stop() {
    if (!m_running) {
        return;
    }
    
    m_shouldStop = true;
    m_running = false;
    
    // Wake up all worker threads
    m_queueCondition.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : m_workerThreads) {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
    m_workerThreads.clear();
    
    // Clear remaining frames
    ClearQueue();
}

bool CameraFramePublisher::PublishFrame(const CameraFrame& frame) {
    if (!m_running || !frame.IsValid()) {
        return false;
    }
    
    auto startTime = std::chrono::steady_clock::now();
    
    // Get listeners that can process this frame format
    auto targetListeners = GetListenersForFormat(frame.format);
    if (targetListeners.empty()) {
        return true; // No listeners interested in this format
    }
    
    // Check queue size and drop old frames if necessary
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        
        // Remove expired frames
        RemoveExpiredFrames();
        
        // Check if queue is full
        if (m_frameQueue.size() >= m_config.maxFrameQueueSize) {
            if (m_config.enableFrameSkipping) {
                // Drop oldest frame to make room
                m_frameQueue.pop();
                {
                    std::lock_guard<std::mutex> statsLock(m_statsMutex);
                    m_stats.framesDroppedQueue++;
                }
            } else {
                return false; // Queue full and skipping disabled
            }
        }
        
        // Add new delivery task to queue
        m_frameQueue.emplace(frame, targetListeners);
    }
    
    // Notify worker threads
    m_queueCondition.notify_one();
    
    // Update statistics
    auto endTime = std::chrono::steady_clock::now();
    auto publishTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    UpdatePublishStats(publishTime.count() / 1000.0);
    
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.framesPublished++;
    }
    
    return true;
}

void CameraFramePublisher::WorkerThreadFunc(int threadId) {
    while (!m_shouldStop) {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        
        // Wait for frames or stop signal
        m_queueCondition.wait(lock, [this]() {
            return m_shouldStop || !m_frameQueue.empty();
        });
        
        if (m_shouldStop) {
            break;
        }
        
        // Get next delivery task
        if (m_frameQueue.empty()) {
            continue;
        }
        
        FrameDeliveryTask task = std::move(m_frameQueue.front());
        m_frameQueue.pop();
        lock.unlock();
        
        // Check if frame is too old
        if (ShouldDropFrame(task)) {
            std::lock_guard<std::mutex> statsLock(m_statsMutex);
            m_stats.framesDroppedAge++;
            continue;
        }
        
        // Process the delivery task
        ProcessDeliveryTask(task, threadId);
    }
}

void CameraFramePublisher::ProcessDeliveryTask(const FrameDeliveryTask& task, int threadId) {
    auto startTime = std::chrono::steady_clock::now();
    
    // Sort listeners by priority if enabled
    auto listeners = task.listeners;
    if (m_config.enablePriorityProcessing) {
        SortListenersByPriority(listeners);
    }
    
    // Deliver frame to each listener
    for (auto& listener : listeners) {
        if (!listener || !listener->IsEnabled()) {
            continue;
        }
        
        auto listenerStartTime = std::chrono::steady_clock::now();
        
        try {
            FrameProcessingResult result = listener->ProcessFrame(task.frame);
            
            // Handle critical errors
            if (result == FrameProcessingResult::CRITICAL_ERROR) {
                // Disable listener on critical error
                listener->SetEnabled(false);
            }
            
        } catch (const std::exception& e) {
            // Handle listener exceptions gracefully
            listener->SetEnabled(false);
        }
        
        auto listenerEndTime = std::chrono::steady_clock::now();
        auto processingTime = std::chrono::duration_cast<std::chrono::microseconds>(
            listenerEndTime - listenerStartTime);
        
        {
            std::lock_guard<std::mutex> lock(m_statsMutex);
            m_stats.totalListenerCalls++;
        }
    }
    
    auto endTime = std::chrono::steady_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    
    // Log performance if enabled
    if (m_config.enablePerformanceLogging && totalTime.count() > 10000) { // > 10ms
        // Could log slow processing here
    }
}

bool CameraFramePublisher::RegisterListener(CameraFrameListenerPtr listener) {
    if (!listener) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    
    // Check if listener already registered
    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listener](const CameraFrameListenerPtr& existing) {
            return existing->GetListenerId() == listener->GetListenerId();
        });
    
    if (it != m_listeners.end()) {
        return false; // Already registered
    }
    
    m_listeners.push_back(listener);
    listener->OnRegistered();
    
    {
        std::lock_guard<std::mutex> statsLock(m_statsMutex);
        m_stats.activeListeners = static_cast<int>(m_listeners.size());
        m_stats.enabledListeners = static_cast<int>(GetEnabledListenersInternal().size());
    }
    
    return true;
}

bool CameraFramePublisher::UnregisterListener(const std::string& listenerId) {
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    
    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listenerId](const CameraFrameListenerPtr& listener) {
            return listener->GetListenerId() == listenerId;
        });
    
    if (it != m_listeners.end()) {
        (*it)->OnUnregistered();
        m_listeners.erase(it);
        
        {
            std::lock_guard<std::mutex> statsLock(m_statsMutex);
            m_stats.activeListeners = static_cast<int>(m_listeners.size());
            m_stats.enabledListeners = static_cast<int>(GetEnabledListenersInternal().size());
        }
        
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
    std::lock_guard<std::mutex> lock(m_listenersMutex);
    
    auto it = std::find_if(m_listeners.begin(), m_listeners.end(),
        [&listenerId](const CameraFrameListenerPtr& listener) {
            return listener->GetListenerId() == listenerId;
        });
    
    if (it != m_listeners.end()) {
        (*it)->SetEnabled(enabled);
        
        {
            std::lock_guard<std::mutex> statsLock(m_statsMutex);
            m_stats.enabledListeners = static_cast<int>(GetEnabledListenersInternal().size());
        }
        
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
        if (listener && listener->IsEnabled()) {
            enabledListeners.push_back(listener);
        }
    }
    
    return enabledListeners;
}

std::vector<CameraFrameListenerPtr> CameraFramePublisher::GetListenersForFormat(CameraFormat format) const {
    auto enabledListeners = GetEnabledListeners();
    
    std::vector<CameraFrameListenerPtr> compatibleListeners;
    for (const auto& listener : enabledListeners) {
        if (listener->CanProcessFormat(format)) {
            compatibleListeners.push_back(listener);
        }
    }
    
    return compatibleListeners;
}

bool CameraFramePublisher::ShouldDropFrame(const FrameDeliveryTask& task) const {
    return task.GetAgeMs() > m_config.maxFrameAgeMs;
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

void CameraFramePublisher::SortListenersByPriority(std::vector<CameraFrameListenerPtr>& listeners) const {
    std::sort(listeners.begin(), listeners.end(),
        [](const CameraFrameListenerPtr& a, const CameraFrameListenerPtr& b) {
            return static_cast<int>(a->GetPriority()) > static_cast<int>(b->GetPriority());
        });
}

void CameraFramePublisher::RemoveExpiredFrames() {
    // This should be called with m_queueMutex already locked
    std::queue<FrameDeliveryTask> validFrames;
    
    while (!m_frameQueue.empty()) {
        auto& task = m_frameQueue.front();
        if (!ShouldDropFrame(task)) {
            validFrames.push(std::move(task));
        } else {
            m_stats.framesDroppedAge++;
        }
        m_frameQueue.pop();
    }
    
    m_frameQueue = std::move(validFrames);
}

PublisherConfig CameraFramePublisher::GetConfig() const {
    return m_config;
}

bool CameraFramePublisher::UpdateConfig(const PublisherConfig& config) {
    bool needsRestart = (config.maxWorkerThreads != m_config.maxWorkerThreads);
    
    if (needsRestart && m_running) {
        Stop();
        m_config = config;
        return Start();
    } else {
        m_config = config;
        return true;
    }
}

PublisherStats CameraFramePublisher::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

void CameraFramePublisher::ResetStats() {
    std::lock_guard<std::mutex> statsLock(m_statsMutex);
    std::lock_guard<std::mutex> listenersLock(m_listenersMutex);
    
    m_stats.Reset();
    m_stats.activeListeners = static_cast<int>(m_listeners.size());
    m_stats.enabledListeners = static_cast<int>(GetEnabledListenersInternal().size());
}

size_t CameraFramePublisher::GetQueueSize() const {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    return m_frameQueue.size();
}

void CameraFramePublisher::ClearQueue() {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    std::queue<FrameDeliveryTask> empty;
    m_frameQueue.swap(empty);
}

void CameraFramePublisher::NotifyConfigChange(const CameraConfig& config) {
    auto listeners = GetListeners();
    for (auto& listener : listeners) {
        if (listener) {
            listener->OnCameraConfigChanged(config);
        }
    }
}