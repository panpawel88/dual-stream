#include "PerformanceStatistics.h"
#include "Logger.h"
#include "imgui.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

PerformanceStatistics& PerformanceStatistics::GetInstance() {
    static PerformanceStatistics instance;
    return instance;
}

void PerformanceStatistics::SetVideoFrameRate(double fps) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_videoFrameRate = fps;
}

void PerformanceStatistics::SetCurrentPlaybackTime(double time) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_currentPlaybackTime = time;
}

void PerformanceStatistics::SetActiveVideoIndex(int index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_activeVideoIndex = index;
}

void PerformanceStatistics::SetPlaybackSpeed(double speed) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_playbackSpeed = speed;
}

void PerformanceStatistics::SetVideoResolution(int width, int height) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_videoWidth = width;
    m_videoHeight = height;
}

void PerformanceStatistics::SetVideoFormat(const std::string& format) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_videoFormat = format;
}

void PerformanceStatistics::RecordDecodeTime(double milliseconds) {
    std::lock_guard<std::mutex> lock(m_mutex);
    AddToHistory(m_decodeTimeHistory, milliseconds);
}

void PerformanceStatistics::SetDecoderType(const std::string& type) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_decoderType = type;
}

void PerformanceStatistics::IncrementDroppedFrames() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_droppedFrames++;
}

void PerformanceStatistics::SetDecoderQueueDepth(int depth) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_decoderQueueDepth = depth;
}

void PerformanceStatistics::RecordApplicationFrameTime(double milliseconds) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Calculate FPS from frame time
    double fps = milliseconds > 0.0 ? 1000.0 / milliseconds : 0.0;
    AddToHistory(m_applicationFpsHistory, fps);
    m_currentApplicationFps = fps;
}

void PerformanceStatistics::RecordPresentationTime(double milliseconds) {
    // Could be used for presentation timing accuracy in the future
}

void PerformanceStatistics::RecordMainLoopTime(double milliseconds) {
    std::lock_guard<std::mutex> lock(m_mutex);
    AddToHistory(m_mainLoopTimeHistory, milliseconds);
}

void PerformanceStatistics::RecordRenderTime(double milliseconds) {
    std::lock_guard<std::mutex> lock(m_mutex);
    AddToHistory(m_renderTimeHistory, milliseconds);
}

void PerformanceStatistics::RecordTextureConversionTime(double milliseconds) {
    // Could be tracked separately in the future
}

void PerformanceStatistics::RecordRenderPassTime(double milliseconds) {
    // Could be tracked separately in the future
}

void PerformanceStatistics::SetMemoryUsage(size_t cpuMB, size_t gpuMB) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cpuMemoryMB = cpuMB;
    m_gpuMemoryMB = gpuMB;
}

void PerformanceStatistics::SetSwitchingStrategy(const std::string& strategy) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_switchingStrategy = strategy;
}

void PerformanceStatistics::AddToHistory(std::deque<double>& history, double value) {
    history.push_back(value);
    while (history.size() > MAX_SAMPLES) {
        history.pop_front();
    }
}

double PerformanceStatistics::CalculateAverage(const std::deque<double>& history) const {
    if (history.empty()) return 0.0;
    double sum = 0.0;
    for (double value : history) {
        sum += value;
    }
    return sum / history.size();
}

double PerformanceStatistics::GetMinValue(const std::deque<double>& history) const {
    if (history.empty()) return 0.0;
    return *std::min_element(history.begin(), history.end());
}

double PerformanceStatistics::GetMaxValue(const std::deque<double>& history) const {
    if (history.empty()) return 0.0;
    return *std::max_element(history.begin(), history.end());
}

void PerformanceStatistics::DrawUI() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (ImGui::CollapsingHeader("Video Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::map<std::string, std::string> videoMetrics;
        videoMetrics["Frame Rate"] = std::to_string(static_cast<int>(m_videoFrameRate)) + " FPS";
        videoMetrics["Resolution"] = std::to_string(m_videoWidth) + "x" + std::to_string(m_videoHeight);
        videoMetrics["Format"] = m_videoFormat.empty() ? "Unknown" : m_videoFormat;
        videoMetrics["Active Video"] = "Video " + std::to_string(m_activeVideoIndex + 1);
        
        std::stringstream timeStream;
        timeStream << std::fixed << std::setprecision(2) << m_currentPlaybackTime << "s";
        videoMetrics["Playback Time"] = timeStream.str();
        
        videoMetrics["Playback Speed"] = std::to_string(m_playbackSpeed) + "x";
        
        DrawMetricSection("Video", videoMetrics);
    }
    
    if (ImGui::CollapsingHeader("Decoding Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::map<std::string, std::string> decodeMetrics;
        decodeMetrics["Decoder Type"] = m_decoderType.empty() ? "Unknown" : m_decoderType;
        decodeMetrics["Dropped Frames"] = std::to_string(m_droppedFrames);
        decodeMetrics["Queue Depth"] = std::to_string(m_decoderQueueDepth);
        
        if (!m_decodeTimeHistory.empty()) {
            std::stringstream avgStream;
            avgStream << std::fixed << std::setprecision(2) << CalculateAverage(m_decodeTimeHistory) << " ms";
            decodeMetrics["Avg Decode Time"] = avgStream.str();
            
            std::stringstream minMaxStream;
            minMaxStream << std::fixed << std::setprecision(2) 
                        << GetMinValue(m_decodeTimeHistory) << " / " 
                        << GetMaxValue(m_decodeTimeHistory) << " ms";
            decodeMetrics["Min/Max Decode"] = minMaxStream.str();
        }
        
        DrawMetricSection("Decoding", decodeMetrics);
        
        if (!m_decodeTimeHistory.empty()) {
            DrawGraphSection("Decode Time History", m_decodeTimeHistory, "ms", 0.0f, 50.0f);
        }
    }
    
    if (ImGui::CollapsingHeader("Application Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::map<std::string, std::string> appMetrics;
        
        if (!m_applicationFpsHistory.empty()) {
            std::stringstream fpsStream;
            fpsStream << std::fixed << std::setprecision(1) << CalculateAverage(m_applicationFpsHistory) << " FPS";
            appMetrics["Average FPS"] = fpsStream.str();
            
            std::stringstream currentStream;
            currentStream << std::fixed << std::setprecision(1) << m_currentApplicationFps << " FPS";
            appMetrics["Current FPS"] = currentStream.str();
        }
        
        if (!m_mainLoopTimeHistory.empty()) {
            std::stringstream loopStream;
            loopStream << std::fixed << std::setprecision(2) << CalculateAverage(m_mainLoopTimeHistory) << " ms";
            appMetrics["Avg Loop Time"] = loopStream.str();
        }
        
        DrawMetricSection("Application", appMetrics);
        
        if (!m_applicationFpsHistory.empty()) {
            DrawGraphSection("Application FPS", m_applicationFpsHistory, "FPS", 0.0f, 120.0f);
        }
    }
    
    if (ImGui::CollapsingHeader("Rendering Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::map<std::string, std::string> renderMetrics;
        
        if (!m_renderTimeHistory.empty()) {
            std::stringstream renderStream;
            renderStream << std::fixed << std::setprecision(2) << CalculateAverage(m_renderTimeHistory) << " ms";
            renderMetrics["Avg Render Time"] = renderStream.str();
            
            std::stringstream minMaxStream;
            minMaxStream << std::fixed << std::setprecision(2) 
                        << GetMinValue(m_renderTimeHistory) << " / " 
                        << GetMaxValue(m_renderTimeHistory) << " ms";
            renderMetrics["Min/Max Render"] = minMaxStream.str();
        }
        
        DrawMetricSection("Rendering", renderMetrics);
        
        if (!m_renderTimeHistory.empty()) {
            DrawGraphSection("Render Time History", m_renderTimeHistory, "ms", 0.0f, 20.0f);
        }
    }
    
    if (ImGui::CollapsingHeader("System Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::map<std::string, std::string> systemMetrics;
        
        if (m_cpuMemoryMB > 0 || m_gpuMemoryMB > 0) {
            systemMetrics["CPU Memory"] = std::to_string(m_cpuMemoryMB) + " MB";
            systemMetrics["GPU Memory"] = std::to_string(m_gpuMemoryMB) + " MB";
        }
        
        systemMetrics["Switching Strategy"] = m_switchingStrategy.empty() ? "Unknown" : m_switchingStrategy;
        
        DrawMetricSection("System", systemMetrics);
    }
}

void PerformanceStatistics::DrawMetricSection(const char* title, const std::map<std::string, std::string>& metrics) {
    if (ImGui::BeginTable((std::string("##") + title).c_str(), 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        
        for (const auto& [key, value] : metrics) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", key.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%s", value.c_str());
        }
        
        ImGui::EndTable();
    }
}

void PerformanceStatistics::DrawGraphSection(const char* title, const std::deque<double>& data, const char* unit, float minVal, float maxVal) {
    if (data.empty()) return;
    
    // Convert deque to vector for ImGui
    std::vector<float> plotData;
    plotData.reserve(data.size());
    for (double value : data) {
        plotData.push_back(static_cast<float>(value));
    }
    
    ImGui::Text("%s", title);
    ImGui::PlotLines(("##" + std::string(title)).c_str(), 
                     plotData.data(), 
                     static_cast<int>(plotData.size()), 
                     0, 
                     nullptr, 
                     minVal, 
                     maxVal, 
                     ImVec2(0, 80));
    
    ImGui::SameLine();
    ImGui::Text("(%s)", unit);
}