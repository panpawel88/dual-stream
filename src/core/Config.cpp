#include "Config.h"
#include "Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

std::unique_ptr<Config> Config::s_instance = nullptr;
std::mutex Config::s_instanceMutex;

Config* Config::GetInstance() {
    std::lock_guard<std::mutex> lock(s_instanceMutex);
    if (!s_instance) {
        s_instance = std::unique_ptr<Config>(new Config());
        s_instance->LoadDefaults();
    }
    return s_instance.get();
}

void Config::LoadDefaults() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Clear existing values
    m_values.clear();
    
    // Video settings
    m_values["video.default_speed"] = "1.0";
    m_values["video.fallback_fps"] = "30.0";
    m_values["video.max_resolution_width"] = "7680";
    m_values["video.max_resolution_height"] = "4320";
    
    // Window settings
    m_values["window.default_width"] = "1280";
    m_values["window.default_height"] = "720";
    m_values["window.limit_to_display"] = "true";
    
    // Camera settings
    m_values["camera.default_width"] = "640";
    m_values["camera.default_height"] = "480";
    m_values["camera.default_fps"] = "30";
    m_values["camera.default_format"] = "BGR8";
    m_values["camera.brightness"] = "-1";
    m_values["camera.enable_depth"] = "false";
    
    // Face detection settings
    m_values["face_detection.algorithm"] = "haar_cascade";
    m_values["face_detection.min_face_size"] = "30";
    m_values["face_detection.max_face_size"] = "300";
    m_values["face_detection.scale_factor"] = "1.1";
    m_values["face_detection.min_neighbors"] = "3";
    m_values["face_detection.stability_frames"] = "5";
    m_values["face_detection.switch_cooldown_ms"] = "2000";
    m_values["face_detection.single_face_threshold"] = "1";
    m_values["face_detection.multiple_face_threshold"] = "2";
    m_values["face_detection.enable_preview"] = "false";
    m_values["face_detection.enable_visualization"] = "false";
    m_values["face_detection.score_threshold"] = "0.9";
    m_values["face_detection.nms_threshold"] = "0.3";
    m_values["face_detection.input_width"] = "320";
    m_values["face_detection.input_height"] = "320";
    
    // Rendering settings
    m_values["rendering.preferred_backend"] = "auto";
    
    // Logging settings
    m_values["logging.default_level"] = "info";
    
    // Frame publisher settings
    m_values["frame_publisher.max_frame_queue_size"] = "5";
    m_values["frame_publisher.max_worker_threads"] = "2";
    m_values["frame_publisher.max_frame_age_ms"] = "100.0";
    m_values["frame_publisher.enable_frame_skipping"] = "true";
    m_values["frame_publisher.enable_priority_processing"] = "true";
    m_values["frame_publisher.enable_performance_logging"] = "false";
    m_values["frame_publisher.stats_report_interval_ms"] = "5000.0";
}

bool Config::LoadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        LOG_WARNING("Could not open config file: ", filePath, ". Using defaults.");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    m_configFilePath = filePath;
    
    std::string line;
    std::string currentSection;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }
        
        // Check for section header
        if (line[0] == '[' && line.back() == ']') {
            currentSection = line.substr(1, line.length() - 2);
            continue;
        }
        
        // Parse key-value pair
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            LOG_WARNING("Invalid config line ", lineNumber, " in ", filePath, ": ", line);
            continue;
        }
        
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);
        
        // Trim whitespace from key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Remove inline comments from value
        size_t commentPos = value.find(';');
        if (commentPos != std::string::npos) {
            value = value.substr(0, commentPos);
            value.erase(value.find_last_not_of(" \t") + 1);
        }
        
        // Build full key with section
        std::string fullKey = currentSection.empty() ? key : currentSection + "." + key;
        m_values[fullKey] = value;
    }
    
    LOG_INFO("Loaded configuration from: ", filePath);
    return true;
}

bool Config::SaveToFile(const std::string& filePath) {
    std::string targetPath = filePath.empty() ? m_configFilePath : filePath;
    if (targetPath.empty()) {
        LOG_ERROR("No file path specified for saving configuration");
        return false;
    }
    
    std::ofstream file(targetPath);
    if (!file.is_open()) {
        LOG_ERROR("Could not open config file for writing: ", targetPath);
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Organize values by section
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> sections;
    
    for (const auto& pair : m_values) {
        size_t dotPos = pair.first.find('.');
        if (dotPos != std::string::npos) {
            std::string section = pair.first.substr(0, dotPos);
            std::string key = pair.first.substr(dotPos + 1);
            sections[section].emplace_back(key, pair.second);
        } else {
            sections[""].emplace_back(pair.first, pair.second);
        }
    }
    
    // Write configuration with comments
    file << "; DualStream Video Player Configuration File\n";
    file << "; Generated automatically - modifications will be preserved\n\n";
    
    for (const auto& section : sections) {
        if (!section.first.empty()) {
            file << "[" << section.first << "]\n";
        }
        
        for (const auto& keyValue : section.second) {
            file << keyValue.first << " = " << keyValue.second << "\n";
        }
        
        file << "\n";
    }
    
    if (!filePath.empty()) {
        m_configFilePath = filePath;
    }
    
    LOG_INFO("Saved configuration to: ", targetPath);
    return true;
}

bool Config::GetBool(const std::string& key, bool defaultValue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(key);
    if (it == m_values.end()) {
        return defaultValue;
    }
    return ParseBool(it->second);
}

int Config::GetInt(const std::string& key, int defaultValue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(key);
    if (it == m_values.end()) {
        return defaultValue;
    }
    try {
        return ParseInt(it->second);
    } catch (const std::exception& e) {
        LOG_WARNING("Invalid integer value for key '", key, "': ", it->second, ". Using default: ", defaultValue);
        return defaultValue;
    }
}

float Config::GetFloat(const std::string& key, float defaultValue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(key);
    if (it == m_values.end()) {
        return defaultValue;
    }
    try {
        return ParseFloat(it->second);
    } catch (const std::exception& e) {
        LOG_WARNING("Invalid float value for key '", key, "': ", it->second, ". Using default: ", defaultValue);
        return defaultValue;
    }
}

double Config::GetDouble(const std::string& key, double defaultValue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(key);
    if (it == m_values.end()) {
        return defaultValue;
    }
    try {
        return ParseDouble(it->second);
    } catch (const std::exception& e) {
        LOG_WARNING("Invalid double value for key '", key, "': ", it->second, ". Using default: ", defaultValue);
        return defaultValue;
    }
}

std::string Config::GetString(const std::string& key, const std::string& defaultValue) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_values.find(key);
    if (it == m_values.end()) {
        return defaultValue;
    }
    return it->second;
}

void Config::SetBool(const std::string& key, bool value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values[key] = ToString(value);
}

void Config::SetInt(const std::string& key, int value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values[key] = ToString(value);
}

void Config::SetFloat(const std::string& key, float value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values[key] = ToString(value);
}

void Config::SetDouble(const std::string& key, double value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values[key] = ToString(value);
}

void Config::SetString(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values[key] = value;
}

bool Config::HasKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_values.find(key) != m_values.end();
}

std::vector<std::string> Config::GetKeysInSection(const std::string& section) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> keys;
    
    std::string prefix = section + ".";
    for (const auto& pair : m_values) {
        if (pair.first.find(prefix) == 0) {
            keys.push_back(pair.first);
        }
    }
    
    return keys;
}

void Config::RemoveKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values.erase(key);
}

void Config::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_values.clear();
}

const std::string& Config::GetConfigFilePath() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_configFilePath;
}

// Private helper methods
bool Config::ParseBool(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "true" || lower == "yes" || lower == "1" || lower == "on";
}

int Config::ParseInt(const std::string& value) {
    return std::stoi(value);
}

float Config::ParseFloat(const std::string& value) {
    return std::stof(value);
}

double Config::ParseDouble(const std::string& value) {
    return std::stod(value);
}

std::string Config::ToString(bool value) {
    return value ? "true" : "false";
}

std::string Config::ToString(int value) {
    return std::to_string(value);
}

std::string Config::ToString(float value) {
    return std::to_string(value);
}

std::string Config::ToString(double value) {
    return std::to_string(value);
}

std::string Config::ToString(const std::string& value) {
    return value;
}