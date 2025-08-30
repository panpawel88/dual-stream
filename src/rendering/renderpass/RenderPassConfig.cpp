#include "RenderPassConfig.h"
#include <sstream>
#include <algorithm>

std::string RenderPassConfig::GetString(const std::string& key, const std::string& defaultValue) const {
    auto it = m_parameters.find(key);
    return (it != m_parameters.end()) ? it->second : defaultValue;
}

float RenderPassConfig::GetFloat(const std::string& key, float defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

int RenderPassConfig::GetInt(const std::string& key, int defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

bool RenderPassConfig::GetBool(const std::string& key, bool defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        return (value == "true" || value == "1" || value == "yes" || value == "on");
    }
    return defaultValue;
}

std::array<float, 2> RenderPassConfig::GetFloat2(const std::string& key, const std::array<float, 2>& defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        auto values = ParseFloatArray(it->second);
        if (values.size() >= 2) {
            return {values[0], values[1]};
        }
    }
    return defaultValue;
}

std::array<float, 3> RenderPassConfig::GetFloat3(const std::string& key, const std::array<float, 3>& defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        auto values = ParseFloatArray(it->second);
        if (values.size() >= 3) {
            return {values[0], values[1], values[2]};
        }
    }
    return defaultValue;
}

std::array<float, 4> RenderPassConfig::GetFloat4(const std::string& key, const std::array<float, 4>& defaultValue) const {
    auto it = m_parameters.find(key);
    if (it != m_parameters.end()) {
        auto values = ParseFloatArray(it->second);
        if (values.size() >= 4) {
            return {values[0], values[1], values[2], values[3]};
        }
    }
    return defaultValue;
}

bool RenderPassConfig::HasParameter(const std::string& key) const {
    return m_parameters.find(key) != m_parameters.end();
}

std::map<std::string, RenderPassParameter> RenderPassConfig::GetAllParameters() const {
    std::map<std::string, RenderPassParameter> result;
    
    for (const auto& pair : m_parameters) {
        const std::string& key = pair.first;
        const std::string& value = pair.second;
        
        // Try to determine type from value
        // Check if it's a comma-separated array first
        if (value.find(',') != std::string::npos) {
            auto values = ParseFloatArray(value);
            if (values.size() == 2) {
                result[key] = std::array<float, 2>{values[0], values[1]};
            } else if (values.size() == 3) {
                result[key] = std::array<float, 3>{values[0], values[1], values[2]};
            } else if (values.size() == 4) {
                result[key] = std::array<float, 4>{values[0], values[1], values[2], values[3]};
            }
        }
        // Check if it's a boolean
        else if (value == "true" || value == "false" || value == "1" || value == "0") {
            result[key] = GetBool(key);
        }
        // Check if it's a float
        else if (value.find('.') != std::string::npos) {
            result[key] = GetFloat(key);
        }
        // Try as integer
        else {
            try {
                int intValue = std::stoi(value);
                result[key] = intValue;
            } catch (...) {
                // If all else fails, treat as float
                result[key] = GetFloat(key);
            }
        }
    }
    
    return result;
}

void RenderPassConfig::SetString(const std::string& key, const std::string& value) {
    m_parameters[key] = value;
}

void RenderPassConfig::SetFloat(const std::string& key, float value) {
    m_parameters[key] = std::to_string(value);
}

void RenderPassConfig::SetInt(const std::string& key, int value) {
    m_parameters[key] = std::to_string(value);
}

void RenderPassConfig::SetBool(const std::string& key, bool value) {
    m_parameters[key] = value ? "true" : "false";
}

std::vector<float> RenderPassConfig::ParseFloatArray(const std::string& value) const {
    std::vector<float> result;
    std::stringstream ss(value);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        try {
            result.push_back(std::stof(item));
        } catch (...) {
            // Skip invalid values
        }
    }
    
    return result;
}