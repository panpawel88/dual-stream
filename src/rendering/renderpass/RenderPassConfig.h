#pragma once

#include "RenderPass.h"
#include <string>
#include <map>
#include <vector>

/**
 * Configuration data for a render pass parsed from INI file
 */
class RenderPassConfig {
public:
    RenderPassConfig() = default;
    
    // Parameter access methods
    std::string GetString(const std::string& key, const std::string& defaultValue = "") const;
    float GetFloat(const std::string& key, float defaultValue = 0.0f) const;
    int GetInt(const std::string& key, int defaultValue = 0) const;
    bool GetBool(const std::string& key, bool defaultValue = false) const;
    
    // Vector parameter access
    std::array<float, 2> GetFloat2(const std::string& key, const std::array<float, 2>& defaultValue = {0.0f, 0.0f}) const;
    std::array<float, 3> GetFloat3(const std::string& key, const std::array<float, 3>& defaultValue = {0.0f, 0.0f, 0.0f}) const;
    std::array<float, 4> GetFloat4(const std::string& key, const std::array<float, 4>& defaultValue = {0.0f, 0.0f, 0.0f, 0.0f}) const;
    
    // Check if parameter exists
    bool HasParameter(const std::string& key) const;
    
    // Get all parameters as render pass parameters
    std::map<std::string, RenderPassParameter> GetAllParameters() const;
    
    // Set parameter values (used by INI parser)
    void SetString(const std::string& key, const std::string& value);
    void SetFloat(const std::string& key, float value);
    void SetInt(const std::string& key, int value);
    void SetBool(const std::string& key, bool value);

private:
    std::map<std::string, std::string> m_parameters;
    
    // Helper methods for parsing
    std::vector<float> ParseFloatArray(const std::string& value) const;
};