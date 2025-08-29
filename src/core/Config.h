#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>

/**
 * Global configuration manager that provides centralized access to application settings.
 * Uses singleton pattern for easy access throughout the application.
 * 
 * Features:
 * - Type-safe access methods (GetBool, GetInt, GetFloat, GetString)
 * - Hierarchical configuration with dot notation (e.g., "video.default_algorithm")
 * - Default value support for all getters
 * - Thread-safe access
 * - INI file loading and saving
 * - Runtime configuration updates
 * 
 * Usage Examples:
 * ```cpp
 * Config* cfg = Config::GetInstance();
 * bool hwDecode = cfg->GetBool("video.enable_hardware_decode", true);
 * int width = cfg->GetInt("window.default_width", 1280);
 * float speed = cfg->GetFloat("video.default_speed", 1.0f);
 * std::string algo = cfg->GetString("video.default_algorithm", "immediate");
 * 
 * // Runtime updates
 * cfg->SetInt("window.default_width", 1920);
 * cfg->SaveToFile();
 * ```
 */
class Config {
public:
    /**
     * Get the global configuration instance
     */
    static Config* GetInstance();

    /**
     * Load configuration from INI file
     * @param filePath Path to INI file
     * @return true if loaded successfully, false otherwise
     */
    bool LoadFromFile(const std::string& filePath);

    /**
     * Save current configuration to INI file
     * @param filePath Path to save INI file (uses loaded path if empty)
     * @return true if saved successfully, false otherwise
     */
    bool SaveToFile(const std::string& filePath = "");

    /**
     * Load default configuration values
     * Called automatically during initialization
     */
    void LoadDefaults();

    /**
     * Get boolean value from configuration
     * @param key Configuration key (e.g., "video.enable_hardware_decode")
     * @param defaultValue Value to return if key not found
     * @return Configuration value or default
     */
    bool GetBool(const std::string& key, bool defaultValue = false);

    /**
     * Get integer value from configuration
     * @param key Configuration key (e.g., "window.default_width")
     * @param defaultValue Value to return if key not found
     * @return Configuration value or default
     */
    int GetInt(const std::string& key, int defaultValue = 0);

    /**
     * Get float value from configuration
     * @param key Configuration key (e.g., "video.default_speed")
     * @param defaultValue Value to return if key not found
     * @return Configuration value or default
     */
    float GetFloat(const std::string& key, float defaultValue = 0.0f);

    /**
     * Get double value from configuration
     * @param key Configuration key (e.g., "face_detection.scale_factor")
     * @param defaultValue Value to return if key not found
     * @return Configuration value or default
     */
    double GetDouble(const std::string& key, double defaultValue = 0.0);

    /**
     * Get string value from configuration
     * @param key Configuration key (e.g., "video.default_algorithm")
     * @param defaultValue Value to return if key not found
     * @return Configuration value or default
     */
    std::string GetString(const std::string& key, const std::string& defaultValue = "");

    /**
     * Set boolean value in configuration
     * @param key Configuration key
     * @param value Value to set
     */
    void SetBool(const std::string& key, bool value);

    /**
     * Set integer value in configuration
     * @param key Configuration key
     * @param value Value to set
     */
    void SetInt(const std::string& key, int value);

    /**
     * Set float value in configuration
     * @param key Configuration key
     * @param value Value to set
     */
    void SetFloat(const std::string& key, float value);

    /**
     * Set double value in configuration
     * @param key Configuration key
     * @param value Value to set
     */
    void SetDouble(const std::string& key, double value);

    /**
     * Set string value in configuration
     * @param key Configuration key
     * @param value Value to set
     */
    void SetString(const std::string& key, const std::string& value);

    /**
     * Check if a configuration key exists
     * @param key Configuration key to check
     * @return true if key exists, false otherwise
     */
    bool HasKey(const std::string& key);

    /**
     * Get all keys in a section
     * @param section Section name (e.g., "video", "camera")
     * @return Vector of full keys in the section
     */
    std::vector<std::string> GetKeysInSection(const std::string& section);

    /**
     * Remove a configuration key
     * @param key Configuration key to remove
     */
    void RemoveKey(const std::string& key);

    /**
     * Clear all configuration data
     */
    void Clear();

    /**
     * Get the path of the currently loaded configuration file
     * @return File path or empty string if no file loaded
     */
    const std::string& GetConfigFilePath() const;

private:
    Config() = default;
    ~Config() = default;

    // Prevent copying
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    /**
     * Parse a string value to the appropriate type
     */
    bool ParseBool(const std::string& value);
    int ParseInt(const std::string& value);
    float ParseFloat(const std::string& value);
    double ParseDouble(const std::string& value);

    /**
     * Convert values to string for saving
     */
    std::string ToString(bool value);
    std::string ToString(int value);
    std::string ToString(float value);
    std::string ToString(double value);
    std::string ToString(const std::string& value);

    /**
     * Thread-safe access to configuration data
     */
    mutable std::mutex m_mutex;

    /**
     * Configuration storage - key-value pairs as strings
     */
    std::unordered_map<std::string, std::string> m_values;

    /**
     * Path to the loaded configuration file
     */
    std::string m_configFilePath;

    /**
     * Static instance for singleton pattern
     */
    static std::unique_ptr<Config> s_instance;
    static std::mutex s_instanceMutex;
};