#pragma once

#include <string>
#include <filesystem>

/**
 * Local Test Configuration
 * Loads machine-specific test configuration that should not be committed to VCS.
 * Similar to .env.local pattern for build configuration.
 */
class LocalTestConfig {
public:
    LocalTestConfig();
    
    // Load configuration from file
    bool LoadFromFile(const std::string& configPath = "test_config.local.json");
    
    // Get test videos base directory
    std::string GetTestVideosDirectory() const;
    
    // Set test videos directory (for CLI override)
    void SetTestVideosDirectory(const std::string& directory);
    
    // Convert relative video path to absolute using base directory
    std::string ResolveVideoPath(const std::string& relativePath) const;
    
    // Check if configuration was loaded successfully
    bool IsLoaded() const { return loaded_; }
    
    // Get default test videos directory
    static std::string GetDefaultTestVideosDirectory();

private:
    std::string testVideosDirectory_;
    bool loaded_;
    
    // Normalize directory path (ensure trailing slash)
    std::string NormalizeDirectoryPath(const std::string& path) const;
    
    // Check if path is absolute
    bool IsAbsolutePath(const std::string& path) const;
};