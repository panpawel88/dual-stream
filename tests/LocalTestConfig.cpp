#include "LocalTestConfig.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "../src/core/Logger.h"

LocalTestConfig::LocalTestConfig() 
    : testVideosDirectory_(GetDefaultTestVideosDirectory())
    , loaded_(false) {
}

bool LocalTestConfig::LoadFromFile(const std::string& configPath) {
    try {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            LOG_INFO("Local test config file not found: ", configPath, " (using defaults)");
            return false;
        }

        nlohmann::json configJson;
        file >> configJson;

        if (configJson.contains("test_videos_directory") && configJson["test_videos_directory"].is_string()) {
            testVideosDirectory_ = NormalizeDirectoryPath(configJson["test_videos_directory"]);
            LOG_INFO("Loaded test videos directory from config: ", testVideosDirectory_);
        }

        loaded_ = true;
        return true;

    } catch (const nlohmann::json::exception& e) {
        LOG_ERROR("JSON parsing error in local config: ", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading local test configuration: ", e.what());
        return false;
    }
}

std::string LocalTestConfig::GetTestVideosDirectory() const {
    return testVideosDirectory_;
}

void LocalTestConfig::SetTestVideosDirectory(const std::string& directory) {
    testVideosDirectory_ = NormalizeDirectoryPath(directory);
    LOG_INFO("Test videos directory set to: ", testVideosDirectory_);
}

std::string LocalTestConfig::ResolveVideoPath(const std::string& relativePath) const {
    if (IsAbsolutePath(relativePath)) {
        LOG_INFO("Using absolute path: ", relativePath);
        return relativePath;
    }
    
    std::filesystem::path basePath(testVideosDirectory_);
    
    // Smart path resolution: if the relativePath starts with "test_videos/" and 
    // testVideosDirectory_ already ends with "test_videos", strip the prefix to avoid duplication
    std::string pathToResolve = relativePath;
    if (pathToResolve.find("test_videos/") == 0) {
        std::string baseStr = basePath.string();
        // Normalize path separators for comparison
        std::replace(baseStr.begin(), baseStr.end(), '\\', '/');
        if (baseStr.ends_with("test_videos") || baseStr.ends_with("test_videos/")) {
            // Strip the "test_videos/" prefix to avoid duplication
            pathToResolve = pathToResolve.substr(12); // Remove "test_videos/"
            LOG_INFO("Stripped test_videos/ prefix from path");
        }
    }
    
    std::filesystem::path fullPath = basePath / pathToResolve;
    std::string resolvedPath = fullPath.string();
    LOG_INFO("Resolved path: ", relativePath, " -> ", resolvedPath);
    return resolvedPath;
}

std::string LocalTestConfig::GetDefaultTestVideosDirectory() {
    return "test_videos/";
}

std::string LocalTestConfig::NormalizeDirectoryPath(const std::string& path) const {
    if (path.empty()) {
        return GetDefaultTestVideosDirectory();
    }
    
    std::string normalized = path;
    
    // Convert backslashes to forward slashes for consistency
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    
    // Ensure trailing slash
    if (normalized.back() != '/') {
        normalized += '/';
    }
    
    return normalized;
}

bool LocalTestConfig::IsAbsolutePath(const std::string& path) const {
    if (path.empty()) {
        return false;
    }
    
    // Check for Windows absolute paths (C:\... or C:/...)
    if (path.length() >= 3 && path[1] == ':' && 
        (path[2] == '\\' || path[2] == '/') &&
        std::isalpha(path[0])) {
        return true;
    }
    
    // Check for Unix absolute paths (/...)
    if (path[0] == '/') {
        return true;
    }
    
    return false;
}