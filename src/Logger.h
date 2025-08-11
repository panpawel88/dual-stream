#pragma once

#include <iostream>
#include <string>
#include <sstream>

enum class LogLevel {
    Error = 0,
    Info = 1,
    Debug = 2
};

class Logger {
public:
    static Logger& GetInstance();
    
    void SetLogLevel(LogLevel level);
    LogLevel GetLogLevel() const;
    
    void Error(const std::string& message);
    void Info(const std::string& message);
    void Debug(const std::string& message);
    
    template<typename... Args>
    void Error(Args&&... args) {
        if (m_logLevel >= LogLevel::Error) {
            LogMessage("[ERROR] ", args...);
        }
    }
    
    template<typename... Args>
    void Info(Args&&... args) {
        if (m_logLevel >= LogLevel::Info) {
            LogMessage("[INFO] ", args...);
        }
    }
    
    template<typename... Args>
    void Debug(Args&&... args) {
        if (m_logLevel >= LogLevel::Debug) {
            LogMessage("[DEBUG] ", args...);
        }
    }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    template<typename... Args>
    void LogMessage(const std::string& prefix, Args&&... args) {
        std::ostringstream oss;
        oss << prefix;
        (oss << ... << args);
        oss << "\n";
        std::cout << oss.str();
    }
    
    LogLevel m_logLevel = LogLevel::Info;
};

#define LOG_ERROR(...) Logger::GetInstance().Error(__VA_ARGS__)
#define LOG_INFO(...) Logger::GetInstance().Info(__VA_ARGS__)
#define LOG_DEBUG(...) Logger::GetInstance().Debug(__VA_ARGS__)