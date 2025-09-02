#pragma once
#include <string>

class NotificationManager {
public:
    static NotificationManager& GetInstance();
    
    void ShowSuccess(const std::string& title, const std::string& content);
    void ShowWarning(const std::string& title, const std::string& content);
    void ShowError(const std::string& title, const std::string& content);
    void ShowInfo(const std::string& title, const std::string& content);
    
    void Update();
    
private:
    NotificationManager() = default;
    ~NotificationManager() = default;
};