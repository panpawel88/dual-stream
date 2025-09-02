#include "NotificationManager.h"
#include "ImGuiNotify.hpp"
#include "imgui.h"

NotificationManager& NotificationManager::GetInstance() {
    static NotificationManager instance;
    return instance;
}

void NotificationManager::ShowSuccess(const std::string& title, const std::string& content) {
    ImGuiToast toast(ImGuiToastType::Success, 3000, content.c_str());
    toast.setTitle(title.c_str());
    ImGui::InsertNotification(toast);
}

void NotificationManager::ShowWarning(const std::string& title, const std::string& content) {
    ImGuiToast toast(ImGuiToastType::Warning, 3000, content.c_str());
    toast.setTitle(title.c_str());
    ImGui::InsertNotification(toast);
}

void NotificationManager::ShowError(const std::string& title, const std::string& content) {
    ImGuiToast toast(ImGuiToastType::Error, 5000, content.c_str());
    toast.setTitle(title.c_str());
    ImGui::InsertNotification(toast);
}

void NotificationManager::ShowInfo(const std::string& title, const std::string& content) {
    ImGuiToast toast(ImGuiToastType::Info, 3000, content.c_str());
    toast.setTitle(title.c_str());
    ImGui::InsertNotification(toast);
}

void NotificationManager::Update() {
    ImGui::RenderNotifications();
}