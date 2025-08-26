#include "SwitchingTriggerFactory.h"
#include "KeyboardSwitchingTrigger.h"
#include "../../camera/processing/FaceDetectionSwitchingTrigger.h"
#include <algorithm>
#include <cctype>

std::shared_ptr<ISwitchingTrigger> SwitchingTriggerFactory::Create(TriggerType triggerType, const TriggerConfig& config) {
    switch (triggerType) {
        case TriggerType::KEYBOARD:
            if (!config.window) {
                return nullptr; // Keyboard trigger requires window instance
            }
            return std::make_shared<KeyboardSwitchingTrigger>(config.window);
            
        case TriggerType::FACE_DETECTION: {
            auto trigger = std::make_shared<FaceDetectionSwitchingTrigger>(config.faceDetectionConfig);
            return trigger;
        }
            
        default:
            // Default to keyboard trigger if type is unrecognized
            if (config.window) {
                return std::make_shared<KeyboardSwitchingTrigger>(config.window);
            }
            return nullptr;
    }
}

std::shared_ptr<ISwitchingTrigger> SwitchingTriggerFactory::Create(TriggerType triggerType, Window* window) {
    TriggerConfig config;
    config.window = window;
    return Create(triggerType, config);
}

TriggerType SwitchingTriggerFactory::ParseTriggerType(const std::string& triggerName) {
    std::string lowerName = triggerName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](char c) {
        return std::tolower(c);
    });
    
    if (lowerName == "keyboard") {
        return TriggerType::KEYBOARD;
    }
    
    if (lowerName == "face" || lowerName == "face_detection") {
        return TriggerType::FACE_DETECTION;
    }
    
    // Add more trigger types here as they are implemented:
    // if (lowerName == "timer") return TriggerType::TIMER;
    // if (lowerName == "network") return TriggerType::NETWORK;
    // if (lowerName == "audio") return TriggerType::AUDIO;
    // if (lowerName == "custom") return TriggerType::CUSTOM;
    
    // Default to keyboard if not recognized
    return TriggerType::KEYBOARD;
}

std::string SwitchingTriggerFactory::GetTriggerTypeName(TriggerType triggerType) {
    switch (triggerType) {
        case TriggerType::KEYBOARD:
            return "Keyboard";
            
        case TriggerType::FACE_DETECTION:
            return "Face Detection";
            
        default:
            return "Unknown";
    }
}