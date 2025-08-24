#include "SwitchingTriggerFactory.h"
#include "KeyboardSwitchingTrigger.h"
#include <algorithm>
#include <cctype>

std::unique_ptr<ISwitchingTrigger> SwitchingTriggerFactory::Create(TriggerType triggerType, Window* window) {
    switch (triggerType) {
        case TriggerType::KEYBOARD:
            if (!window) {
                return nullptr; // Keyboard trigger requires window instance
            }
            return std::make_unique<KeyboardSwitchingTrigger>(window);
            
        default:
            // Default to keyboard trigger if type is unrecognized
            if (window) {
                return std::make_unique<KeyboardSwitchingTrigger>(window);
            }
            return nullptr;
    }
}

TriggerType SwitchingTriggerFactory::ParseTriggerType(const std::string& triggerName) {
    std::string lowerName = triggerName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](char c) {
        return std::tolower(c);
    });
    
    if (lowerName == "keyboard") {
        return TriggerType::KEYBOARD;
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
            
        default:
            return "Unknown";
    }
}