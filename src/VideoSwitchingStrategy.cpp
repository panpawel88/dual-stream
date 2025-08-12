#include "VideoSwitchingStrategy.h"
#include "ImmediateSwitchStrategy.h"
#include "PredecodedSwitchStrategy.h"
#include "KeyframeSwitchStrategy.h"
#include <algorithm>
#include <cctype>

std::unique_ptr<VideoSwitchingStrategy> VideoSwitchingStrategyFactory::Create(SwitchingAlgorithm algorithm) {
    switch (algorithm) {
        case SwitchingAlgorithm::IMMEDIATE:
            return std::make_unique<ImmediateSwitchStrategy>();
        case SwitchingAlgorithm::PREDECODED:
            return std::make_unique<PredecodedSwitchStrategy>();
        case SwitchingAlgorithm::KEYFRAME_SYNC:
            return std::make_unique<KeyframeSwitchStrategy>();
        default:
            return nullptr;
    }
}

SwitchingAlgorithm VideoSwitchingStrategyFactory::ParseAlgorithm(const std::string& algorithmName) {
    std::string lowerName = algorithmName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    if (lowerName == "immediate") {
        return SwitchingAlgorithm::IMMEDIATE;
    } else if (lowerName == "predecoded") {
        return SwitchingAlgorithm::PREDECODED;
    } else if (lowerName == "keyframe-sync" || lowerName == "keyframe_sync") {
        return SwitchingAlgorithm::KEYFRAME_SYNC;
    } else {
        return static_cast<SwitchingAlgorithm>(-1); // Invalid
    }
}

std::string VideoSwitchingStrategyFactory::GetAlgorithmName(SwitchingAlgorithm algorithm) {
    switch (algorithm) {
        case SwitchingAlgorithm::IMMEDIATE:
            return "immediate";
        case SwitchingAlgorithm::PREDECODED:
            return "predecoded";
        case SwitchingAlgorithm::KEYFRAME_SYNC:
            return "keyframe-sync";
        default:
            return "unknown";
    }
}