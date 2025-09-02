#pragma once

#include <memory>
#include <string>
#include "ISwitchingTrigger.h"
#include "../../camera/processing/FaceDetectionSwitchingTrigger.h"

// Forward declarations
class Window;
class CameraManager;

/**
 * Enumeration of available trigger types.
 */
enum class TriggerType {
    KEYBOARD,  // Default keyboard input trigger (keys 1 and 2)
    FACE_DETECTION, // Face detection camera trigger
    // Future trigger types can be added here:
    // TIMER,     // Time-based switching
    // NETWORK,   // Network/remote control trigger
    // AUDIO,     // Audio level-based trigger
    // CUSTOM     // Custom user-defined trigger
};

/**
 * Configuration structure for trigger creation.
 * Contains type-specific parameters for different trigger types.
 */
struct TriggerConfig {
    // Keyboard trigger parameters
    Window* window = nullptr;
    size_t videoCount = 2; // Number of videos available for switching (default 2)
    
    // Face detection trigger parameters
    FaceDetectionSwitchingTrigger::FaceDetectionConfig faceDetectionConfig;
    CameraManager* cameraManager = nullptr;
    
    // Future trigger parameters can be added here
};

/**
 * Factory for creating video switching trigger instances.
 */
class SwitchingTriggerFactory {
public:
    /**
     * Create a switching trigger of the specified type.
     * @param triggerType The type of trigger to create
     * @param config Configuration parameters for the trigger
     * @return Shared pointer to the created trigger, or nullptr if creation failed
     */
    static std::shared_ptr<ISwitchingTrigger> Create(TriggerType triggerType, const TriggerConfig& config = TriggerConfig());

    /**
     * Parse trigger type from string name.
     * @param triggerName String representation of trigger type
     * @return Parsed trigger type, defaults to KEYBOARD if parsing fails
     */
    static TriggerType ParseTriggerType(const std::string& triggerName);
    
    /**
     * Get string name of trigger type.
     * @param triggerType The trigger type to get name for
     * @return String name of the trigger type
     */
    static std::string GetTriggerTypeName(TriggerType triggerType);
};