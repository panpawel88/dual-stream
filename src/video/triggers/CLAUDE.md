# Video Switching Trigger System

This directory implements the trigger system that determines **when** video switching should occur, completely separated from **how** the switching is performed (which is handled by switching strategies).

## Architecture Overview

The trigger system uses the Strategy pattern to enable different input methods and switching triggers:
- **ISwitchingTrigger** - Abstract interface for trigger strategies
- **SwitchingTriggerFactory** - Factory for creating trigger instances  
- **Concrete Triggers** - Specific trigger implementations

## Core Interface

### ISwitchingTrigger
**File:** `ISwitchingTrigger.h`
**Purpose:** Abstract interface defining trigger behavior

**Key Interface Methods:**
```cpp
virtual bool ShouldSwitchToVideo1() = 0;    // Check for video 1 switch trigger
virtual bool ShouldSwitchToVideo2() = 0;    // Check for video 2 switch trigger
virtual void Update() = 0;                  // Update trigger state each frame
virtual void Reset() = 0;                   // Reset after switch completed
virtual std::string GetName() const = 0;    // Trigger identification
```

**Design Principles:**
- **Stateless Querying:** Triggers maintain internal state, expose via boolean queries
- **Frame-Based Updates:** `Update()` called each frame to process input
- **Reset After Action:** `Reset()` prevents repeated triggering from same input
- **Separation of Concerns:** Triggers decide "when", not "how" to switch

## Trigger Types

### TriggerType Enumeration
**File:** `SwitchingTriggerFactory.h`
```cpp
enum class TriggerType {
    KEYBOARD,  // Default keyboard input trigger (keys 1 and 2)
    // Future extensions:
    // TIMER,     // Time-based automatic switching
    // NETWORK,   // Network/remote control trigger  
    // AUDIO,     // Audio level-based trigger
    // CUSTOM     // Custom user-defined trigger
};
```

## Implemented Triggers

### KeyboardSwitchingTrigger  
**File:** `KeyboardSwitchingTrigger.h/cpp`
**Purpose:** Default keyboard-based trigger implementing original key 1/2 behavior

**Key Features:**
- **Key Detection:** Monitors '1' and '2' key presses via Window input system
- **Edge Triggering:** Triggers only on key press edges (not key holds)
- **State Management:** Prevents repeated triggering until key release
- **Clean Integration:** Uses existing Window key input infrastructure

**Implementation Logic:**
```cpp
class KeyboardSwitchingTrigger : public ISwitchingTrigger {
private:
    Window* m_window;           // Window for input access
    bool m_key1Triggered;       // Edge trigger state for key 1
    bool m_key2Triggered;       // Edge trigger state for key 2
    
public:
    void Update() override {
        // Check for key press edges and set trigger flags
        if (m_window->IsKeyPressed('1') && !m_key1Triggered) {
            m_key1Triggered = true;
        }
        if (m_window->IsKeyPressed('2') && !m_key2Triggered) {
            m_key2Triggered = true;
        }
    }
    
    bool ShouldSwitchToVideo1() override { return m_key1Triggered; }
    bool ShouldSwitchToVideo2() override { return m_key2Triggered; }
    
    void Reset() override {
        // Clear triggers and underlying key states
        if (m_key1Triggered) {
            m_window->ClearKeyPress('1');
            m_key1Triggered = false;
        }
        if (m_key2Triggered) {
            m_window->ClearKeyPress('2');  
            m_key2Triggered = false;
        }
    }
};
```

### Factory Pattern Implementation

### SwitchingTriggerFactory
**File:** `SwitchingTriggerFactory.h/cpp`
**Purpose:** Creates appropriate trigger instances based on type

**Factory Methods:**
```cpp
// Create trigger instance
static std::unique_ptr<ISwitchingTrigger> Create(TriggerType triggerType, Window* window = nullptr);

// Parse trigger type from command line
static TriggerType ParseTriggerType(const std::string& triggerName);

// Get human-readable trigger name
static std::string GetTriggerTypeName(TriggerType triggerType);
```

**Creation Logic:**
```cpp
std::unique_ptr<ISwitchingTrigger> SwitchingTriggerFactory::Create(TriggerType triggerType, Window* window) {
    switch (triggerType) {
        case TriggerType::KEYBOARD:
            if (!window) return nullptr; // Keyboard trigger requires window
            return std::make_unique<KeyboardSwitchingTrigger>(window);
            
        default:
            // Default to keyboard if type unrecognized
            if (window) return std::make_unique<KeyboardSwitchingTrigger>(window);
            return nullptr;
    }
}
```

## Integration with VideoManager

### Trigger Lifecycle Management
```cpp
// VideoManager initialization
auto switchingTrigger = SwitchingTriggerFactory::Create(args.triggerType, &window);
videoManager.SetSwitchingTrigger(std::move(switchingTrigger));

// Main application loop
while (window.ProcessMessages()) {
    // Update trigger state
    videoManager.UpdateSwitchingTrigger();
    
    // Process any triggered switches
    videoManager.ProcessSwitchingTriggers();
    
    // Continue with frame processing...
}
```

### Trigger Processing in VideoManager
```cpp
void VideoManager::UpdateSwitchingTrigger() {
    if (m_switchingTrigger) {
        m_switchingTrigger->Update();  // Update trigger state
    }
}

bool VideoManager::ProcessSwitchingTriggers() {
    if (!m_switchingTrigger) return false;
    
    bool switched = false;
    
    // Check for switch to video 1
    if (m_switchingTrigger->ShouldSwitchToVideo1()) {
        LOG_INFO("Trigger initiated switch to video 1");
        if (SwitchToVideo(ActiveVideo::VIDEO_1)) {
            switched = true;
        }
    }
    
    // Check for switch to video 2
    if (m_switchingTrigger->ShouldSwitchToVideo2()) {
        LOG_INFO("Trigger initiated switch to video 2");
        if (SwitchToVideo(ActiveVideo::VIDEO_2)) {
            switched = true;
        }
    }
    
    // Reset trigger state after successful switch
    if (switched) {
        m_switchingTrigger->Reset();
    }
    
    return switched;
}
```

## Command Line Integration

### Trigger Selection
```cpp
// Command line parsing supports trigger selection
struct VideoPlayerArgs {
    TriggerType triggerType;  // Default: TriggerType::KEYBOARD
    // ...
};

// Usage examples:
// ./ffmpeg_player video1.mp4 video2.mp4 --trigger keyboard
// ./ffmpeg_player video1.mp4 video2.mp4 -t keyboard
```

### Parser Integration
```cpp
TriggerType SwitchingTriggerFactory::ParseTriggerType(const std::string& triggerName) {
    std::string lowerName = triggerName;
    std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
    
    if (lowerName == "keyboard") return TriggerType::KEYBOARD;
    // Future: timer, network, audio, custom...
    
    return TriggerType::KEYBOARD; // Default fallback
}
```

## Extensibility Framework

### Adding New Triggers

**Step 1:** Add to TriggerType enum
```cpp
enum class TriggerType {
    KEYBOARD,
    TIMER,      // New: Time-based automatic switching
    NETWORK,    // New: Network/remote control trigger
    // ...
};
```

**Step 2:** Implement ISwitchingTrigger
```cpp
class TimerSwitchingTrigger : public ISwitchingTrigger {
private:
    std::chrono::steady_clock::time_point m_lastSwitchTime;
    double m_switchInterval;  // Seconds between switches
    ActiveVideo m_nextVideo;
    
public:
    void Update() override {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastSwitchTime);
        
        if (elapsed.count() / 1000.0 >= m_switchInterval) {
            // Time to switch
        }
    }
    
    // Implement other interface methods...
};
```

**Step 3:** Update Factory
```cpp
std::unique_ptr<ISwitchingTrigger> SwitchingTriggerFactory::Create(TriggerType type, Window* window) {
    switch (type) {
        case TriggerType::KEYBOARD: return std::make_unique<KeyboardSwitchingTrigger>(window);
        case TriggerType::TIMER:    return std::make_unique<TimerSwitchingTrigger>();
        // ...
    }
}
```

## Future Trigger Examples

### Timer-Based Trigger
- **Use Case:** Automatic slideshow-style switching
- **Configuration:** Switch interval, sequence pattern
- **State:** Last switch time, next target video

### Network Trigger  
- **Use Case:** Remote control via network commands
- **Configuration:** Network port, command protocol
- **State:** Connection status, pending commands

### Audio-Level Trigger
- **Use Case:** Switch based on audio input levels
- **Configuration:** Audio threshold, input device
- **State:** Current audio level, trigger hysteresis

### Custom Script Trigger
- **Use Case:** User-defined switching logic
- **Configuration:** Script path, execution parameters  
- **State:** Script output, execution status

## Thread Safety and Performance

### Thread Safety
- **Single-Threaded Design:** All triggers operate on main thread
- **Window Integration:** Keyboard trigger uses existing Window input system
- **State Isolation:** Each trigger maintains independent state

### Performance Considerations
- **Minimal Overhead:** Trigger updates are lightweight operations
- **Edge Detection:** Only process state changes, not continuous states
- **Efficient Polling:** Integration with existing message loop

This trigger system provides a clean, extensible framework for implementing various switching input methods while maintaining separation from the actual switching logic.