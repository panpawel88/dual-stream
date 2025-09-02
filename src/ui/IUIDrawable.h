#pragma once
#include <string>

/**
 * Interface for modules that can provide runtime UI controls.
 * Implementing classes can register themselves with UIRegistry
 * to have their UI automatically rendered in the overlay.
 */
class IUIDrawable {
public:
    virtual ~IUIDrawable() = default;
    
    /**
     * Draw the UI for this module.
     * Called within an ImGui context during overlay rendering.
     * Implementations should use ImGui widgets directly.
     */
    virtual void DrawUI() = 0;
    
    /**
     * Get the display name for this module in the UI.
     * Used as the section header in the overlay.
     */
    virtual std::string GetUIName() const = 0;
    
    /**
     * Whether this module's UI should be shown by default.
     * Returns false to start with collapsed/hidden UI section.
     */
    virtual bool IsUIVisibleByDefault() const { return true; }
    
    /**
     * Get UI category for grouping related modules together.
     * Examples: "Render Passes", "Video Processing", "Camera", etc.
     */
    virtual std::string GetUICategory() const { return "General"; }
};