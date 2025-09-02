#pragma once

// Forward declarations
struct ToastMessage;
struct ToastConfig;

/**
 * Abstract interface for platform-specific toast rendering.
 * Each graphics backend (D3D11, OpenGL) implements this interface
 * to render toast notifications on top of video content.
 */
class IToastRenderer {
public:
    virtual ~IToastRenderer() = default;
    
    /**
     * Initialize the toast renderer with the given configuration.
     * @param config Toast configuration settings
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool Initialize(const ToastConfig& config) = 0;
    
    /**
     * Render a toast message with the current alpha value.
     * @param toast The toast message to render
     */
    virtual void RenderToast(const ToastMessage& toast) = 0;
    
    /**
     * Update any animation or timing state (called each frame).
     * Some renderers may need to update internal state.
     */
    virtual void Update() {}
    
    /**
     * Clean up any resources allocated by the renderer.
     */
    virtual void Cleanup() = 0;
    
    /**
     * Check if the renderer is properly initialized.
     * @return true if ready to render, false otherwise
     */
    virtual bool IsInitialized() const = 0;
};