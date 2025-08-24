#pragma once

#include <windows.h>
#include "RenderTexture.h"

/**
 * Renderer type enumeration for downcasting when platform-specific access is needed
 */
enum class RendererType {
    OpenGL,
    DirectX11
};

/**
 * Abstract renderer interface with clean separation of concerns.
 * Renderers work with generic RenderTexture objects and don't need to know
 * about video decoding specifics or different graphics API details.
 */
class IRenderer {
public:
    virtual ~IRenderer() = default;
    
    /**
     * Initialize the renderer with the given window and dimensions.
     */
    virtual bool Initialize(HWND hwnd, int width, int height) = 0;
    
    /**
     * Clean up renderer resources.
     */
    virtual void Cleanup() = 0;
    
    /**
     * Present a texture to the screen.
     * This is the main rendering method that handles all texture types.
     */
    virtual bool Present(const RenderTexture& texture) = 0;
    
    /**
     * Resize the renderer viewport.
     */
    virtual bool Resize(int width, int height) = 0;
    
    /**
     * Check if renderer is initialized.
     */
    virtual bool IsInitialized() const = 0;
    
    /**
     * Get the renderer type for downcasting when platform-specific access is needed.
     */
    virtual RendererType GetRendererType() const = 0;
    
    /**
     * Check if the renderer supports CUDA interop (for hardware frame optimization).
     * Returns false for renderers that don't support CUDA.
     */
    virtual bool SupportsCudaInterop() const = 0;
};