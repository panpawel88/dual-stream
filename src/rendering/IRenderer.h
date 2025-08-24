#pragma once

#include <windows.h>
#include "RenderTexture.h"

/**
 * Renderer type enumeration
 */
enum class RendererType {
    OpenGL,
    DirectX11
};

/**
 * Abstract renderer interface with clean separation of concerns.
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
     * Get the renderer type.
     */
    virtual RendererType GetRendererType() const = 0;
    
    /**
     * Check if the renderer supports CUDA interop (for hardware frame optimization).
     * Returns false for renderers that don't support CUDA.
     */
    virtual bool SupportsCudaInterop() const = 0;
};