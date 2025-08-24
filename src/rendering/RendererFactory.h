#pragma once

#include <memory>
#include "IRenderer.h"

/**
 * Factory class for creating renderer instances.
 * Encapsulates all preprocessor logic for renderer selection.
 */
class RendererFactory {
public:
    /**
     * Creates the appropriate renderer based on compile-time configuration.
     * Returns OpenGL renderer if USE_OPENGL_RENDERER is defined, otherwise D3D11 renderer.
     */
    static std::unique_ptr<IRenderer> CreateRenderer();
    
    /**
     * Gets the name of the renderer that will be created.
     */
    static const char* GetRendererName();
    
private:
    RendererFactory() = delete; // Static class
};