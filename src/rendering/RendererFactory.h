#pragma once

#include <memory>
#include <string>
#include "IRenderer.h"

/**
 * Backend preference enumeration
 */
enum class RendererBackend {
    Auto,       // Automatically select best available backend
    DirectX11,  // Force DirectX 11 renderer
    OpenGL      // Force OpenGL renderer
};

/**
 * Factory class for creating renderer instances.
 * Supports runtime backend selection based on configuration.
 */
class RendererFactory {
public:
    /**
     * Creates the appropriate renderer based on runtime configuration.
     * Falls back to compile-time default if preferred backend fails.
     */
    static std::unique_ptr<IRenderer> CreateRenderer(RendererBackend preference = RendererBackend::Auto);
    
    /**
     * Gets the name of the specified renderer backend.
     */
    static const char* GetRendererName(RendererBackend backend);
    
    /**
     * Gets the name of the default compile-time renderer.
     */
    static const char* GetDefaultRendererName();
    
    /**
     * Parse backend string from configuration file.
     */
    static RendererBackend ParseBackendString(const std::string& backendStr);
    
private:
    RendererFactory() = delete; // Static class
};