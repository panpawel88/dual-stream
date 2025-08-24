#pragma once

#include "RenderTexture.h"

// Forward declarations
struct DecodedFrame;
class IRenderer;

/**
 * Converts video decoder frames to generic render textures.
 * This class isolates video decoding concepts from the rendering system,
 * allowing renderers to work with generic texture abstractions.
 */
class TextureConverter {
public:
    /**
     * Convert a decoded frame to a render texture optimized for the given renderer.
     * The converter analyzes the renderer type and frame data to create the most
     * appropriate texture representation.
     */
    static RenderTexture ConvertFrame(const DecodedFrame& frame, IRenderer* renderer);
    
    /**
     * Create a null texture for black screen rendering.
     */
    static RenderTexture CreateNullTexture();
    
private:
    TextureConverter() = delete; // Static utility class
};