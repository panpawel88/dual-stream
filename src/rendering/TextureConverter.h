#pragma once

#include "RenderTexture.h"

// Forward declarations
struct DecodedFrame;
class IRenderer;

/**
 * Converts decoded frames to generic render textures.
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
     * Create a null texture.
     */
    static RenderTexture CreateNullTexture();
    
private:
    TextureConverter() = delete; // Static utility class
};