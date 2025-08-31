#pragma once

#include "../RenderPassContext.h"
#include <glad/gl.h>

/**
 * OpenGL specific render pass context
 * This file can safely include OpenGL headers since it's only used by OpenGL code
 */
struct OpenGLRenderPassContext : public RenderPassContextBase {
    GLuint uvTexture;           // Second texture for YUV formats (0 if not used)
    GLenum textureFormat;       // GL_RGBA8, GL_RG8, etc.
    GLenum textureInternalFormat;  // Internal format for texture creation
    GLenum textureDataFormat;   // GL_RGBA, GL_RG, etc.
    GLenum textureDataType;     // GL_UNSIGNED_BYTE, etc.
};