#pragma once

#include <cstdint>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

/**
 * Generic texture format enumeration
 */
enum class TextureFormat {
    RGBA8,      // 8-bit RGBA
    BGRA8,      // 8-bit BGRA  
    NV12,       // YUV NV12 format
    YUV420P     // YUV 420 planar
};

/**
 * Texture data types supported by renderers
 */
enum class TextureType {
    Software,   // CPU memory (uint8_t* data)
    D3D11,      // DirectX 11 texture
    OpenGL,     // OpenGL texture ID (managed internally)
    CUDA        // CUDA device memory for interop
};

/**
 * Generic texture abstraction that can represent textures from different graphics APIs.
 */
struct RenderTexture {
    TextureType type;
    TextureFormat format;
    int width;
    int height;
    bool isYUV; // Indicates if YUV->RGB conversion is needed in shader
    
    // Software texture data
    struct {
        const uint8_t* data;
        int pitch; // bytes per row
    } software;
    
    // D3D11 texture data
    struct {
        ComPtr<ID3D11Texture2D> texture;
        DXGI_FORMAT dxgiFormat;
    } d3d11;
    
    // CUDA device memory (for OpenGL interop)
    struct {
        void* devicePtr;
        size_t pitch;
        void* glResource; // OpenGL interop resource handle (opaque)
    } cuda;
    
    // OpenGL texture ID (managed by OpenGL renderer internally)
    struct {
        unsigned int textureId;
    } opengl;
    
    RenderTexture() 
        : type(TextureType::Software)
        , format(TextureFormat::RGBA8)
        , width(0)
        , height(0)
        , isYUV(false)
    {
        software.data = nullptr;
        software.pitch = 0;
        cuda.devicePtr = nullptr;
        cuda.pitch = 0;
        cuda.glResource = nullptr;
        opengl.textureId = 0;
    }
    
    // Check if texture has valid data
    bool IsValid() const {
        switch (type) {
            case TextureType::Software:
                return software.data != nullptr && width > 0 && height > 0 && software.pitch > 0;
            case TextureType::D3D11:
                return d3d11.texture != nullptr;
            case TextureType::CUDA:
                return cuda.devicePtr != nullptr && width > 0 && height > 0;
            case TextureType::OpenGL:
                return opengl.textureId != 0 && width > 0 && height > 0;
        }
        return false;
    }
    
    // Create a null/empty texture
    static RenderTexture CreateNull() {
        RenderTexture nullTexture;
        nullTexture.type = TextureType::Software;
        nullTexture.width = 0;
        nullTexture.height = 0;
        nullTexture.software.data = nullptr;
        nullTexture.software.pitch = 0;
        return nullTexture;
    }
};