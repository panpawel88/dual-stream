#pragma once

#include "../CameraFrame.h"
#include "../../rendering/IRenderer.h"
#include <memory>
#include <mutex>

// Forward declarations for platform-specific types
#ifdef _WIN32
#include <d3d11.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

// Include OpenGL headers (always available)
#include "../../rendering/OpenGLHeaders.h"

/**
 * Camera frame to texture converter for ImGui display.
 * Handles both DirectX 11 and OpenGL backends with caching.
 */
class CameraFrameTexture {
public:
    CameraFrameTexture();
    ~CameraFrameTexture();

    /**
     * Initialize with current renderer type.
     * @param renderer Current renderer instance
     * @return true if initialization successful
     */
    bool Initialize(IRenderer* renderer);

    /**
     * Update texture with new camera frame.
     * @param frame Camera frame to display
     * @return true if texture updated successfully
     */
    bool UpdateTexture(std::shared_ptr<const CameraFrame> frame);

    /**
     * Get ImGui texture ID for rendering.
     * @return ImTextureID for ImGui::Image()
     */
    void* GetImGuiTextureID() const;

    /**
     * Get current texture dimensions.
     * @param width Output width
     * @param height Output height
     */
    void GetTextureDimensions(int& width, int& height) const;

    /**
     * Check if texture is valid for rendering.
     * @return true if texture is ready for display
     */
    bool IsValid() const;

    /**
     * Set maximum texture dimensions for preview.
     * @param maxWidth Maximum width (default: 640)
     * @param maxHeight Maximum height (default: 480)
     */
    void SetMaxDimensions(int maxWidth, int maxHeight);

    /**
     * Clean up resources.
     */
    void Cleanup();

private:
    IRenderer* m_renderer;
    RendererType m_rendererType;

    // Texture dimensions
    int m_textureWidth;
    int m_textureHeight;
    int m_maxWidth;
    int m_maxHeight;

    // Thread safety
    mutable std::mutex m_textureMutex;

    // DirectX 11 resources
#ifdef _WIN32
    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    ComPtr<ID3D11Texture2D> m_d3dTexture;
    ComPtr<ID3D11ShaderResourceView> m_d3dSRV;
#endif

    // OpenGL resources
    unsigned int m_glTexture;

    // Helper methods
    bool InitializeD3D11();
    bool InitializeOpenGL();
    bool UpdateD3D11Texture(const CameraFrame& frame);
    bool UpdateOpenGLTexture(const CameraFrame& frame);
    void CleanupD3D11Resources();
    void CleanupOpenGLResources();

    // Utility methods
    void CalculateScaledDimensions(int srcWidth, int srcHeight, int& dstWidth, int& dstHeight);
    bool ConvertFrameToRGBA(const CameraFrame& frame, std::vector<uint8_t>& rgbaData, int& width, int& height);
};