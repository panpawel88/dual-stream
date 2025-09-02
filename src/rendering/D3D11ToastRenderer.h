#pragma once

#include "IToastRenderer.h"
#include "ui/ToastManager.h"
#include <d3d11.h>
#include <d2d1.h>
#include <dwrite.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

/**
 * DirectX 11 implementation of toast rendering using Direct2D/DirectWrite.
 * Renders toast notifications as overlay on top of the video content.
 */
class D3D11ToastRenderer : public IToastRenderer {
public:
    D3D11ToastRenderer(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dContext);
    ~D3D11ToastRenderer();
    
    // IToastRenderer interface
    bool Initialize(const ToastConfig& config) override;
    void RenderToast(const ToastMessage& toast) override;
    void Cleanup() override;
    bool IsInitialized() const override { return m_initialized; }

private:
    bool m_initialized;
    
    // D3D11 resources (references, not owned)
    ID3D11Device* m_d3dDevice;
    ID3D11DeviceContext* m_d3dContext;
    
    // Direct2D/DirectWrite resources
    ComPtr<ID2D1Factory> m_d2dFactory;
    ComPtr<ID2D1RenderTarget> m_d2dRenderTarget;
    ComPtr<IDWriteFactory> m_writeFactory;
    ComPtr<IDWriteTextFormat> m_textFormat;
    
    // Brushes for rendering
    ComPtr<ID2D1SolidColorBrush> m_textBrush;
    ComPtr<ID2D1SolidColorBrush> m_backgroundBrush;
    
    // Configuration
    ToastConfig m_config;
    
    // Viewport dimensions (for positioning)
    int m_viewportWidth;
    int m_viewportHeight;
    
    /**
     * Initialize Direct2D and DirectWrite resources
     */
    bool InitializeD2DResources();
    
    /**
     * Create text format with the configured font settings
     */
    bool CreateTextFormat();
    
    /**
     * Create brushes for text and background rendering
     */
    bool CreateBrushes();
    
    /**
     * Calculate toast position based on configuration and text size
     */
    D2D1_RECT_F CalculateToastRect(const std::string& text);
    
    /**
     * Get viewport dimensions from D3D11 context
     */
    void UpdateViewportDimensions();
    
    /**
     * Convert ToastConfig::Color to D2D1_COLOR_F
     */
    D2D1_COLOR_F ColorToD2D(const ToastConfig::Color& color, float alpha = 1.0f);
};