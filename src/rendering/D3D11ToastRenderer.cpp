#include "D3D11ToastRenderer.h"
#include "ui/ToastManager.h"
#include "core/Logger.h"
#include <algorithm>
#include <string>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

D3D11ToastRenderer::D3D11ToastRenderer(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dContext)
    : m_initialized(false)
    , m_d3dDevice(d3dDevice)
    , m_d3dContext(d3dContext)
    , m_viewportWidth(0)
    , m_viewportHeight(0) {
}

D3D11ToastRenderer::~D3D11ToastRenderer() {
    Cleanup();
}

bool D3D11ToastRenderer::Initialize(const ToastConfig& config) {
    if (m_initialized) {
        Cleanup();
    }
    
    m_config = config;
    UpdateViewportDimensions();
    
    if (!InitializeD2DResources()) {
        LOG_ERROR("Failed to initialize Direct2D resources for toast rendering");
        return false;
    }
    
    if (!CreateTextFormat()) {
        LOG_ERROR("Failed to create DirectWrite text format");
        return false;
    }
    
    if (!CreateBrushes()) {
        LOG_ERROR("Failed to create Direct2D brushes");
        return false;
    }
    
    m_initialized = true;
    LOG_DEBUG("D3D11ToastRenderer initialized successfully");
    return true;
}

void D3D11ToastRenderer::RenderToast(const ToastMessage& toast) {
    if (!m_initialized || !m_d2dRenderTarget || toast.currentAlpha <= 0.0f) {
        return;
    }
    
    // Update viewport dimensions in case of window resize
    UpdateViewportDimensions();
    
    // Calculate toast rectangle
    D2D1_RECT_F toastRect = CalculateToastRect(toast.text);
    
    // Begin Direct2D rendering
    m_d2dRenderTarget->BeginDraw();
    
    // Create rounded rectangle for background
    D2D1_ROUNDED_RECT roundedRect;
    roundedRect.rect = toastRect;
    roundedRect.radiusX = static_cast<FLOAT>(m_config.cornerRadius);
    roundedRect.radiusY = static_cast<FLOAT>(m_config.cornerRadius);
    
    // Update brush colors with current alpha
    D2D1_COLOR_F bgColor = ColorToD2D(m_config.backgroundColor, toast.currentAlpha);
    D2D1_COLOR_F textColor = ColorToD2D(m_config.textColor, toast.currentAlpha);
    
    m_backgroundBrush->SetColor(bgColor);
    m_textBrush->SetColor(textColor);
    
    // Draw background with rounded corners
    m_d2dRenderTarget->FillRoundedRectangle(roundedRect, m_backgroundBrush.Get());
    
    // Calculate text rectangle (inset by padding)
    D2D1_RECT_F textRect;
    textRect.left = toastRect.left + m_config.padding;
    textRect.top = toastRect.top + m_config.padding;
    textRect.right = toastRect.right - m_config.padding;
    textRect.bottom = toastRect.bottom - m_config.padding;
    
    // Convert string to wide string for DirectWrite
    std::wstring wtext(toast.text.begin(), toast.text.end());
    
    // Draw text
    m_d2dRenderTarget->DrawText(
        wtext.c_str(),
        static_cast<UINT32>(wtext.length()),
        m_textFormat.Get(),
        textRect,
        m_textBrush.Get()
    );
    
    // End Direct2D rendering
    HRESULT hr = m_d2dRenderTarget->EndDraw();
    if (FAILED(hr)) {
        LOG_ERROR("Direct2D EndDraw failed: ", hr);
    }
}

void D3D11ToastRenderer::Cleanup() {
    m_textBrush.Reset();
    m_backgroundBrush.Reset();
    m_textFormat.Reset();
    m_writeFactory.Reset();
    m_d2dRenderTarget.Reset();
    m_d2dFactory.Reset();
    
    m_initialized = false;
}

bool D3D11ToastRenderer::InitializeD2DResources() {
    // Create Direct2D factory
    HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, m_d2dFactory.GetAddressOf());
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Direct2D factory: ", hr);
        return false;
    }
    
    // Get DXGI surface from swap chain back buffer
    ComPtr<ID3D11Texture2D> backBuffer;
    ComPtr<IDXGISwapChain> swapChain;
    
    // We need to get the swap chain from the device context
    // This is a bit complex, but we'll create a render target from the current back buffer
    D3D11_VIEWPORT viewport;
    UINT numViewports = 1;
    m_d3dContext->RSGetViewports(&numViewports, &viewport);
    
    // Create a render target properties
    D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED),
        0.0f, 0.0f,
        D2D1_RENDER_TARGET_USAGE_NONE,
        D2D1_FEATURE_LEVEL_DEFAULT
    );
    
    // For now, we'll create a compatible render target
    // In a real implementation, we'd want to get the actual swap chain surface
    D2D1_SIZE_U size = D2D1::SizeU(
        static_cast<UINT32>(m_viewportWidth),
        static_cast<UINT32>(m_viewportHeight)
    );
    
    hr = m_d2dFactory->CreateHwndRenderTarget(
        rtProps,
        D2D1::HwndRenderTargetProperties(GetActiveWindow(), size),
        reinterpret_cast<ID2D1HwndRenderTarget**>(m_d2dRenderTarget.GetAddressOf())
    );
    
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create Direct2D render target: ", hr);
        return false;
    }
    
    return true;
}

bool D3D11ToastRenderer::CreateTextFormat() {
    // Create DirectWrite factory
    HRESULT hr = DWriteCreateFactory(
        DWRITE_FACTORY_TYPE_SHARED,
        __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(m_writeFactory.GetAddressOf())
    );
    
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create DirectWrite factory: ", hr);
        return false;
    }
    
    // Create text format
    hr = m_writeFactory->CreateTextFormat(
        L"Arial",                       // Font family
        nullptr,                        // Font collection (nullptr for system)
        DWRITE_FONT_WEIGHT_NORMAL,     // Font weight
        DWRITE_FONT_STYLE_NORMAL,      // Font style
        DWRITE_FONT_STRETCH_NORMAL,    // Font stretch
        static_cast<FLOAT>(m_config.fontSize),  // Font size
        L"en-us",                      // Locale
        m_textFormat.GetAddressOf()
    );
    
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create DirectWrite text format: ", hr);
        return false;
    }
    
    // Set text alignment
    m_textFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);
    m_textFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    
    return true;
}

bool D3D11ToastRenderer::CreateBrushes() {
    if (!m_d2dRenderTarget) {
        return false;
    }
    
    // Create text brush (will be updated with alpha in RenderToast)
    D2D1_COLOR_F textColor = ColorToD2D(m_config.textColor);
    HRESULT hr = m_d2dRenderTarget->CreateSolidColorBrush(textColor, m_textBrush.GetAddressOf());
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create text brush: ", hr);
        return false;
    }
    
    // Create background brush (will be updated with alpha in RenderToast)
    D2D1_COLOR_F bgColor = ColorToD2D(m_config.backgroundColor);
    hr = m_d2dRenderTarget->CreateSolidColorBrush(bgColor, m_backgroundBrush.GetAddressOf());
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create background brush: ", hr);
        return false;
    }
    
    return true;
}

D2D1_RECT_F D3D11ToastRenderer::CalculateToastRect(const std::string& text) {
    if (!m_textFormat) {
        return D2D1::RectF(0, 0, 0, 0);
    }
    
    // Convert to wide string
    std::wstring wtext(text.begin(), text.end());
    
    // Create text layout to measure text dimensions
    ComPtr<IDWriteTextLayout> textLayout;
    HRESULT hr = m_writeFactory->CreateTextLayout(
        wtext.c_str(),
        static_cast<UINT32>(wtext.length()),
        m_textFormat.Get(),
        static_cast<FLOAT>(m_config.maxWidth - 2 * m_config.padding),
        1000.0f,  // Max height
        textLayout.GetAddressOf()
    );
    
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create text layout for measurement: ", hr);
        // Return a default rectangle
        return D2D1::RectF(0, 0, 200, 40);
    }
    
    // Get text metrics
    DWRITE_TEXT_METRICS textMetrics;
    textLayout->GetMetrics(&textMetrics);
    
    // Calculate toast dimensions
    float toastWidth = std::min(textMetrics.width + 2 * m_config.padding, static_cast<float>(m_config.maxWidth));
    float toastHeight = textMetrics.height + 2 * m_config.padding;
    
    // Calculate position based on configuration
    float x = 0, y = 0;
    
    switch (m_config.position) {
        case ToastPosition::TOP_LEFT:
            x = static_cast<float>(m_config.offsetX);
            y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::TOP_CENTER:
            x = (m_viewportWidth - toastWidth) / 2 + m_config.offsetX;
            y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::TOP_RIGHT:
            x = m_viewportWidth - toastWidth - m_config.offsetX;
            y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::CENTER_LEFT:
            x = static_cast<float>(m_config.offsetX);
            y = (m_viewportHeight - toastHeight) / 2 + m_config.offsetY;
            break;
        case ToastPosition::CENTER:
            x = (m_viewportWidth - toastWidth) / 2 + m_config.offsetX;
            y = (m_viewportHeight - toastHeight) / 2 + m_config.offsetY;
            break;
        case ToastPosition::CENTER_RIGHT:
            x = m_viewportWidth - toastWidth - m_config.offsetX;
            y = (m_viewportHeight - toastHeight) / 2 + m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_LEFT:
            x = static_cast<float>(m_config.offsetX);
            y = m_viewportHeight - toastHeight - m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_CENTER:
            x = (m_viewportWidth - toastWidth) / 2 + m_config.offsetX;
            y = m_viewportHeight - toastHeight - m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_RIGHT:
            x = m_viewportWidth - toastWidth - m_config.offsetX;
            y = m_viewportHeight - toastHeight - m_config.offsetY;
            break;
    }
    
    return D2D1::RectF(x, y, x + toastWidth, y + toastHeight);
}

void D3D11ToastRenderer::UpdateViewportDimensions() {
    D3D11_VIEWPORT viewport;
    UINT numViewports = 1;
    m_d3dContext->RSGetViewports(&numViewports, &viewport);
    
    m_viewportWidth = static_cast<int>(viewport.Width);
    m_viewportHeight = static_cast<int>(viewport.Height);
}

D2D1_COLOR_F D3D11ToastRenderer::ColorToD2D(const ToastConfig::Color& color, float alpha) {
    return D2D1::ColorF(
        color.r / 255.0f,
        color.g / 255.0f,
        color.b / 255.0f,
        (color.a / 255.0f) * alpha
    );
}