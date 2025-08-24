#include "D3D11Renderer.h"
#include "core/Logger.h"
#include <iostream>
#include <d3dcompiler.h>

// Simple vertex shader source
const char* g_vertexShaderSource = R"(
struct VS_INPUT {
    float3 pos : POSITION;
    float2 tex : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.pos = float4(input.pos, 1.0f);
    output.tex = input.tex;
    return output;
}
)";

// RGB pixel shader source
const char* g_pixelShaderRGBSource = R"(
Texture2D videoTexture : register(t0);
SamplerState videoSampler : register(s0);

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    return videoTexture.Sample(videoSampler, input.tex);
}
)";

// NV12 YUV to RGB conversion with separate Y and UV textures
const char* g_pixelShaderYUVSource = R"(
Texture2D yTexture : register(t0);   // Y plane
Texture2D uvTexture : register(t1);  // UV plane
SamplerState videoSampler : register(s0);

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    // Sample Y (luminance)
    float y = yTexture.Sample(videoSampler, input.tex).r;
    
    // Sample UV (chrominance) - for NV12, UV is at half resolution
    float2 chroma = uvTexture.Sample(videoSampler, input.tex).rg;
    float u = chroma.r - 0.5;
    float v = chroma.g - 0.5;
    
    // BT.709 YUV to RGB conversion (full range)
    float r = y + 1.402 * v;
    float g = y - 0.344 * u - 0.714 * v;
    float b = y + 1.772 * u;
    
    return float4(saturate(r), saturate(g), saturate(b), 1.0);
}
)";

D3D11Renderer::D3D11Renderer()
    : m_initialized(false)
    , m_hwnd(nullptr)
    , m_width(0)
    , m_height(0) {
}

D3D11Renderer::~D3D11Renderer() {
    Cleanup();
}

bool D3D11Renderer::Initialize(HWND hwnd, int width, int height) {
    if (m_initialized) {
        Cleanup();
    }
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    
    LOG_INFO("Initializing D3D11 renderer (", width, "x", height, ")");
    
    // Create device and swap chain
    if (!CreateDeviceAndSwapChain()) {
        LOG_ERROR("Failed to create D3D11 device and swap chain");
        Cleanup();
        return false;
    }
    
    // Create render target
    if (!CreateRenderTarget()) {
        LOG_ERROR("Failed to create render target");
        Cleanup();
        return false;
    }
    
    // Create shaders
    if (!CreateShaders()) {
        LOG_ERROR("Failed to create shaders");
        Cleanup();
        return false;
    }
    
    // Create geometry
    if (!CreateGeometry()) {
        LOG_ERROR("Failed to create geometry");
        Cleanup();
        return false;
    }
    
    // Create states
    if (!CreateStates()) {
        LOG_ERROR("Failed to create render states");
        Cleanup();
        return false;
    }
    
    // Set viewport
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    m_context->RSSetViewports(1, &viewport);
    
    m_initialized = true;
    LOG_INFO("D3D11 renderer initialized successfully");
    return true;
}

void D3D11Renderer::Cleanup() {
    Reset();
}

bool D3D11Renderer::Present(const RenderTexture& texture) {
    if (!m_initialized) {
        return false;
    }
    
    // Clear render target
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_context->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
    
    // Handle different texture types
    bool renderSuccess = false;
    
    if (!texture.IsValid()) {
        // Present black screen for invalid texture
        renderSuccess = true;
    } else {
        switch (texture.type) {
            case TextureType::D3D11:
                renderSuccess = PresentD3D11Texture(texture);
                break;
                
            case TextureType::Software:
                renderSuccess = PresentSoftwareTexture(texture);
                break;
                
            case TextureType::CUDA:
                LOG_WARNING("CUDA texture not supported in D3D11 renderer");
                renderSuccess = false;
                break;
                
            case TextureType::OpenGL:
                LOG_WARNING("OpenGL texture not supported in D3D11 renderer");
                renderSuccess = false;
                break;
                
            default:
                LOG_ERROR("Unknown texture type");
                renderSuccess = false;
                break;
        }
    }
    
    // Always present, even for failed renders (shows black screen)
    HRESULT hr = m_swapChain->Present(1, 0); // VSync enabled
    if (FAILED(hr)) {
        LOG_ERROR("Failed to present frame. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    return renderSuccess;
}

bool D3D11Renderer::Resize(int width, int height) {
    if (!m_initialized || (width == m_width && height == m_height)) {
        return true;
    }
    
    m_width = width;
    m_height = height;
    
    // Release render target view and back buffer
    m_renderTargetView.Reset();
    m_backBuffer.Reset();
    
    // Resize swap chain
    HRESULT hr = m_swapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to resize swap chain buffers. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Recreate render target
    if (!CreateRenderTarget()) {
        LOG_ERROR("Failed to recreate render target after resize");
        return false;
    }
    
    // Update viewport
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(width);
    viewport.Height = static_cast<float>(height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    m_context->RSSetViewports(1, &viewport);
    
    return true;
}

bool D3D11Renderer::CreateDeviceAndSwapChain() {
    // Create swap chain description
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = m_width;
    swapChainDesc.BufferDesc.Height = m_height;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
    swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = m_hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.Windowed = TRUE;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    
    // Create device, context and swap chain
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,                    // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,   // Hardware acceleration
        nullptr,                    // Software rasterizer
        0,                         // Debug flags
        featureLevels,             // Feature levels
        ARRAYSIZE(featureLevels),  // Feature level count
        D3D11_SDK_VERSION,         // SDK version
        &swapChainDesc,            // Swap chain description
        &m_swapChain,              // Output swap chain
        &m_device,                 // Output device
        &featureLevel,             // Output feature level
        &m_context                 // Output context
    );
    
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create D3D11 device and swap chain. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    LOG_INFO("Created D3D11 device with feature level: ", std::hex, featureLevel);
    return true;
}

bool D3D11Renderer::CreateRenderTarget() {
    // Get back buffer
    HRESULT hr = m_swapChain->GetBuffer(0, IID_PPV_ARGS(&m_backBuffer));
    if (FAILED(hr)) {
        LOG_ERROR("Failed to get back buffer. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Create render target view
    hr = m_device->CreateRenderTargetView(m_backBuffer.Get(), nullptr, &m_renderTargetView);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create render target view. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Set render target
    m_context->OMSetRenderTargets(1, m_renderTargetView.GetAddressOf(), nullptr);
    
    return true;
}

bool D3D11Renderer::CreateShaders() {
    HRESULT hr;
    
    // Compile vertex shader
    ComPtr<ID3DBlob> vsBlob;
    ComPtr<ID3DBlob> errorBlob;
    
    hr = D3DCompile(g_vertexShaderSource, strlen(g_vertexShaderSource), nullptr, nullptr, nullptr,
                   "main", "vs_4_0", 0, 0, &vsBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG_ERROR("Vertex shader compilation failed: ", (char*)errorBlob->GetBufferPointer());
        }
        return false;
    }
    
    // Create vertex shader
    hr = m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create vertex shader. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Create input layout
    D3D11_INPUT_ELEMENT_DESC inputDesc[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };
    
    hr = m_device->CreateInputLayout(inputDesc, ARRAYSIZE(inputDesc),
                                    vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create input layout. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Compile RGB pixel shader
    ComPtr<ID3DBlob> psRGBBlob;
    hr = D3DCompile(g_pixelShaderRGBSource, strlen(g_pixelShaderRGBSource), nullptr, nullptr, nullptr,
                   "main", "ps_4_0", 0, 0, &psRGBBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG_ERROR("RGB pixel shader compilation failed: ", (char*)errorBlob->GetBufferPointer());
        }
        return false;
    }
    
    // Create RGB pixel shader
    hr = m_device->CreatePixelShader(psRGBBlob->GetBufferPointer(), psRGBBlob->GetBufferSize(), nullptr, &m_pixelShaderRGB);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create RGB pixel shader. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Compile YUV pixel shader
    ComPtr<ID3DBlob> psYUVBlob;
    hr = D3DCompile(g_pixelShaderYUVSource, strlen(g_pixelShaderYUVSource), nullptr, nullptr, nullptr,
                   "main", "ps_4_0", 0, 0, &psYUVBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG_ERROR("YUV pixel shader compilation failed: ", (char*)errorBlob->GetBufferPointer());
        }
        return false;
    }
    
    // Create YUV pixel shader
    hr = m_device->CreatePixelShader(psYUVBlob->GetBufferPointer(), psYUVBlob->GetBufferSize(), nullptr, &m_pixelShaderYUV);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create YUV pixel shader. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    return true;
}

bool D3D11Renderer::CreateGeometry() {
    // Create fullscreen quad vertices
    QuadVertex vertices[] = {
        // Position (x, y, z)    // TexCoord (u, v)
        { {-1.0f,  1.0f, 0.0f}, {0.0f, 0.0f} }, // Top-left
        { { 1.0f,  1.0f, 0.0f}, {1.0f, 0.0f} }, // Top-right
        { { 1.0f, -1.0f, 0.0f}, {1.0f, 1.0f} }, // Bottom-right
        { {-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f} }  // Bottom-left
    };
    
    // Create vertex buffer
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(vertices);
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = vertices;
    
    HRESULT hr = m_device->CreateBuffer(&bufferDesc, &initData, &m_vertexBuffer);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create vertex buffer. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Create indices
    UINT indices[] = { 0, 1, 2, 0, 2, 3 };
    
    // Create index buffer
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(indices);
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    
    initData.pSysMem = indices;
    
    hr = m_device->CreateBuffer(&bufferDesc, &initData, &m_indexBuffer);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create index buffer. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    return true;
}

bool D3D11Renderer::CreateStates() {
    HRESULT hr;
    
    // Create sampler state
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    
    hr = m_device->CreateSamplerState(&samplerDesc, &m_samplerState);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create sampler state. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Create blend state (no blending)
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = FALSE;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    hr = m_device->CreateBlendState(&blendDesc, &m_blendState);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create blend state. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    // Create rasterizer state
    D3D11_RASTERIZER_DESC rasterizerDesc = {};
    rasterizerDesc.FillMode = D3D11_FILL_SOLID;
    rasterizerDesc.CullMode = D3D11_CULL_BACK;
    rasterizerDesc.FrontCounterClockwise = FALSE;
    rasterizerDesc.DepthClipEnable = TRUE;
    
    hr = m_device->CreateRasterizerState(&rasterizerDesc, &m_rasterizerState);
    if (FAILED(hr)) {
        LOG_ERROR("Failed to create rasterizer state. HRESULT: 0x", std::hex, hr);
        return false;
    }
    
    return true;
}

bool D3D11Renderer::UpdateFrameTexture(ID3D11Texture2D* videoTexture, bool isYUV, DXGI_FORMAT format) {
    if (!videoTexture) {
        return false;
    }
    
    // Get the actual texture format
    D3D11_TEXTURE2D_DESC textureDesc;
    videoTexture->GetDesc(&textureDesc);
    
    LOG_DEBUG("UpdateFrameTexture - Texture format: ", textureDesc.Format, ", isYUV: ", isYUV);
    
    // Create shader resource view for the video texture
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    
    // Handle different texture formats appropriately
    switch (textureDesc.Format) {
        case DXGI_FORMAT_NV12: {
            // For NV12, create Y plane view first
            srvDesc.Format = DXGI_FORMAT_R8_UNORM; // Y plane
            LOG_DEBUG("Creating Y plane SRV for NV12");
            
            HRESULT hr = m_device->CreateShaderResourceView(videoTexture, &srvDesc, &m_currentFrameSRV);
            if (FAILED(hr)) {
                LOG_DEBUG("Failed to create Y plane SRV. HRESULT: 0x", std::hex, hr);
                return false;
            }
            
            // Create UV plane view - for NV12, UV is at half resolution and different offset
            D3D11_SHADER_RESOURCE_VIEW_DESC uvDesc = {};
            uvDesc.Format = DXGI_FORMAT_R8G8_UNORM; // UV interleaved
            uvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            uvDesc.Texture2D.MipLevels = 1;
            uvDesc.Texture2D.MostDetailedMip = 0;
            
            // Try to create a view for the UV data
            hr = m_device->CreateShaderResourceView(videoTexture, &uvDesc, &m_currentFrameUVSRV);
            if (FAILED(hr)) {
                LOG_DEBUG("Failed to create UV plane SRV, will use Y-only. HRESULT: 0x", std::hex, hr);
                m_currentFrameUVSRV.Reset();
            } else {
                LOG_DEBUG("UV plane SRV created successfully");
            }
            
            return true;
        }
        case DXGI_FORMAT_420_OPAQUE:
            // 420_OPAQUE can't be used directly as SRV, need to convert
            LOG_DEBUG("420_OPAQUE format not supported for direct SRV creation");
            return false;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM:
            // RGB formats - use as-is
            srvDesc.Format = textureDesc.Format;
            LOG_DEBUG("Creating SRV for RGB texture with original format");
            break;
        default:
            // Try using the original format
            srvDesc.Format = textureDesc.Format;
            LOG_DEBUG("Creating SRV with original format: ", textureDesc.Format);
            break;
    }
    
    // For non-NV12 formats, create single SRV
    m_currentFrameSRV.Reset(); // Release previous SRV
    m_currentFrameUVSRV.Reset(); // Release UV SRV
    
    HRESULT hr = m_device->CreateShaderResourceView(videoTexture, &srvDesc, &m_currentFrameSRV);
    if (FAILED(hr)) {
        LOG_DEBUG("Failed to create SRV with format ", srvDesc.Format, ", trying nullptr descriptor");
        // Try with nullptr descriptor to let D3D11 auto-determine
        hr = m_device->CreateShaderResourceView(videoTexture, nullptr, &m_currentFrameSRV);
        if (FAILED(hr)) {
            LOG_ERROR("Failed to create shader resource view for video texture. HRESULT: 0x", std::hex, hr);
            LOG_ERROR("Texture format: ", textureDesc.Format, ", SRV format: ", srvDesc.Format);
            return false;
        }
        LOG_DEBUG("SRV created successfully with nullptr descriptor");
    } else {
        LOG_DEBUG("SRV created successfully with explicit format");
    }
    
    return true;
}

void D3D11Renderer::SetupRenderState(bool isYUV) {
    // Set vertex buffer
    UINT stride = sizeof(QuadVertex);
    UINT offset = 0;
    m_context->IASetVertexBuffers(0, 1, m_vertexBuffer.GetAddressOf(), &stride, &offset);
    
    // Set index buffer
    m_context->IASetIndexBuffer(m_indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    
    // Set input layout
    m_context->IASetInputLayout(m_inputLayout.Get());
    
    // Set primitive topology
    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // Set shaders
    m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
    
    if (isYUV) {
        m_context->PSSetShader(m_pixelShaderYUV.Get(), nullptr, 0);
        LOG_DEBUG("Using YUV pixel shader for YUV texture");
    } else {
        m_context->PSSetShader(m_pixelShaderRGB.Get(), nullptr, 0);
        LOG_DEBUG("Using RGB pixel shader for RGB texture");
    }
    
    // Set texture and sampler
    if (m_currentFrameSRV) {
        m_context->PSSetShaderResources(0, 1, m_currentFrameSRV.GetAddressOf());
        
        // Bind UV texture if available (for NV12)
        if (m_currentFrameUVSRV) {
            m_context->PSSetShaderResources(1, 1, m_currentFrameUVSRV.GetAddressOf());
            LOG_DEBUG("Bound both Y and UV textures for NV12 rendering");
        } else {
            LOG_DEBUG("Only Y texture bound, UV not available");
        }
    }
    m_context->PSSetSamplers(0, 1, m_samplerState.GetAddressOf());
    
    // Set blend state
    float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    m_context->OMSetBlendState(m_blendState.Get(), blendFactor, 0xFFFFFFFF);
    
    // Set rasterizer state
    m_context->RSSetState(m_rasterizerState.Get());
}

void D3D11Renderer::DrawQuad() {
    m_context->DrawIndexed(6, 0, 0);
}

bool D3D11Renderer::PresentD3D11Texture(const RenderTexture& texture) {
    if (texture.type != TextureType::D3D11 || !texture.d3d11.texture) {
        return false;
    }
    
    // Update frame texture
    if (!UpdateFrameTexture(texture.d3d11.texture.Get(), texture.isYUV, texture.d3d11.dxgiFormat)) {
        return false;
    }
    
    // Only draw if we have a texture to render
    if (m_currentFrameSRV) {
        // Setup render state
        SetupRenderState(texture.isYUV);
        
        // Draw fullscreen quad
        DrawQuad();
    }
    
    return true;
}

bool D3D11Renderer::PresentSoftwareTexture(const RenderTexture& texture) {
    if (texture.type != TextureType::Software || !texture.software.data) {
        return false;
    }
    
    // TODO: Implement software texture upload to D3D11 texture
    // This would require creating a D3D11 texture from the CPU data
    // For now, return false to indicate unsupported
    LOG_WARNING("Software texture presentation not yet implemented for D3D11 renderer");
    return false;
}

void D3D11Renderer::Reset() {
    m_initialized = false;
    
    // Reset all COM objects
    m_currentFrameUVSRV.Reset();
    m_currentFrameSRV.Reset();
    m_rasterizerState.Reset();
    m_blendState.Reset();
    m_samplerState.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();
    m_inputLayout.Reset();
    m_pixelShaderYUV.Reset();
    m_pixelShaderRGB.Reset();
    m_vertexShader.Reset();
    m_renderTargetView.Reset();
    m_backBuffer.Reset();
    m_swapChain.Reset();
    m_context.Reset();
    m_device.Reset();
    
    m_hwnd = nullptr;
    m_width = 0;
    m_height = 0;
}