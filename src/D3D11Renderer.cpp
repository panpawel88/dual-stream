#include "D3D11Renderer.h"
#include "Logger.h"
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

// Simple pixel shader source
const char* g_pixelShaderSource = R"(
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
        std::cerr << "Failed to create D3D11 device and swap chain\n";
        Cleanup();
        return false;
    }
    
    // Create render target
    if (!CreateRenderTarget()) {
        std::cerr << "Failed to create render target\n";
        Cleanup();
        return false;
    }
    
    // Create shaders
    if (!CreateShaders()) {
        std::cerr << "Failed to create shaders\n";
        Cleanup();
        return false;
    }
    
    // Create geometry
    if (!CreateGeometry()) {
        std::cerr << "Failed to create geometry\n";
        Cleanup();
        return false;
    }
    
    // Create states
    if (!CreateStates()) {
        std::cerr << "Failed to create render states\n";
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

bool D3D11Renderer::Present(ID3D11Texture2D* videoTexture) {
    if (!m_initialized) {
        return false;
    }
    
    // Update frame texture if provided
    if (videoTexture) {
        if (!UpdateFrameTexture(videoTexture)) {
            return false;
        }
    }
    
    // Clear render target
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_context->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
    
    // Only draw if we have a texture to render
    if (m_currentFrameSRV) {
        // Setup render state
        SetupRenderState();
        
        // Draw fullscreen quad
        DrawQuad();
    }
    
    // Present
    HRESULT hr = m_swapChain->Present(1, 0); // VSync enabled
    if (FAILED(hr)) {
        std::cerr << "Failed to present frame. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    return true;
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
        std::cerr << "Failed to resize swap chain buffers. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    // Recreate render target
    if (!CreateRenderTarget()) {
        std::cerr << "Failed to recreate render target after resize\n";
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
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
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
        std::cerr << "Failed to create D3D11 device and swap chain. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    LOG_INFO("Created D3D11 device with feature level: ", std::hex, featureLevel);
    return true;
}

bool D3D11Renderer::CreateRenderTarget() {
    // Get back buffer
    HRESULT hr = m_swapChain->GetBuffer(0, IID_PPV_ARGS(&m_backBuffer));
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    // Create render target view
    hr = m_device->CreateRenderTargetView(m_backBuffer.Get(), nullptr, &m_renderTargetView);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view. HRESULT: 0x" << std::hex << hr << "\n";
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
            std::cerr << "Vertex shader compilation failed: " << (char*)errorBlob->GetBufferPointer() << "\n";
        }
        return false;
    }
    
    // Create vertex shader
    hr = m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
    if (FAILED(hr)) {
        std::cerr << "Failed to create vertex shader. HRESULT: 0x" << std::hex << hr << "\n";
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
        std::cerr << "Failed to create input layout. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    // Compile pixel shader
    ComPtr<ID3DBlob> psBlob;
    hr = D3DCompile(g_pixelShaderSource, strlen(g_pixelShaderSource), nullptr, nullptr, nullptr,
                   "main", "ps_4_0", 0, 0, &psBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Pixel shader compilation failed: " << (char*)errorBlob->GetBufferPointer() << "\n";
        }
        return false;
    }
    
    // Create pixel shader
    hr = m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader);
    if (FAILED(hr)) {
        std::cerr << "Failed to create pixel shader. HRESULT: 0x" << std::hex << hr << "\n";
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
        std::cerr << "Failed to create vertex buffer. HRESULT: 0x" << std::hex << hr << "\n";
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
        std::cerr << "Failed to create index buffer. HRESULT: 0x" << std::hex << hr << "\n";
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
        std::cerr << "Failed to create sampler state. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    // Create blend state (no blending)
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = FALSE;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    hr = m_device->CreateBlendState(&blendDesc, &m_blendState);
    if (FAILED(hr)) {
        std::cerr << "Failed to create blend state. HRESULT: 0x" << std::hex << hr << "\n";
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
        std::cerr << "Failed to create rasterizer state. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    return true;
}

bool D3D11Renderer::UpdateFrameTexture(ID3D11Texture2D* videoTexture) {
    if (!videoTexture) {
        return false;
    }
    
    // Create shader resource view for the video texture
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; // Match texture format
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    
    m_currentFrameSRV.Reset(); // Release previous SRV
    
    HRESULT hr = m_device->CreateShaderResourceView(videoTexture, &srvDesc, &m_currentFrameSRV);
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view for video texture. HRESULT: 0x" << std::hex << hr << "\n";
        return false;
    }
    
    return true;
}

void D3D11Renderer::SetupRenderState() {
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
    m_context->PSSetShader(m_pixelShader.Get(), nullptr, 0);
    
    // Set texture and sampler
    if (m_currentFrameSRV) {
        m_context->PSSetShaderResources(0, 1, m_currentFrameSRV.GetAddressOf());
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

void D3D11Renderer::Reset() {
    m_initialized = false;
    
    // Reset all COM objects
    m_currentFrameSRV.Reset();
    m_rasterizerState.Reset();
    m_blendState.Reset();
    m_samplerState.Reset();
    m_indexBuffer.Reset();
    m_vertexBuffer.Reset();
    m_inputLayout.Reset();
    m_pixelShader.Reset();
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