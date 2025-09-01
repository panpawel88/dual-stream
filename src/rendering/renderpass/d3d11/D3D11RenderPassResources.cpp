#include "D3D11RenderPassResources.h"
#include "core/Logger.h"

D3D11RenderPassResources& D3D11RenderPassResources::GetInstance() {
    static D3D11RenderPassResources instance;
    return instance;
}

bool D3D11RenderPassResources::Initialize(ID3D11Device* device) {
    if (m_initialized) {
        return true; // Already initialized
    }
    
    if (!CreateFullscreenQuad(device)) {
        LOG_ERROR("D3D11RenderPassResources: Failed to create fullscreen quad");
        return false;
    }
    
    if (!CreateSamplerStates(device)) {
        LOG_ERROR("D3D11RenderPassResources: Failed to create sampler states");
        return false;
    }
    
    if (!CreateBlendStates(device)) {
        LOG_ERROR("D3D11RenderPassResources: Failed to create blend states");
        return false;
    }
    
    if (!CreateRasterizerStates(device)) {
        LOG_ERROR("D3D11RenderPassResources: Failed to create rasterizer states");
        return false;
    }
    
    m_initialized = true;
    LOG_INFO("D3D11RenderPassResources initialized successfully");
    return true;
}

void D3D11RenderPassResources::Cleanup() {
    m_backCullRasterizer.Reset();
    m_noCullRasterizer.Reset();
    m_additiveBlendState.Reset();
    m_alphaBlendState.Reset();
    m_noBlendState.Reset();
    m_pointClampSampler.Reset();
    m_linearClampSampler.Reset();
    m_fullscreenIndexBuffer.Reset();
    m_fullscreenVertexBuffer.Reset();
    
    m_initialized = false;
}

bool D3D11RenderPassResources::CreateFullscreenQuad(ID3D11Device* device) {
    // Fullscreen quad vertices (position + texture coordinates)
    float vertices[] = {
        // Position (x, y, z)    // TexCoord (u, v)
        -1.0f, -1.0f, 0.0f,      0.0f, 1.0f,  // Bottom-left
         1.0f, -1.0f, 0.0f,      1.0f, 1.0f,  // Bottom-right
         1.0f,  1.0f, 0.0f,      1.0f, 0.0f,  // Top-right
        -1.0f,  1.0f, 0.0f,      0.0f, 0.0f   // Top-left
    };
    
    D3D11_BUFFER_DESC vertexDesc = {};
    vertexDesc.ByteWidth = sizeof(vertices);
    vertexDesc.Usage = D3D11_USAGE_IMMUTABLE;
    vertexDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA vertexData = {};
    vertexData.pSysMem = vertices;
    
    if (FAILED(device->CreateBuffer(&vertexDesc, &vertexData, &m_fullscreenVertexBuffer))) {
        return false;
    }
    
    // Index buffer for quad (2 triangles)
    uint16_t indices[] = { 0, 1, 2, 0, 2, 3 };
    
    D3D11_BUFFER_DESC indexDesc = {};
    indexDesc.ByteWidth = sizeof(indices);
    indexDesc.Usage = D3D11_USAGE_IMMUTABLE;
    indexDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA indexData = {};
    indexData.pSysMem = indices;
    
    return SUCCEEDED(device->CreateBuffer(&indexDesc, &indexData, &m_fullscreenIndexBuffer));
}

bool D3D11RenderPassResources::CreateSamplerStates(ID3D11Device* device) {
    // Linear clamp sampler (most common for render passes)
    D3D11_SAMPLER_DESC linearDesc = {};
    linearDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    linearDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    linearDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    linearDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    linearDesc.MipLODBias = 0.0f;
    linearDesc.MaxAnisotropy = 1;
    linearDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    linearDesc.MinLOD = 0;
    linearDesc.MaxLOD = D3D11_FLOAT32_MAX;
    
    if (FAILED(device->CreateSamplerState(&linearDesc, &m_linearClampSampler))) {
        return false;
    }
    
    // Point clamp sampler (for pixel-perfect sampling)
    D3D11_SAMPLER_DESC pointDesc = linearDesc;
    pointDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    
    return SUCCEEDED(device->CreateSamplerState(&pointDesc, &m_pointClampSampler));
}

bool D3D11RenderPassResources::CreateBlendStates(ID3D11Device* device) {
    // No blending (opaque)
    D3D11_BLEND_DESC noBlendDesc = {};
    noBlendDesc.RenderTarget[0].BlendEnable = FALSE;
    noBlendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    if (FAILED(device->CreateBlendState(&noBlendDesc, &m_noBlendState))) {
        return false;
    }
    
    // Alpha blending
    D3D11_BLEND_DESC alphaBlendDesc = {};
    alphaBlendDesc.RenderTarget[0].BlendEnable = TRUE;
    alphaBlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    alphaBlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    alphaBlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    alphaBlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    alphaBlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    alphaBlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    alphaBlendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    if (FAILED(device->CreateBlendState(&alphaBlendDesc, &m_alphaBlendState))) {
        return false;
    }
    
    // Additive blending
    D3D11_BLEND_DESC additiveBlendDesc = {};
    additiveBlendDesc.RenderTarget[0].BlendEnable = TRUE;
    additiveBlendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    additiveBlendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
    additiveBlendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    additiveBlendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    additiveBlendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    additiveBlendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    additiveBlendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    
    return SUCCEEDED(device->CreateBlendState(&additiveBlendDesc, &m_additiveBlendState));
}

bool D3D11RenderPassResources::CreateRasterizerStates(ID3D11Device* device) {
    // No culling (standard for fullscreen passes)
    D3D11_RASTERIZER_DESC noCullDesc = {};
    noCullDesc.FillMode = D3D11_FILL_SOLID;
    noCullDesc.CullMode = D3D11_CULL_NONE;
    noCullDesc.FrontCounterClockwise = FALSE;
    noCullDesc.DepthBias = 0;
    noCullDesc.SlopeScaledDepthBias = 0.0f;
    noCullDesc.DepthBiasClamp = 0.0f;
    noCullDesc.DepthClipEnable = TRUE;
    noCullDesc.ScissorEnable = FALSE;
    noCullDesc.MultisampleEnable = FALSE;
    noCullDesc.AntialiasedLineEnable = FALSE;
    
    if (FAILED(device->CreateRasterizerState(&noCullDesc, &m_noCullRasterizer))) {
        return false;
    }
    
    // Back face culling (for special cases)
    D3D11_RASTERIZER_DESC backCullDesc = noCullDesc;
    backCullDesc.CullMode = D3D11_CULL_BACK;
    
    return SUCCEEDED(device->CreateRasterizerState(&backCullDesc, &m_backCullRasterizer));
}