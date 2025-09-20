#include "CameraFrameTexture.h"
#include "../../core/Logger.h"
#include "../../rendering/D3D11Renderer.h"
#include <opencv2/opencv.hpp>
#include <algorithm>

#ifdef _WIN32
#include <d3d11.h>
#endif

#ifdef HAVE_OPENGL
#include <GL/gl.h>
#endif

CameraFrameTexture::CameraFrameTexture()
    : m_renderer(nullptr)
    , m_rendererType(RendererType::DirectX11)
    , m_textureWidth(0)
    , m_textureHeight(0)
    , m_maxWidth(640)
    , m_maxHeight(480)
    , m_glTexture(0)
{
}

CameraFrameTexture::~CameraFrameTexture() {
    Cleanup();
}

bool CameraFrameTexture::Initialize(IRenderer* renderer) {
    if (!renderer) {
        LOG_ERROR("CameraFrameTexture::Initialize: null renderer");
        return false;
    }

    m_renderer = renderer;
    m_rendererType = renderer->GetRendererType();

    switch (m_rendererType) {
        case RendererType::DirectX11:
            return InitializeD3D11();
        case RendererType::OpenGL:
            return InitializeOpenGL();
        default:
            LOG_ERROR("CameraFrameTexture::Initialize: unsupported renderer type");
            return false;
    }
}

bool CameraFrameTexture::InitializeD3D11() {
#ifdef _WIN32
    // Cast renderer to D3D11Renderer to access device and context
    D3D11Renderer* d3d11Renderer = dynamic_cast<D3D11Renderer*>(m_renderer);
    if (!d3d11Renderer) {
        LOG_ERROR("CameraFrameTexture: Failed to cast renderer to D3D11Renderer");
        return false;
    }

    // Get D3D11 device and context
    m_d3dDevice = d3d11Renderer->GetDevice();
    if (!m_d3dDevice) {
        LOG_ERROR("CameraFrameTexture: Failed to get D3D11 device");
        return false;
    }

    m_d3dDevice->GetImmediateContext(m_d3dContext.GetAddressOf());
    if (!m_d3dContext) {
        LOG_ERROR("CameraFrameTexture: Failed to get D3D11 device context");
        return false;
    }

    LOG_INFO("CameraFrameTexture: Initialized D3D11 backend");
    return true;
#else
    LOG_ERROR("CameraFrameTexture: D3D11 not available on this platform");
    return false;
#endif
}

bool CameraFrameTexture::InitializeOpenGL() {
#ifdef HAVE_OPENGL
    // Generate OpenGL texture
    glGenTextures(1, &m_glTexture);
    if (m_glTexture == 0) {
        LOG_ERROR("CameraFrameTexture: Failed to generate OpenGL texture");
        return false;
    }

    LOG_INFO("CameraFrameTexture: Initialized OpenGL backend with texture ID ", m_glTexture);
    return true;
#else
    LOG_ERROR("CameraFrameTexture: OpenGL not available");
    return false;
#endif
}

bool CameraFrameTexture::UpdateTexture(std::shared_ptr<const CameraFrame> frame) {
    if (!frame || !IsValid()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_textureMutex);

    switch (m_rendererType) {
        case RendererType::DirectX11:
            return UpdateD3D11Texture(*frame);
        case RendererType::OpenGL:
            return UpdateOpenGLTexture(*frame);
        default:
            return false;
    }
}

bool CameraFrameTexture::UpdateD3D11Texture(const CameraFrame& frame) {
#ifdef _WIN32
    if (!m_d3dDevice || !m_d3dContext) {
        LOG_ERROR("CameraFrameTexture: D3D11 device or context not initialized");
        return false;
    }

    // Convert frame to RGBA format
    std::vector<uint8_t> rgbaData;
    int width, height;
    if (!ConvertFrameToRGBA(frame, rgbaData, width, height)) {
        return false;
    }

    // Check if we need to recreate the texture (size changed or not created yet)
    bool needsRecreate = !m_d3dTexture || m_textureWidth != width || m_textureHeight != height;

    if (needsRecreate) {
        // Release old resources
        m_d3dSRV.Reset();
        m_d3dTexture.Reset();

        // Create texture description
        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        texDesc.MiscFlags = 0;

        // Create texture with initial data
        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = rgbaData.data();
        initData.SysMemPitch = width * 4; // 4 bytes per pixel (RGBA)

        HRESULT hr = m_d3dDevice->CreateTexture2D(&texDesc, &initData, m_d3dTexture.GetAddressOf());
        if (FAILED(hr)) {
            LOG_ERROR("CameraFrameTexture: Failed to create D3D11 texture, hr=", std::hex, hr);
            return false;
        }

        // Create shader resource view
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = texDesc.Format;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = 1;

        hr = m_d3dDevice->CreateShaderResourceView(m_d3dTexture.Get(), &srvDesc, m_d3dSRV.GetAddressOf());
        if (FAILED(hr)) {
            LOG_ERROR("CameraFrameTexture: Failed to create shader resource view, hr=", std::hex, hr);
            m_d3dTexture.Reset();
            return false;
        }

        m_textureWidth = width;
        m_textureHeight = height;
        LOG_DEBUG("CameraFrameTexture: Created new D3D11 texture ", width, "x", height);
    } else {
        // Update existing texture
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        HRESULT hr = m_d3dContext->Map(m_d3dTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        if (FAILED(hr)) {
            LOG_ERROR("CameraFrameTexture: Failed to map texture for writing, hr=", std::hex, hr);
            return false;
        }

        // Copy RGBA data to texture
        uint8_t* destData = static_cast<uint8_t*>(mappedResource.pData);
        uint32_t destPitch = mappedResource.RowPitch;
        uint32_t srcPitch = width * 4; // 4 bytes per pixel

        for (int y = 0; y < height; ++y) {
            memcpy(destData + y * destPitch, rgbaData.data() + y * srcPitch, srcPitch);
        }

        m_d3dContext->Unmap(m_d3dTexture.Get(), 0);
        LOG_DEBUG("CameraFrameTexture: Updated D3D11 texture ", width, "x", height);
    }

    return true;
#else
    return false;
#endif
}

bool CameraFrameTexture::UpdateOpenGLTexture(const CameraFrame& frame) {
#ifdef HAVE_OPENGL
    if (m_glTexture == 0) {
        return false;
    }

    // Convert frame to RGBA format
    std::vector<uint8_t> rgbaData;
    int width, height;
    if (!ConvertFrameToRGBA(frame, rgbaData, width, height)) {
        return false;
    }

    // Bind and update OpenGL texture
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbaData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_textureWidth = width;
    m_textureHeight = height;

    LOG_DEBUG("CameraFrameTexture: Updated OpenGL texture ", width, "x", height);
    return true;
#else
    return false;
#endif
}

void* CameraFrameTexture::GetImGuiTextureID() const {
    std::lock_guard<std::mutex> lock(m_textureMutex);

    switch (m_rendererType) {
        case RendererType::DirectX11:
#ifdef _WIN32
            return m_d3dSRV.Get();
#else
            return nullptr;
#endif
        case RendererType::OpenGL:
            return reinterpret_cast<void*>(static_cast<uintptr_t>(m_glTexture));
        default:
            return nullptr;
    }
}

void CameraFrameTexture::GetTextureDimensions(int& width, int& height) const {
    std::lock_guard<std::mutex> lock(m_textureMutex);
    width = m_textureWidth;
    height = m_textureHeight;
}

bool CameraFrameTexture::IsValid() const {
    switch (m_rendererType) {
        case RendererType::DirectX11:
#ifdef _WIN32
            // Only check for device and context - SRV is created on first update
            return m_d3dDevice && m_d3dContext;
#else
            return false;
#endif
        case RendererType::OpenGL:
            return m_glTexture != 0;
        default:
            return false;
    }
}

void CameraFrameTexture::SetMaxDimensions(int maxWidth, int maxHeight) {
    m_maxWidth = maxWidth;
    m_maxHeight = maxHeight;
}

void CameraFrameTexture::Cleanup() {
    std::lock_guard<std::mutex> lock(m_textureMutex);

    switch (m_rendererType) {
        case RendererType::DirectX11:
            CleanupD3D11Resources();
            break;
        case RendererType::OpenGL:
            CleanupOpenGLResources();
            break;
    }

    m_textureWidth = 0;
    m_textureHeight = 0;
    m_renderer = nullptr;
}

void CameraFrameTexture::CleanupD3D11Resources() {
#ifdef _WIN32
    m_d3dSRV.Reset();
    m_d3dTexture.Reset();
    m_d3dContext.Reset();
    m_d3dDevice.Reset();
#endif
}

void CameraFrameTexture::CleanupOpenGLResources() {
#ifdef HAVE_OPENGL
    if (m_glTexture != 0) {
        glDeleteTextures(1, &m_glTexture);
        m_glTexture = 0;
    }
#endif
}

void CameraFrameTexture::CalculateScaledDimensions(int srcWidth, int srcHeight, int& dstWidth, int& dstHeight) {
    if (srcWidth <= m_maxWidth && srcHeight <= m_maxHeight) {
        dstWidth = srcWidth;
        dstHeight = srcHeight;
        return;
    }

    double scaleX = static_cast<double>(m_maxWidth) / srcWidth;
    double scaleY = static_cast<double>(m_maxHeight) / srcHeight;
    double scale = std::min(scaleX, scaleY);

    dstWidth = static_cast<int>(srcWidth * scale);
    dstHeight = static_cast<int>(srcHeight * scale);
}

bool CameraFrameTexture::ConvertFrameToRGBA(const CameraFrame& frame, std::vector<uint8_t>& rgbaData, int& width, int& height) {
    if (!frame.mat.data) {
        LOG_ERROR("CameraFrameTexture: Invalid frame data");
        return false;
    }

    cv::Mat rgbaMat;
    cv::Mat sourceMat = frame.mat;

    // Calculate scaled dimensions
    CalculateScaledDimensions(sourceMat.cols, sourceMat.rows, width, height);

    // Resize if needed
    if (width != sourceMat.cols || height != sourceMat.rows) {
        cv::resize(sourceMat, sourceMat, cv::Size(width, height));
    }

    // Convert to RGBA based on input format
    switch (frame.format) {
        case CameraFormat::BGR8:
            cv::cvtColor(sourceMat, rgbaMat, cv::COLOR_BGR2RGBA);
            break;
        case CameraFormat::RGB8:
            cv::cvtColor(sourceMat, rgbaMat, cv::COLOR_RGB2RGBA);
            break;
        case CameraFormat::GRAY8:
            cv::cvtColor(sourceMat, rgbaMat, cv::COLOR_GRAY2RGBA);
            break;
        case CameraFormat::BGRA8:
            cv::cvtColor(sourceMat, rgbaMat, cv::COLOR_BGRA2RGBA);
            break;
        case CameraFormat::RGBA8:
            rgbaMat = sourceMat.clone();
            break;
        default:
            LOG_ERROR("CameraFrameTexture: Unsupported camera format");
            return false;
    }

    // Copy data to output vector
    size_t dataSize = rgbaMat.total() * rgbaMat.elemSize();
    rgbaData.resize(dataSize);
    std::memcpy(rgbaData.data(), rgbaMat.data, dataSize);

    return true;
}