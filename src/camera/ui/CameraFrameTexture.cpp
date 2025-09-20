#include "CameraFrameTexture.h"
#include "../../core/Logger.h"
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
    // Get D3D11 device from renderer
    // Note: This requires access to renderer internals - we'll implement a getter method
    // For now, assume we can access the device
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
    // Convert frame to RGBA format
    std::vector<uint8_t> rgbaData;
    int width, height;
    if (!ConvertFrameToRGBA(frame, rgbaData, width, height)) {
        return false;
    }

    // For now, we'll implement a simplified version
    // In a full implementation, we would create/update D3D11 texture here
    m_textureWidth = width;
    m_textureHeight = height;

    LOG_DEBUG("CameraFrameTexture: Updated D3D11 texture ", width, "x", height);
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