#pragma once

#include <windows.h>
#include "OpenGLHeaders.h"
#include <string>
#include <chrono>
#include "IRenderer.h"

#if HAVE_CUDA
#include <memory>
// Forward declarations
class CudaOpenGLInterop;
#endif

// Forward declarations
class IToastRenderer;

// Forward declarations for render pass system
class OpenGLRenderPassPipeline;

class OpenGLRenderer : public IRenderer {
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    
    // IRenderer interface implementation
    bool Initialize(HWND hwnd, int width, int height) override;
    void Cleanup() override;
    bool Present(const RenderTexture& texture) override;
    bool Resize(int width, int height) override;
    bool IsInitialized() const override { return m_initialized; }
    RendererType GetRendererType() const override { return RendererType::OpenGL; }
    bool SupportsCudaInterop() const override;
    
    // OpenGL-specific methods
    bool IsCudaInteropAvailable() const;
    
private:
    bool m_initialized;
    HWND m_hwnd;
    HDC m_hdc;
    HGLRC m_hrc;
    int m_width;          // Window/renderer width
    int m_height;         // Window/renderer height
    int m_textureWidth;   // Texture width (video dimensions)
    int m_textureHeight;  // Texture height (video dimensions)
    
    // OpenGL resources
    GLuint m_texture;
    GLuint m_program;
    GLuint m_vertexShader;
    GLuint m_fragmentShader;
    GLuint m_vao;
    GLuint m_vbo;
    GLuint m_ebo;
    
    // Shader uniforms
    GLint m_textureUniform;
    GLint m_isYUVUniform;
    
#if HAVE_CUDA
    // CUDA interop resources
    std::unique_ptr<CudaOpenGLInterop> m_cudaInterop;
    void* m_cudaTextureResource;  // CUDA graphics resource handle for the main texture
#endif
    
    bool SetupOpenGLContext();
    bool CreateOpenGLContext();
    bool CreateShaders();
    void CreateGeometry();
    void CreateTexture();
    
    bool EnableDebugOutput();
    static void GLAPIENTRY DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
    
    // OpenGL version info
    int m_glMajorVersion;
    int m_glMinorVersion;
    bool m_coreProfile;
    bool m_debugContext;
    
    // WGL extensions for modern context creation
    typedef HGLRC (WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
    
    // Frame timing for render pass effects
    int m_frameNumber;
    float m_totalTime;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    
    void SetupRenderState(bool isYUV = false);
    void DrawQuad();
    bool PresentSoftwareTexture(const RenderTexture& texture);
    bool ResizeTextureIfNeeded(int newWidth, int newHeight);
#if HAVE_CUDA
    bool PresentCudaTexture(const RenderTexture& texture);
#endif
    
#if HAVE_CUDA
    bool InitializeCudaInterop();
    void CleanupCudaInterop();
    bool TestCudaInterop(); // Test if CUDA interop actually works
#endif
    
    void Reset();
    
    // Render pass pipeline
    std::unique_ptr<OpenGLRenderPassPipeline> m_renderPassPipeline;
};

// Vertex structure for full-screen quad
struct GLQuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};