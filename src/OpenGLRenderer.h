#pragma once

#include <windows.h>
#include <glad/gl.h>
#include <string>

#if USE_OPENGL_RENDERER && HAVE_CUDA
#include <memory>
// Forward declarations
class CudaOpenGLInterop;
struct DecodedFrame;
#endif

class OpenGLRenderer {
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    
    bool Initialize(HWND hwnd, int width, int height);
    void Cleanup();
    
    bool Present(const uint8_t* data, int width, int height, int pitch);
    bool Resize(int width, int height);
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // Hardware texture presentation (for CUDA decoded frames)
    bool PresentHardware(const DecodedFrame& frame);
#endif
    
    // Getters
    bool IsInitialized() const { return m_initialized; }
#if USE_OPENGL_RENDERER && HAVE_CUDA
    bool IsCudaInteropAvailable() const; // Defined in .cpp file
#endif
    
private:
    bool m_initialized;
    HWND m_hwnd;
    HDC m_hdc;
    HGLRC m_hrc;
    int m_width;
    int m_height;
    
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
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // CUDA interop resources
    std::unique_ptr<CudaOpenGLInterop> m_cudaInterop;
    void* m_cudaTextureResource;  // CUDA graphics resource handle for the main texture
#endif
    
    // Initialization helpers
    bool SetupOpenGLContext();
    bool CreateOpenGLContext();
    bool CreateShaders();
    bool CreateGeometry();
    bool CreateTexture();
    
    // Modern OpenGL helpers
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
    
    // Rendering helpers
    void SetupRenderState(bool isYUV = false);
    void DrawQuad();
    
#if USE_OPENGL_RENDERER && HAVE_CUDA
    // CUDA interop helpers
    bool InitializeCudaInterop();
    void CleanupCudaInterop();
    bool TestCudaInterop(); // Test if CUDA interop actually works
#endif
    
    void Reset();
};

// Vertex structure for full-screen quad
struct GLQuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};