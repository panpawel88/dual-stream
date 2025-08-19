#pragma once

#include <windows.h>
#include <glad/glad.h>
#include <string>

class OpenGLRenderer {
public:
    OpenGLRenderer();
    ~OpenGLRenderer();
    
    bool Initialize(HWND hwnd, int width, int height);
    void Cleanup();
    
    bool Present(const uint8_t* data, int width, int height, int pitch);
    bool Resize(int width, int height);
    
    // Getters
    bool IsInitialized() const { return m_initialized; }
    
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
    
    // Initialization helpers
    bool SetupOpenGL();
    bool CreateModernContext();
    bool CreateShaders();
    bool CreateGeometry();
    bool CreateTexture();
    
    // Modern OpenGL helpers
    bool EnableDebugOutput();
    static void GLAPIENTRY DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
    
    // Context creation helpers
    bool CreateLegacyContext();
    bool UpgradeToModernContext();
    
    // OpenGL version info
    int m_glMajorVersion;
    int m_glMinorVersion;
    bool m_coreProfile;
    bool m_debugContext;
    
    // WGL extensions for modern context creation
    typedef HGLRC (WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
    
    // Rendering helpers
    void SetupRenderState();
    void DrawQuad();
    
    void Reset();
};

// Vertex structure for full-screen quad
struct GLQuadVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};