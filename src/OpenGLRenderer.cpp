#include "OpenGLRenderer.h"
#include "Logger.h"
#include <iostream>
#include <vector>

// Simple vertex shader source (compatible with OpenGL 2.0)
const char* g_glVertexShaderSource = R"(
attribute vec3 position;
attribute vec2 texCoord;

varying vec2 TexCoord;

void main()
{
    gl_Position = vec4(position, 1.0);
    TexCoord = texCoord;
}
)";

// Fragment shader source (compatible with OpenGL 2.0)
const char* g_glFragmentShaderSource = R"(
varying vec2 TexCoord;

uniform sampler2D videoTexture;

void main()
{
    gl_FragColor = texture2D(videoTexture, TexCoord);
}
)";

OpenGLRenderer::OpenGLRenderer()
    : m_initialized(false)
    , m_hwnd(nullptr)
    , m_hdc(nullptr)
    , m_hrc(nullptr)
    , m_width(0)
    , m_height(0)
    , m_texture(0)
    , m_program(0)
    , m_vertexShader(0)
    , m_fragmentShader(0)
    , m_vao(0)
    , m_vbo(0)
    , m_ebo(0)
    , m_textureUniform(-1) {
    
    // Initialize function pointers to nullptr
    glGenBuffers = nullptr;
    glBindBuffer = nullptr;
    glBufferData = nullptr;
    glDeleteBuffers = nullptr;
    glGenVertexArrays = nullptr;
    glBindVertexArray = nullptr;
    glDeleteVertexArrays = nullptr;
    glEnableVertexAttribArray = nullptr;
    glVertexAttribPointer = nullptr;
    glCreateShader = nullptr;
    glShaderSource = nullptr;
    glCompileShader = nullptr;
    glGetShaderiv = nullptr;
    glGetShaderInfoLog = nullptr;
    glDeleteShader = nullptr;
    glCreateProgram = nullptr;
    glAttachShader = nullptr;
    glLinkProgram = nullptr;
    glGetProgramiv = nullptr;
    glGetProgramInfoLog = nullptr;
    glDeleteProgram = nullptr;
    glUseProgram = nullptr;
    glGetUniformLocation = nullptr;
    glUniform1i = nullptr;
    glActiveTexture = nullptr;
}

OpenGLRenderer::~OpenGLRenderer() {
    Cleanup();
}

bool OpenGLRenderer::Initialize(HWND hwnd, int width, int height) {
    if (m_initialized) {
        Cleanup();
    }
    
    m_hwnd = hwnd;
    m_width = width;
    m_height = height;
    
    LOG_INFO("Initializing OpenGL renderer (", width, "x", height, ")");
    
    // Setup OpenGL context
    if (!SetupOpenGL()) {
        LOG_ERROR("Failed to setup OpenGL context");
        Cleanup();
        return false;
    }
    
    // Load OpenGL extensions
    if (!LoadOpenGLExtensions()) {
        LOG_ERROR("Failed to load OpenGL extensions");
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
    
    // Create texture
    if (!CreateTexture()) {
        LOG_ERROR("Failed to create texture");
        Cleanup();
        return false;
    }
    
    // Set viewport
    glViewport(0, 0, width, height);
    
    // Enable texture
    glEnable(GL_TEXTURE_2D);
    
    // Set clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    m_initialized = true;
    LOG_INFO("OpenGL renderer initialized successfully");
    return true;
}

void OpenGLRenderer::Cleanup() {
    Reset();
}

bool OpenGLRenderer::Present(const uint8_t* data, int width, int height, int pitch) {
    if (!m_initialized) {
        return false;
    }
    
    // Update texture with new frame data
    if (data && width > 0 && height > 0 && pitch > 0) {
        glBindTexture(GL_TEXTURE_2D, m_texture);
        
        // Set pixel store parameters for proper pitch handling
        glPixelStorei(GL_UNPACK_ROW_LENGTH, pitch / 4); // Assuming BGRA format (4 bytes per pixel)
        
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA_EXT, GL_UNSIGNED_BYTE, data);
        
        // Reset pixel store
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }
    
    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Only draw if we have valid data
    if (data) {
        // Setup render state
        SetupRenderState();
        
        // Draw fullscreen quad
        DrawQuad();
    }
    
    // Swap buffers
    SwapBuffers(m_hdc);
    
    return true;
}

bool OpenGLRenderer::Resize(int width, int height) {
    if (!m_initialized || (width == m_width && height == m_height)) {
        return true;
    }
    
    m_width = width;
    m_height = height;
    
    // Update viewport
    glViewport(0, 0, width, height);
    
    return true;
}

bool OpenGLRenderer::SetupOpenGL() {
    // Get device context
    m_hdc = GetDC(m_hwnd);
    if (!m_hdc) {
        LOG_ERROR("Failed to get device context");
        return false;
    }
    
    // Set pixel format
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    
    int pixelFormat = ChoosePixelFormat(m_hdc, &pfd);
    if (!pixelFormat) {
        LOG_ERROR("Failed to choose pixel format");
        return false;
    }
    
    if (!SetPixelFormat(m_hdc, pixelFormat, &pfd)) {
        LOG_ERROR("Failed to set pixel format");
        return false;
    }
    
    // Create OpenGL context
    m_hrc = wglCreateContext(m_hdc);
    if (!m_hrc) {
        LOG_ERROR("Failed to create OpenGL context");
        return false;
    }
    
    // Make context current
    if (!wglMakeCurrent(m_hdc, m_hrc)) {
        LOG_ERROR("Failed to make OpenGL context current");
        return false;
    }
    
    // Get OpenGL version info
    const char* version = (const char*)glGetString(GL_VERSION);
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    
    LOG_INFO("OpenGL Version: ", version ? version : "Unknown");
    LOG_INFO("OpenGL Vendor: ", vendor ? vendor : "Unknown");
    LOG_INFO("OpenGL Renderer: ", renderer ? renderer : "Unknown");
    
    return true;
}

bool OpenGLRenderer::LoadOpenGLExtensions() {
    // Load required OpenGL extensions
    glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)wglGetProcAddress("glDeleteVertexArrays");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer");
    glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
    glDeleteShader = (PFNGLDELETESHADERPROC)wglGetProcAddress("glDeleteShader");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)wglGetProcAddress("glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)wglGetProcAddress("glGetProgramInfoLog");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)wglGetProcAddress("glDeleteProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
    glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
    
    // Check if all required functions were loaded
    if (!glGenBuffers || !glBindBuffer || !glBufferData || !glDeleteBuffers ||
        !glGenVertexArrays || !glBindVertexArray || !glDeleteVertexArrays ||
        !glEnableVertexAttribArray || !glVertexAttribPointer ||
        !glCreateShader || !glShaderSource || !glCompileShader ||
        !glGetShaderiv || !glGetShaderInfoLog || !glDeleteShader ||
        !glCreateProgram || !glAttachShader || !glLinkProgram ||
        !glGetProgramiv || !glGetProgramInfoLog || !glDeleteProgram ||
        !glUseProgram || !glGetUniformLocation || !glUniform1i ||
        !glActiveTexture) {
        
        LOG_ERROR("Failed to load required OpenGL extensions");
        return false;
    }
    
    return true;
}

bool OpenGLRenderer::CreateShaders() {
    // Check if shader functions are available
    if (!glCreateShader || !glShaderSource || !glCompileShader || 
        !glGetShaderiv || !glGetShaderInfoLog || !glCreateProgram ||
        !glAttachShader || !glLinkProgram || !glGetProgramiv ||
        !glGetProgramInfoLog || !glGetUniformLocation) {
        
        LOG_INFO("Shader functions not available, using fixed function pipeline");
        m_program = 0;
        m_vertexShader = 0;
        m_fragmentShader = 0;
        m_textureUniform = -1;
        return true; // Not an error, just use fixed function
    }
    
    GLint success;
    char infoLog[512];
    
    // Create vertex shader
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_vertexShader, 1, &g_glVertexShaderSource, nullptr);
    glCompileShader(m_vertexShader);
    
    glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(m_vertexShader, 512, nullptr, infoLog);
        LOG_ERROR("Vertex shader compilation failed: ", infoLog);
        return false;
    }
    
    // Create fragment shader
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_fragmentShader, 1, &g_glFragmentShaderSource, nullptr);
    glCompileShader(m_fragmentShader);
    
    glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(m_fragmentShader, 512, nullptr, infoLog);
        LOG_ERROR("Fragment shader compilation failed: ", infoLog);
        return false;
    }
    
    // Create shader program
    m_program = glCreateProgram();
    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);
    glLinkProgram(m_program);
    
    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_program, 512, nullptr, infoLog);
        LOG_ERROR("Shader program linking failed: ", infoLog);
        return false;
    }
    
    // Get uniform locations
    m_textureUniform = glGetUniformLocation(m_program, "videoTexture");
    
    LOG_INFO("Shaders compiled and linked successfully");
    return true;
}

bool OpenGLRenderer::CreateGeometry() {
    // Create fullscreen quad vertices
    GLQuadVertex vertices[] = {
        // Position (x, y, z)    // TexCoord (u, v)
        { {-1.0f,  1.0f, 0.0f}, {0.0f, 0.0f} }, // Top-left
        { { 1.0f,  1.0f, 0.0f}, {1.0f, 0.0f} }, // Top-right
        { { 1.0f, -1.0f, 0.0f}, {1.0f, 1.0f} }, // Bottom-right
        { {-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f} }  // Bottom-left
    };
    
    // Create indices
    GLuint indices[] = { 0, 1, 2, 0, 2, 3 };
    
    // Try to use VBO if available, otherwise fall back to immediate mode
    if (glGenBuffers && glBindBuffer && glBufferData) {
        // Generate VAO if available (OpenGL 3.0+)
        if (glGenVertexArrays && glBindVertexArray) {
            glGenVertexArrays(1, &m_vao);
            glBindVertexArray(m_vao);
        }
        
        // Generate VBO
        glGenBuffers(1, &m_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        // Generate EBO
        glGenBuffers(1, &m_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Set vertex attributes if VAO is available
        if (m_vao && glVertexAttribPointer && glEnableVertexAttribArray) {
            // Position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLQuadVertex), (void*)0);
            glEnableVertexAttribArray(0);
            
            // Texture coordinate attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLQuadVertex), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
        }
        
        // Unbind
        if (glBindBuffer) {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        if (m_vao && glBindVertexArray) {
            glBindVertexArray(0);
        }
    } else {
        LOG_INFO("VBO not available, will use immediate mode rendering");
    }
    
    return true;
}

bool OpenGLRenderer::CreateTexture() {
    // Generate texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Create initial texture with placeholder data (will be updated during Present)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, nullptr);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

void OpenGLRenderer::SetupRenderState() {
    // Use shader program if available
    if (glUseProgram && m_program) {
        glUseProgram(m_program);
        
        // Bind texture
        if (glActiveTexture) {
            glActiveTexture(GL_TEXTURE0);
        }
        glBindTexture(GL_TEXTURE_2D, m_texture);
        
        // Set texture uniform
        if (glUniform1i && m_textureUniform >= 0) {
            glUniform1i(m_textureUniform, 0);
        }
        
        // Bind VAO if available
        if (m_vao && glBindVertexArray) {
            glBindVertexArray(m_vao);
        }
    } else {
        // Fallback to fixed function pipeline
        glBindTexture(GL_TEXTURE_2D, m_texture);
        glEnable(GL_TEXTURE_2D);
    }
}

void OpenGLRenderer::DrawQuad() {
    if (m_vao && m_ebo) {
        // Use VBO/VAO rendering
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    } else {
        // Use immediate mode rendering
        glBegin(GL_TRIANGLES);
        
        // First triangle
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f, 0.0f); // Top-left
        glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f, 0.0f); // Top-right
        glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f, 0.0f); // Bottom-right
        
        // Second triangle
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f, 0.0f); // Top-left
        glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f, 0.0f); // Bottom-right
        glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.0f); // Bottom-left
        
        glEnd();
    }
}

void OpenGLRenderer::Reset() {
    m_initialized = false;
    
    // Delete OpenGL resources
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }
    
    if (m_vao && glDeleteVertexArrays) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    
    if (m_vbo && glDeleteBuffers) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    
    if (m_ebo && glDeleteBuffers) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }
    
    if (m_program && glDeleteProgram) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
    
    if (m_vertexShader && glDeleteShader) {
        glDeleteShader(m_vertexShader);
        m_vertexShader = 0;
    }
    
    if (m_fragmentShader && glDeleteShader) {
        glDeleteShader(m_fragmentShader);
        m_fragmentShader = 0;
    }
    
    // Release OpenGL context
    if (m_hrc) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(m_hrc);
        m_hrc = nullptr;
    }
    
    if (m_hdc) {
        ReleaseDC(m_hwnd, m_hdc);
        m_hdc = nullptr;
    }
    
    m_hwnd = nullptr;
    m_width = 0;
    m_height = 0;
    m_textureUniform = -1;
}