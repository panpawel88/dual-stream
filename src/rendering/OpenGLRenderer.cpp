#include "OpenGLRenderer.h"
#include "core/Logger.h"
#include "renderpass/RenderPassConfigLoader.h"
#include "renderpass/opengl/OpenGLRenderPassContext.h"
#include "renderpass/opengl/OpenGLRenderPassPipeline.h"
#include "core/Config.h"
#include <iostream>
#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>

#if HAVE_CUDA
#include "CudaOpenGLInterop.h"
#endif

// WGL constants for modern context creation
#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
#define WGL_CONTEXT_FLAGS_ARB             0x2094
#define WGL_CONTEXT_PROFILE_MASK_ARB      0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001
#define WGL_CONTEXT_DEBUG_BIT_ARB         0x00000001
#endif

// Modern vertex shader source (OpenGL 4.6 Core Profile)
const char* g_glVertexShaderSource460 = R"(
#version 460 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(position, 1.0);
    TexCoord = texCoord;
}
)";

// Modern fragment shader source (OpenGL 4.6 Core Profile)
const char* g_glFragmentShaderSource460 = R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;

void main()
{
    // CUDA hardware decoding converts YUV to RGBA, so always treat as RGBA texture
    // The isYUV uniform is kept for compatibility but not used in CUDA path
    FragColor = texture(videoTexture, TexCoord);
}
)";


OpenGLRenderer::OpenGLRenderer()
    : m_initialized(false)
    , m_hwnd(nullptr)
    , m_hdc(nullptr)
    , m_hrc(nullptr)
    , m_width(0)
    , m_height(0)
    , m_textureWidth(0)
    , m_textureHeight(0)
    , m_texture(0)
    , m_program(0)
    , m_vertexShader(0)
    , m_fragmentShader(0)
    , m_vao(0)
    , m_vbo(0)
    , m_ebo(0)
    , m_textureUniform(-1)
    , m_isYUVUniform(-1)
    , m_glMajorVersion(0)
    , m_glMinorVersion(0)
    , m_coreProfile(false)
    , m_debugContext(false)
    , wglCreateContextAttribsARB(nullptr)
    , wglSwapIntervalEXT(nullptr)
    , m_frameNumber(0)
    , m_totalTime(0.0f)
#ifdef TRACY_ENABLE
    , m_tracyGpuContextInitialized(false)
#endif
{
    
    // Load rendering configuration from config system
    Config* config = Config::GetInstance();
    m_vsyncMode = config->GetInt("rendering.vsync_mode", 1);
    
    // Validate range - avoid potential macro conflicts
    if (m_vsyncMode < 0) m_vsyncMode = 0;
    if (m_vsyncMode > 2) m_vsyncMode = 2;
    
    LOG_INFO("OpenGL Renderer configuration: VSync=", m_vsyncMode);
    
#if HAVE_CUDA
    m_cudaInterop = nullptr; // Will be initialized in Initialize()
    m_cudaTextureResource = nullptr;
#endif
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
    m_textureWidth = width;   // Initially same as window
    m_textureHeight = height;
    
    LOG_INFO("Initializing OpenGL renderer (", width, "x", height, ")");
    
    // Setup OpenGL 4.6 Core Profile context
    if (!SetupOpenGLContext()) {
        LOG_ERROR("Failed to create OpenGL 4.6 Core Profile context");
        Cleanup();
        return false;
    }
    
    // Get OpenGL version info (GLAD is now initialized)
    // Try to get version info - may fail on very old contexts
    glGetIntegerv(GL_MAJOR_VERSION, &m_glMajorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &m_glMinorVersion);
    
    // Check if we got valid version info
    if (m_glMajorVersion == 0) {
        // Fallback: parse version string
        const char* version = (const char*)glGetString(GL_VERSION);
        if (version) {
            sscanf(version, "%d.%d", &m_glMajorVersion, &m_glMinorVersion);
        }
    }
    
    // Try to get context info - these may not be available in legacy contexts
    GLint contextFlags = 0;
    if (m_glMajorVersion >= 3) {
        glGetIntegerv(GL_CONTEXT_FLAGS, &contextFlags);
        m_debugContext = (contextFlags & GL_CONTEXT_FLAG_DEBUG_BIT) != 0;
        
        GLint profileMask = 0;
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profileMask);
        m_coreProfile = (profileMask & GL_CONTEXT_CORE_PROFILE_BIT) != 0;
    } else {
        m_debugContext = false;
        m_coreProfile = false;
    }
    
    LOG_INFO("OpenGL ", m_glMajorVersion, ".", m_glMinorVersion, " ", 
             m_coreProfile ? "Core" : "Compatibility", " Profile loaded");
    
    // Verify we have at least OpenGL 4.6
    if (m_glMajorVersion < 4 || (m_glMajorVersion == 4 && m_glMinorVersion < 6)) {
        LOG_ERROR("OpenGL 4.6 is required but only ", m_glMajorVersion, ".", m_glMinorVersion, " is available");
        Cleanup();
        return false;
    }
    
    // Enable debug output if available
    if (m_debugContext) {
        EnableDebugOutput();
    }
    
    // Create shaders
    if (!CreateShaders()) {
        LOG_ERROR("Failed to create shaders");
        Cleanup();
        return false;
    }
    
    // Create geometry
    CreateGeometry();
    
    // Create texture
    CreateTexture();
    
#if HAVE_CUDA
    // Initialize CUDA interop for hardware decoding
    if (!InitializeCudaInterop()) {
        LOG_WARNING("CUDA interop initialization failed - hardware decoding will fall back to software");
    } else if (!TestCudaInterop()) {
        LOG_WARNING("CUDA interop test failed - disabling CUDA interop, will use software decoding");
        CleanupCudaInterop(); // Disable CUDA interop
    } else {
        LOG_INFO("CUDA interop initialized and tested successfully - hardware CUDA frames supported");
    }
#endif
    
    // Set viewport
    glViewport(0, 0, width, height);
    
    // Set clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Initialize render pass pipeline
    Config* config = Config::GetInstance();
    m_renderPassPipeline = RenderPassConfigLoader::LoadOpenGLPipeline(config, m_hwnd);
    if (m_renderPassPipeline) {
        LOG_INFO("OpenGL render pass pipeline initialized successfully");
    } else {
        LOG_INFO("OpenGL render pass pipeline disabled or failed to initialize");
    }

    // Initialize timing for render pass effects
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastFrameTime = m_startTime;
    m_frameNumber = 0;
    m_totalTime = 0.0f;

    // Initialize Tracy GPU profiling context
#ifdef TRACY_ENABLE
    PROFILE_GPU_OPENGL_CONTEXT();
    m_tracyGpuContextInitialized = true;
    LOG_INFO("Tracy OpenGL GPU profiling context initialized");
#endif

    m_initialized = true;
    LOG_INFO("OpenGL renderer initialized successfully");
    return true;
}

void OpenGLRenderer::Cleanup() {
    Reset();
}

bool OpenGLRenderer::Present(const RenderTexture& texture) {
    PROFILE_RENDER();

    if (!m_initialized) {
        return false;
    }
    
    // Update frame timing for render pass effects
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto deltaTime = std::chrono::duration<float>(currentTime - m_startTime);
    auto frameDelta = std::chrono::duration<float>(currentTime - m_lastFrameTime);
    
    m_totalTime = deltaTime.count();
    m_lastFrameTime = currentTime;
    m_frameNumber++;
    
    // Clear screen first
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Handle different texture types
    if (!texture.IsValid()) {
        // Present black screen for invalid texture
        SwapBuffers(m_hdc);
        return true;
    }
    
    bool renderSuccess = false;
    
    switch (texture.type) {
        case TextureType::Software:
            renderSuccess = PresentSoftwareTexture(texture);
            break;
            
#if HAVE_CUDA            
        case TextureType::CUDA:
            renderSuccess = PresentCudaTexture(texture);
            break;
#endif
            
        case TextureType::D3D11:
            LOG_WARNING("D3D11 texture not supported in OpenGL renderer");
            renderSuccess = false;
            break;
            
        case TextureType::OpenGL:
            LOG_WARNING("OpenGL texture ID type not yet implemented");
            renderSuccess = false;
            break;
            
        default:
            LOG_ERROR("Unknown texture type");
            renderSuccess = false;
            break;
    }
    
    
    // Always swap buffers, even for failed renders (shows black screen)
    SwapBuffers(m_hdc);

    // Collect Tracy GPU profiling data
#ifdef TRACY_ENABLE
    if (m_tracyGpuContextInitialized) {
        PROFILE_GPU_OPENGL_COLLECT();
    }
#endif

    return renderSuccess;
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

// Hybrid OpenGL function loader for Win32/WGL contexts
void* GetAnyGLFuncAddress(const char* name) {
    void* p = (void*)wglGetProcAddress(name);
    if (p == 0 || p == (void*)0x1 || p == (void*)0x2 || 
        p == (void*)0x3 || p == (void*)-1) {
        HMODULE module = LoadLibraryA("opengl32.dll");
        if (module) {
            p = (void*)GetProcAddress(module, name);
        }
    }
    return p;
}

bool OpenGLRenderer::SetupOpenGLContext() {
    // Get device context
    m_hdc = GetDC(m_hwnd);
    if (!m_hdc) {
        LOG_ERROR("Failed to get device context");
        return false;
    }
    
    // Create OpenGL 4.6 Core Profile context directly
    if (!CreateOpenGLContext()) {
        LOG_ERROR("Failed to create OpenGL 4.6 Core Profile context");
        return false;
    }
    
    // Initialize GLAD immediately after context creation
    if (!gladLoadGL((GLADloadfunc)GetAnyGLFuncAddress)) {
        LOG_ERROR("Failed to initialize GLAD for OpenGL 4.6");
        return false;
    }
    
    // Get OpenGL version info for logging
    const char* version = (const char*)glGetString(GL_VERSION);
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    
    LOG_INFO("OpenGL Version: ", version ? version : "Unknown");
    LOG_INFO("OpenGL Vendor: ", vendor ? vendor : "Unknown");
    LOG_INFO("OpenGL Renderer: ", renderer ? renderer : "Unknown");
    
    return true;
}

bool OpenGLRenderer::CreateOpenGLContext() {
    // Set pixel format
    PIXELFORMATDESCRIPTOR pfd = {
        .nSize = sizeof(PIXELFORMATDESCRIPTOR),
        .nVersion = 1,
        .dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        .iPixelType = PFD_TYPE_RGBA,
        .cColorBits = 32,
        .cDepthBits = 24,
        .cStencilBits = 8,
        .iLayerType = PFD_MAIN_PLANE
    };
    
    int pixelFormat = ChoosePixelFormat(m_hdc, &pfd);
    if (!pixelFormat) {
        LOG_ERROR("Failed to choose pixel format");
        return false;
    }
    
    if (!SetPixelFormat(m_hdc, pixelFormat, &pfd)) {
        LOG_ERROR("Failed to set pixel format");
        return false;
    }
    
    // Create temporary legacy context to get WGL extensions
    HGLRC tempContext = wglCreateContext(m_hdc);
    if (!tempContext) {
        LOG_ERROR("Failed to create temporary OpenGL context");
        return false;
}
    
    if (!wglMakeCurrent(m_hdc, tempContext)) {
        LOG_ERROR("Failed to make temporary OpenGL context current");
        wglDeleteContext(tempContext);
        return false;
    }
    
    // Load WGL extensions
    wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
    
    if (!wglCreateContextAttribsARB) {
        LOG_ERROR("wglCreateContextAttribsARB not available - OpenGL 4.6 Core Profile not supported");
        wglDeleteContext(tempContext);
        return false;
    }
    
    // Create OpenGL 4.6 Core Profile context with debug
    const int contextAttribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 6,
        WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
        WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0
    };
    
    HGLRC newContext = wglCreateContextAttribsARB(m_hdc, nullptr, contextAttribs);
    if (!newContext) {
        // Try without debug if debug failed
        const int contextAttribsNoDebug[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
            WGL_CONTEXT_MINOR_VERSION_ARB, 6,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0
        };
        
        newContext = wglCreateContextAttribsARB(m_hdc, nullptr, contextAttribsNoDebug);
        if (!newContext) {
            LOG_ERROR("Failed to create OpenGL 4.6 Core Profile context");
            wglDeleteContext(tempContext);
            return false;
        }
        LOG_INFO("Created OpenGL 4.6 Core Profile context");
    } else {
        LOG_INFO("Created OpenGL 4.6 Core Profile context with debug");
    }
    
    // Switch to new context
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(tempContext);
    m_hrc = newContext;
    
    if (!wglMakeCurrent(m_hdc, m_hrc)) {
        LOG_ERROR("Failed to make OpenGL 4.6 context current");
        wglDeleteContext(newContext);
        m_hrc = nullptr;
        return false;
    }
    
    // Get WGL extension for VSync control
    wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    if (wglSwapIntervalEXT) {
        int swapInterval = 0;
        
        switch (m_vsyncMode) {
            case 0: // VSync off
                swapInterval = 0;
                break;
            case 1: // VSync on
                swapInterval = 1;
                break;
            case 2: // Adaptive VSync (try -1, fallback to 1)
                swapInterval = -1;
                if (!wglSwapIntervalEXT(-1)) {
                    LOG_WARNING("Adaptive VSync not supported, falling back to regular VSync");
                    swapInterval = 1;
                }
                break;
            default:
                swapInterval = 1;
                break;
        }
        
        if (swapInterval >= 0 && !wglSwapIntervalEXT(swapInterval)) {
            LOG_WARNING("Failed to set VSync mode to ", swapInterval);
        } else {
            LOG_INFO("OpenGL VSync set to mode: ", swapInterval);
        }
    } else {
        LOG_WARNING("wglSwapIntervalEXT not available, VSync control disabled");
    }
    
    return true;
}


bool OpenGLRenderer::CreateShaders() {
    GLint success;
    char infoLog[512];
    
    // Create vertex shader (OpenGL 4.6 only)
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_vertexShader, 1, &g_glVertexShaderSource460, nullptr);
    glCompileShader(m_vertexShader);
    
    glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(m_vertexShader, 512, nullptr, infoLog);
        LOG_ERROR("Vertex shader compilation failed: ", infoLog);
        glDeleteShader(m_vertexShader);
        return false;
    }
    
    // Create fragment shader (OpenGL 4.6 only)
    m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_fragmentShader, 1, &g_glFragmentShaderSource460, nullptr);
    glCompileShader(m_fragmentShader);
    
    glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(m_fragmentShader, 512, nullptr, infoLog);
        LOG_ERROR("Fragment shader compilation failed: ", infoLog);
        glDeleteShader(m_vertexShader);
        glDeleteShader(m_fragmentShader);
        return false;
    }
    
    LOG_INFO("Successfully compiled GLSL 4.60 shaders");
    
    // Create shader program
    m_program = glCreateProgram();
    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);
    glLinkProgram(m_program);
    
    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_program, 512, nullptr, infoLog);
        LOG_WARNING("Shader program linking failed: ", infoLog);
        LOG_INFO("Falling back to fixed function pipeline");
        m_program = 0;
        return true; // Continue without shaders
    }
    
    // Get uniform locations
    m_textureUniform = glGetUniformLocation(m_program, "videoTexture");
    m_isYUVUniform = glGetUniformLocation(m_program, "isYUV");
    
    LOG_INFO("Shaders compiled and linked successfully");
    return true;
}

bool OpenGLRenderer::EnableDebugOutput() {
    if (!GLAD_GL_VERSION_4_3 && !GLAD_GL_KHR_debug) {
        LOG_INFO("OpenGL debug output not available");
        return false;
    }
    
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(DebugCallback, nullptr);
    
    // Filter out notification messages
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
    
    LOG_INFO("OpenGL debug output enabled");
    return true;
}

void GLAPIENTRY OpenGLRenderer::DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
    // Ignore certain verbose info messages
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) {
        return;
    }
    
    std::string severityStr;
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         severityStr = "HIGH"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       severityStr = "MEDIUM"; break;
        case GL_DEBUG_SEVERITY_LOW:          severityStr = "LOW"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: severityStr = "NOTIFICATION"; break;
        default:                             severityStr = "UNKNOWN"; break;
    }
    
    std::string typeStr;
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               typeStr = "ERROR"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeStr = "DEPRECATED_BEHAVIOR"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typeStr = "UNDEFINED_BEHAVIOR"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         typeStr = "PORTABILITY"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         typeStr = "PERFORMANCE"; break;
        case GL_DEBUG_TYPE_OTHER:               typeStr = "OTHER"; break;
        default:                                typeStr = "UNKNOWN"; break;
    }
    
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        LOG_ERROR("OpenGL ", severityStr, " ", typeStr, ": ", message);
    } else if (severity == GL_DEBUG_SEVERITY_MEDIUM) {
        LOG_WARNING("OpenGL ", severityStr, " ", typeStr, ": ", message);
    } else {
        LOG_INFO("OpenGL ", severityStr, " ", typeStr, ": ", message);
    }
}

void OpenGLRenderer::CreateGeometry() {
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
    
    // Generate VAO
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
    
    // Generate VBO
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Generate EBO
    glGenBuffers(1, &m_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Set vertex attributes
    // Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLQuadVertex), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute (location = 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLQuadVertex), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    LOG_INFO("Created OpenGL 4.6 VAO/VBO geometry buffers");
}

void OpenGLRenderer::CreateTexture() {
    // Generate texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Create initial texture with placeholder data (will be updated during Present)
    // Always create RGBA texture - CUDA conversion will handle YUV to RGBA
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_textureWidth, m_textureHeight, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, nullptr);
    
    // Check for OpenGL errors
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        LOG_ERROR("OpenGL error creating texture: ", error);
        if (error == GL_OUT_OF_MEMORY) {
            LOG_ERROR("OpenGL out of memory - texture size ", m_textureWidth, "x", m_textureHeight, " may be too large");
        } else if (error == GL_INVALID_VALUE) {
            LOG_ERROR("Invalid texture dimensions: ", m_textureWidth, "x", m_textureHeight);
        }
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool OpenGLRenderer::ResizeTextureIfNeeded(int newWidth, int newHeight) {
    if (newWidth == m_textureWidth && newHeight == m_textureHeight) {
        return true; // No resize needed
    }
    
#if HAVE_CUDA
    // Clean up existing CUDA interop before resizing texture
    if (m_cudaInterop && m_cudaTextureResource) {
        CleanupCudaInterop();
    }
#endif
    
    // Update texture dimensions only (NOT window dimensions)
    m_textureWidth = newWidth;
    m_textureHeight = newHeight;
    
    // Recreate the OpenGL texture with new dimensions
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }
    
    CreateTexture();
    
#if HAVE_CUDA
    // Reinitialize CUDA interop with the resized texture
    if (!InitializeCudaInterop()) {
        LOG_ERROR("Failed to reinitialize CUDA interop after texture resize");
        return false;
    }
    
    if (!TestCudaInterop()) {
        LOG_ERROR("CUDA interop test failed after texture resize");
        CleanupCudaInterop();
        return false;
    }
#endif
    
    return true;
}

void OpenGLRenderer::SetupRenderState(bool isYUV) {
    // Use OpenGL 4.6 Core Profile shader program
    glUseProgram(m_program);
    
    // Bind texture to unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Set texture uniform
    glUniform1i(m_textureUniform, 0);
    
    // Set YUV flag uniform
    glUniform1i(m_isYUVUniform, isYUV ? 1 : 0);
    
    // Bind VAO
    glBindVertexArray(m_vao);
}

void OpenGLRenderer::DrawQuad() {
    // Use OpenGL 4.6 Core Profile VAO/VBO rendering
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

bool OpenGLRenderer::PresentSoftwareTexture(const RenderTexture& texture) {
    if (texture.type != TextureType::Software || !texture.software.data) {
        return false;
    }
    
    // Check for dimension mismatch and resize texture if needed
    if (texture.width != m_textureWidth || texture.height != m_textureHeight) {
        if (!ResizeTextureIfNeeded(texture.width, texture.height)) {
            LOG_ERROR("Failed to resize OpenGL texture to match video dimensions");
            return false;
        }
    }
    
    // Update texture with new frame data
    glBindTexture(GL_TEXTURE_2D, m_texture);
    
    // Set pixel store parameters for proper pitch handling
    glPixelStorei(GL_UNPACK_ROW_LENGTH, texture.software.pitch / 4); // Assuming RGBA format (4 bytes per pixel)
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture.width, texture.height, GL_RGBA, GL_UNSIGNED_BYTE, texture.software.data);
    
    // Reset pixel store
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    
    // Use render pass pipeline if available and enabled
    if (m_renderPassPipeline && m_renderPassPipeline->IsEnabled()) {
        // Create OpenGL render pass context
        OpenGLRenderPassContext context;
        // Use the frame delta calculated in Present() method
        auto frameDelta = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_lastFrameTime);
        context.deltaTime = frameDelta.count();
        context.totalTime = m_totalTime;
        context.frameNumber = m_frameNumber;
        context.inputWidth = m_textureWidth;
        context.inputHeight = m_textureHeight;
        context.outputWidth = m_width;
        context.outputHeight = m_height;
        context.isYUV = texture.isYUV;
        context.uvTexture = 0; // No separate UV texture for software rendering
        context.textureFormat = GL_RGBA8;
        context.textureInternalFormat = GL_RGBA8;
        context.textureDataFormat = GL_RGBA;
        context.textureDataType = GL_UNSIGNED_BYTE;
        context.flipY = false; // Will be set by pipeline based on output target
        
        // Execute pipeline with our texture as input and default framebuffer as output
#ifdef TRACY_ENABLE
        PROFILE_GPU_OPENGL_ZONE("RenderPassPipeline");
#endif
        return m_renderPassPipeline->Execute(context, m_texture, 0, 0);
    } else {
        // Direct rendering without render passes (fallback to original behavior)
        SetupRenderState(texture.isYUV);
        DrawQuad();
        return true;
    }
}

void OpenGLRenderer::Reset() {
    m_initialized = false;
    
    // Clean up render pass pipeline
    m_renderPassPipeline.reset();
    
#if HAVE_CUDA
    // Clean up CUDA interop
    CleanupCudaInterop();
#endif
    
    // Delete OpenGL resources
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }
    
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    
    if (m_ebo) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }
    
    if (m_program) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
    
    if (m_vertexShader) {
        glDeleteShader(m_vertexShader);
        m_vertexShader = 0;
    }
    
    if (m_fragmentShader) {
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
    m_textureWidth = 0;
    m_textureHeight = 0;
    m_textureUniform = -1;
    m_isYUVUniform = -1;
    m_glMajorVersion = 0;
    m_glMinorVersion = 0;
    m_coreProfile = false;
    m_debugContext = false;
    wglCreateContextAttribsARB = nullptr;
    
    // Reset frame timing
    m_frameNumber = 0;
    m_totalTime = 0.0f;
}

#if HAVE_CUDA

bool OpenGLRenderer::InitializeCudaInterop() {
    try {
        m_cudaInterop = std::make_unique<CudaOpenGLInterop>();
        if (!m_cudaInterop->Initialize()) {
            LOG_ERROR("Failed to initialize CUDA/OpenGL interop");
            m_cudaInterop.reset();
            return false;
        }
        
        // Register the main OpenGL texture with CUDA for interop
        if (!m_cudaInterop->RegisterTexture(m_texture, &m_cudaTextureResource)) {
            LOG_ERROR("Failed to register OpenGL texture with CUDA");
            m_cudaInterop.reset();
            m_cudaTextureResource = nullptr;
            return false;
        }
        
        LOG_INFO("CUDA/OpenGL interop initialized successfully - texture registered");
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception initializing CUDA interop: ", e.what());
        m_cudaInterop.reset();
        m_cudaTextureResource = nullptr;
        return false;
    }
}

void OpenGLRenderer::CleanupCudaInterop() {
    if (m_cudaInterop) {
        if (m_cudaTextureResource) {
            if (!m_cudaInterop->UnregisterTexture(m_cudaTextureResource)) {
                LOG_WARNING("Failed to unregister CUDA texture resource during cleanup");
            }
            m_cudaTextureResource = nullptr;
        }
        m_cudaInterop->Cleanup();
        m_cudaInterop.reset();
    }
}

bool OpenGLRenderer::TestCudaInterop() {
    if (!m_cudaInterop || !m_cudaInterop->IsInitialized()) {
        return false;
    }
    
    if (!m_cudaTextureResource) {
        LOG_ERROR("OpenGL texture not registered with CUDA");
        return false;
    }

    // This validates that the texture registration and interop system works
    return m_cudaInterop->TestResourceMapping(m_cudaTextureResource, nullptr);
}

bool OpenGLRenderer::IsCudaInteropAvailable() const {
    return m_cudaInterop && m_cudaInterop->IsInitialized();
}

bool OpenGLRenderer::SupportsCudaInterop() const {
#if HAVE_CUDA
    return true; // OpenGL renderer supports CUDA when compiled with CUDA
#else
    return false; // CUDA not compiled in
#endif
}

bool OpenGLRenderer::PresentCudaTexture(const RenderTexture& texture) {
    if (texture.type != TextureType::CUDA) {
        return false;
    }
    
    if (!m_cudaInterop || !m_cudaInterop->IsInitialized() || !m_cudaTextureResource) {
        LOG_ERROR("CUDA interop not available or texture not registered");
        return false;
    }
    
    // Check for dimension mismatch and resize texture if needed
    if (texture.width != m_textureWidth || texture.height != m_textureHeight) {
        if (!ResizeTextureIfNeeded(texture.width, texture.height)) {
            LOG_ERROR("Failed to resize OpenGL texture to match video dimensions");
            return false;
        }
    }
    
    // Copy CUDA device memory to the pre-registered OpenGL texture
    // Use YUV conversion if the texture is in YUV format, otherwise direct copy
    bool copySuccess = false;
    if (texture.isYUV) {
        copySuccess = m_cudaInterop->CopyYuvToTexture(
            texture.cuda.devicePtr, 
            texture.cuda.pitch, 
            m_cudaTextureResource, 
            texture.width, 
            texture.height);
    } else {
        copySuccess = m_cudaInterop->CopyDeviceToTexture(
            texture.cuda.devicePtr, 
            texture.cuda.pitch, 
            m_cudaTextureResource, 
            texture.width, 
            texture.height);
    }
    
    if (!copySuccess) {
        LOG_ERROR("Failed to copy CUDA device memory to OpenGL texture");
        return false;
    }
    
    // Use render pass pipeline if available and enabled
    if (m_renderPassPipeline && m_renderPassPipeline->IsEnabled()) {
        // Create OpenGL render pass context
        OpenGLRenderPassContext context;
        // Use the frame delta calculated in Present() method
        auto frameDelta = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_lastFrameTime);
        context.deltaTime = frameDelta.count();
        context.totalTime = m_totalTime;
        context.frameNumber = m_frameNumber;
        context.inputWidth = m_textureWidth;
        context.inputHeight = m_textureHeight;
        context.outputWidth = m_width;
        context.outputHeight = m_height;
        context.isYUV = false; // Always false since CUDA converted YUV to RGBA
        context.uvTexture = 0; // No separate UV texture for CUDA interop
        context.textureFormat = GL_RGBA8;
        context.textureInternalFormat = GL_RGBA8;
        context.textureDataFormat = GL_RGBA;
        context.textureDataType = GL_UNSIGNED_BYTE;
        context.flipY = false; // Will be set by pipeline based on output target
        
        // Execute pipeline with our texture as input and default framebuffer as output
#ifdef TRACY_ENABLE
        PROFILE_GPU_OPENGL_ZONE("RenderPassPipeline");
#endif
        return m_renderPassPipeline->Execute(context, m_texture, 0, 0);
    } else {
        // Direct rendering without render passes (fallback to original behavior)
        SetupRenderState(false); // Always false since CUDA converted YUV to RGBA
        DrawQuad();
        return true;
    }
}

#else // HAVE_CUDA

bool OpenGLRenderer::IsCudaInteropAvailable() const {
    return false; // CUDA not available
}

bool OpenGLRenderer::SupportsCudaInterop() const {
    return false; // CUDA not available
}

#endif // HAVE_CUDA

bool OpenGLRenderer::CaptureFramebuffer(uint8_t* outputBuffer, size_t bufferSize, int& width, int& height) {
    if (!m_initialized || !outputBuffer) {
        LOG_ERROR("OpenGLRenderer: Cannot capture framebuffer - renderer not initialized or invalid buffer");
        return false;
    }
    
    width = m_width;
    height = m_height;
    
    // Check buffer size (RGBA8 = 4 bytes per pixel)
    size_t requiredSize = static_cast<size_t>(width * height * 4);
    if (bufferSize < requiredSize) {
        LOG_ERROR("OpenGLRenderer: Buffer too small for framebuffer capture. Required: ", requiredSize, ", provided: ", bufferSize);
        return false;
    }
    
    // Bind the default framebuffer (back buffer)
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    
    // Check for OpenGL errors before reading
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        LOG_ERROR("OpenGLRenderer: OpenGL error before framebuffer capture: 0x", std::hex, error);
        return false;
    }
    
    // Read pixels from the framebuffer
    // Note: OpenGL reads from bottom-left, so we need to flip vertically
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, outputBuffer);
    
    // Check for errors after reading
    error = glGetError();
    if (error != GL_NO_ERROR) {
        LOG_ERROR("OpenGLRenderer: OpenGL error during framebuffer capture: 0x", std::hex, error);
        return false;
    }
    
    // Flip the image vertically since OpenGL coordinates are bottom-left origin
    // but most image processing expects top-left origin
    size_t rowSize = width * 4;
    std::vector<uint8_t> tempRow(rowSize);
    
    for (int row = 0; row < height / 2; ++row) {
        uint8_t* topRow = outputBuffer + row * rowSize;
        uint8_t* bottomRow = outputBuffer + (height - 1 - row) * rowSize;
        
        // Swap top and bottom rows
        memcpy(tempRow.data(), topRow, rowSize);
        memcpy(topRow, bottomRow, rowSize);
        memcpy(bottomRow, tempRow.data(), rowSize);
    }
    
    LOG_DEBUG("OpenGLRenderer: Successfully captured framebuffer (", width, "x", height, ")");
    return true;
}