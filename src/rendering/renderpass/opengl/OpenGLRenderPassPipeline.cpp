#include "OpenGLRenderPassPipeline.h"
#include "../RenderPassConfig.h"
#include "OpenGLRenderPassResources.h"
#include "core/Logger.h"
#include <algorithm>

OpenGLRenderPassPipeline::OpenGLRenderPassPipeline()
    : m_enabled(true)
    , m_yuvToRgbPass(nullptr)
    , m_intermediateFramebuffer{0, 0}
    , m_intermediateTexture{0, 0}
    , m_textureWidth(0)
    , m_textureHeight(0)
    , m_textureFormat(GL_RGBA8)
    , m_copyProgram(0)
    , m_copyVertexShader(0)
    , m_copyFragmentShader(0)
    , m_initialized(false) {
}

OpenGLRenderPassPipeline::~OpenGLRenderPassPipeline() {
    Cleanup();
}

bool OpenGLRenderPassPipeline::Initialize() {
    if (m_initialized) {
        return true;
    }
    
    // Initialize shared resources
    OpenGLRenderPassResources* resources = OpenGLRenderPassResources::GetInstance();
    if (!resources) {
        LOG_ERROR("OpenGLRenderPassPipeline: Failed to initialize shared resources");
        return false;
    }
    
    // Create copy resources
    if (!CreateCopyResources()) {
        LOG_ERROR("OpenGLRenderPassPipeline: Failed to create copy resources");
        return false;
    }
    
    m_initialized = true;
    LOG_INFO("OpenGLRenderPassPipeline initialized successfully");
    return true;
}

void OpenGLRenderPassPipeline::Cleanup() {
    // Clear passes
    m_passes.clear();
    m_yuvToRgbPass.reset();
    
    // Clean up intermediate framebuffers
    for (int i = 0; i < 2; i++) {
        if (m_intermediateTexture[i] != 0) {
            glDeleteTextures(1, &m_intermediateTexture[i]);
            m_intermediateTexture[i] = 0;
        }
        if (m_intermediateFramebuffer[i] != 0) {
            glDeleteFramebuffers(1, &m_intermediateFramebuffer[i]);
            m_intermediateFramebuffer[i] = 0;
        }
    }
    
    // Clean up copy resources
    if (m_copyProgram != 0) {
        glDeleteProgram(m_copyProgram);
        m_copyProgram = 0;
    }
    if (m_copyVertexShader != 0) {
        glDeleteShader(m_copyVertexShader);
        m_copyVertexShader = 0;
    }
    if (m_copyFragmentShader != 0) {
        glDeleteShader(m_copyFragmentShader);
        m_copyFragmentShader = 0;
    }
    
    m_textureWidth = 0;
    m_textureHeight = 0;
    m_initialized = false;
}

void OpenGLRenderPassPipeline::AddPass(std::unique_ptr<IRenderPass> pass) {
    // Try to cast to OpenGLRenderPass
    auto* openglPass = dynamic_cast<OpenGLRenderPass*>(pass.get());
    if (openglPass) {
        // Transfer ownership to OpenGL-specific container
        pass.release();
        AddOpenGLPass(std::unique_ptr<OpenGLRenderPass>(openglPass));
    } else {
        LOG_ERROR("OpenGLRenderPassPipeline: Cannot add non-OpenGL render pass");
    }
}

void OpenGLRenderPassPipeline::AddOpenGLPass(std::unique_ptr<OpenGLRenderPass> pass) {
    m_passes.push_back(std::move(pass));
}

bool OpenGLRenderPassPipeline::Execute(const OpenGLRenderPassContext& context,
                                      GLuint inputTexture,
                                      GLuint outputFramebuffer,
                                      GLuint outputTexture) {
    if (!m_enabled || m_passes.empty()) {
        // Pipeline disabled or no passes - direct copy
        return DirectCopy(inputTexture, outputFramebuffer, outputTexture,
                         context.inputWidth, context.inputHeight);
    }
    
    // Ensure intermediate framebuffers are ready
    if (!EnsureIntermediateFramebuffers(context.inputWidth, context.inputHeight)) {
        LOG_ERROR("OpenGLRenderPassPipeline: Failed to prepare intermediate framebuffers");
        return false;
    }
    
    // Count enabled passes
    size_t enabledPasses = 0;
    for (const auto& pass : m_passes) {
        if (pass->IsEnabled()) {
            enabledPasses++;
        }
    }
    
    if (enabledPasses == 0) {
        // No enabled passes - direct copy
        return DirectCopy(inputTexture, outputFramebuffer, outputTexture,
                         context.inputWidth, context.inputHeight);
    }
    
    GLuint currentInputTexture = inputTexture;
    GLuint currentOutputFramebuffer = outputFramebuffer;
    GLuint currentOutputTexture = outputTexture;
    
    size_t passIndex = 0;
    for (const auto& pass : m_passes) {
        if (!pass->IsEnabled()) {
            continue;
        }
        
        // Determine output target
        bool isLastPass = (passIndex == enabledPasses - 1);
        if (isLastPass) {
            // Last pass renders to final output
            currentOutputFramebuffer = outputFramebuffer;
            currentOutputTexture = outputTexture;
        } else {
            // Intermediate pass renders to ping-pong buffer
            int bufferIndex = passIndex % 2;
            currentOutputFramebuffer = m_intermediateFramebuffer[bufferIndex];
            currentOutputTexture = m_intermediateTexture[bufferIndex];
        }
        
        // Execute pass
        if (!pass->Execute(context, currentInputTexture, currentOutputFramebuffer, currentOutputTexture)) {
            LOG_ERROR("OpenGLRenderPassPipeline: Pass '", pass->GetName(), "' failed");
            return false;
        }
        
        // Next pass uses current output as input
        if (!isLastPass) {
            currentInputTexture = currentOutputTexture;
        }
        
        passIndex++;
    }
    
    return true;
}

bool OpenGLRenderPassPipeline::SetPassEnabled(const std::string& passName, bool enabled) {
    for (const auto& pass : m_passes) {
        if (pass->GetName() == passName) {
            pass->SetEnabled(enabled);
            return true;
        }
    }
    return false;
}

IRenderPass* OpenGLRenderPassPipeline::GetPass(const std::string& passName) const {
    for (const auto& pass : m_passes) {
        if (pass->GetName() == passName) {
            return pass.get();
        }
    }
    return nullptr;
}

bool OpenGLRenderPassPipeline::UpdatePassParameters(const std::string& passName,
                                                   const std::map<std::string, RenderPassParameter>& parameters) {
    for (const auto& pass : m_passes) {
        if (pass->GetName() == passName) {
            pass->UpdateParameters(parameters);
            return true;
        }
    }
    return false;
}

bool OpenGLRenderPassPipeline::EnsureIntermediateFramebuffers(int width, int height) {
    if (m_textureWidth == width && m_textureHeight == height && 
        m_intermediateFramebuffer[0] != 0 && m_intermediateFramebuffer[1] != 0) {
        return true; // Already correctly sized
    }
    
    // Clean up existing framebuffers
    for (int i = 0; i < 2; i++) {
        if (m_intermediateTexture[i] != 0) {
            glDeleteTextures(1, &m_intermediateTexture[i]);
            m_intermediateTexture[i] = 0;
        }
        if (m_intermediateFramebuffer[i] != 0) {
            glDeleteFramebuffers(1, &m_intermediateFramebuffer[i]);
            m_intermediateFramebuffer[i] = 0;
        }
    }
    
    // Create new framebuffers
    for (int i = 0; i < 2; i++) {
        if (!CreateIntermediateFramebuffer(width, height, m_textureFormat,
                                          m_intermediateFramebuffer[i], m_intermediateTexture[i])) {
            LOG_ERROR("OpenGLRenderPassPipeline: Failed to create intermediate framebuffer ", i);
            return false;
        }
    }
    
    m_textureWidth = width;
    m_textureHeight = height;
    return true;
}

bool OpenGLRenderPassPipeline::CreateIntermediateFramebuffer(int width, int height, GLenum format,
                                                            GLuint& framebuffer, GLuint& texture) {
    // Create texture
    glGenTextures(1, &texture);
    if (texture == 0) {
        return false;
    }
    
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Create framebuffer
    glGenFramebuffers(1, &framebuffer);
    if (framebuffer == 0) {
        glDeleteTextures(1, &texture);
        texture = 0;
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOG_ERROR("OpenGLRenderPassPipeline: Framebuffer not complete: ", status);
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteTextures(1, &texture);
        framebuffer = 0;
        texture = 0;
        return false;
    }
    
    return true;
}

bool OpenGLRenderPassPipeline::DirectCopy(GLuint inputTexture, GLuint outputFramebuffer, GLuint outputTexture,
                                         int width, int height) {
    if (m_copyProgram == 0) {
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
    glViewport(0, 0, width, height);
    
    glUseProgram(m_copyProgram);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTexture);
    
    GLint textureLocation = glGetUniformLocation(m_copyProgram, "inputTexture");
    if (textureLocation != -1) {
        glUniform1i(textureLocation, 0);
    }
    
    // Render fullscreen quad
    OpenGLRenderPassResources* resources = OpenGLRenderPassResources::GetInstance();
    if (resources) {
        resources->RenderFullscreenQuad();
    }
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    
    return true;
}

bool OpenGLRenderPassPipeline::CreateCopyResources() {
    // Vertex shader source
    const char* vertexShaderSource = R"(
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
    
    // Fragment shader source
    const char* fragmentShaderSource = R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D inputTexture;

void main()
{
    FragColor = texture(inputTexture, TexCoord);
}
)";
    
    // Compile vertex shader
    m_copyVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(m_copyVertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(m_copyVertexShader);
    
    GLint success;
    glGetShaderiv(m_copyVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(m_copyVertexShader, 512, nullptr, infoLog);
        LOG_ERROR("OpenGLRenderPassPipeline: Copy vertex shader compilation failed: ", infoLog);
        return false;
    }
    
    // Compile fragment shader
    m_copyFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(m_copyFragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(m_copyFragmentShader);
    
    glGetShaderiv(m_copyFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(m_copyFragmentShader, 512, nullptr, infoLog);
        LOG_ERROR("OpenGLRenderPassPipeline: Copy fragment shader compilation failed: ", infoLog);
        return false;
    }
    
    // Create and link program
    m_copyProgram = glCreateProgram();
    glAttachShader(m_copyProgram, m_copyVertexShader);
    glAttachShader(m_copyProgram, m_copyFragmentShader);
    glLinkProgram(m_copyProgram);
    
    glGetProgramiv(m_copyProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(m_copyProgram, 512, nullptr, infoLog);
        LOG_ERROR("OpenGLRenderPassPipeline: Copy program linking failed: ", infoLog);
        return false;
    }
    
    return true;
}