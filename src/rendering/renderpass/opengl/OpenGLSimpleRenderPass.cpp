#include "OpenGLSimpleRenderPass.h"
#include "../RenderPassConfig.h"
#include "OpenGLRenderPassResources.h"
#include "core/Logger.h"
#include <fstream>
#include <sstream>

OpenGLSimpleRenderPass::~OpenGLSimpleRenderPass() {
    Cleanup();
}

bool OpenGLSimpleRenderPass::Initialize(const RenderPassConfig& config) {
    // Load shader configuration
    std::string shaderName = config.GetString("shader");
    if (shaderName.empty()) {
        // Try loading shaders from virtual methods first
        if (!LoadShadersFromSource()) {
            LOG_ERROR("OpenGLRenderPass '", m_name, "': No shader specified and virtual shader sources not provided");
            return false;
        }
    } else {
        // Check if it's a built-in shader or file path
        if (shaderName.find('.') == std::string::npos) {
            // Built-in shader - for now, fall back to virtual methods
            m_shaderName = shaderName;
            if (!LoadShadersFromSource()) {
                LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to load built-in shader '", shaderName, "'");
                return false;
            }
        } else {
            // External shader file
            m_vertexShaderPath = "src/rendering/shaders/opengl/FullscreenQuad.glsl";
            m_fragmentShaderPath = "src/rendering/shaders/opengl/" + shaderName;
            
            if (!LoadVertexShader(m_vertexShaderPath) || !LoadFragmentShader(m_fragmentShaderPath)) {
                LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to load shader files");
                return false;
            }
            
            if (!LinkProgram()) {
                LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to link shader program");
                return false;
            }
        }
    }
    
    // Initialize shared resources
    if (!InitializeSharedResources()) {
        LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to initialize shared resources");
        return false;
    }
    
    // Load parameters from config
    auto parameters = config.GetAllParameters();
    if (!parameters.empty()) {
        UpdateParameters(parameters);
    }
    
    // Create uniform buffer if needed
    m_uniformBufferSize = GetUniformBufferSize();
    if (m_uniformBufferSize > 0) {
        if (!CreateUniformBuffer(m_uniformBufferSize)) {
            LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to create uniform buffer");
            return false;
        }
        m_uniformBufferData.resize(m_uniformBufferSize);
        m_uniformBufferDirty = true;
    }
    
    LOG_INFO("OpenGLRenderPass '", m_name, "' initialized successfully");
    return true;
}

void OpenGLSimpleRenderPass::Cleanup() {
    if (m_uniformBuffer != 0) {
        glDeleteBuffers(1, &m_uniformBuffer);
        m_uniformBuffer = 0;
    }
    
    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
    
    if (m_vertexShader != 0) {
        glDeleteShader(m_vertexShader);
        m_vertexShader = 0;
    }
    
    if (m_fragmentShader != 0) {
        glDeleteShader(m_fragmentShader);
        m_fragmentShader = 0;
    }
    
    // Note: Shared resources (geometry, samplers, render states) are managed by OpenGLRenderPassResources
}

bool OpenGLSimpleRenderPass::Execute(const OpenGLRenderPassContext& context,
                                   GLuint inputTexture,
                                   GLuint outputFramebuffer,
                                   GLuint outputTexture) {
    if (!m_enabled || m_program == 0) {
        return false;
    }
    
    // Bind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, outputFramebuffer);
    
    // Set viewport
    glViewport(0, 0, context.inputWidth, context.inputHeight);
    
    // Use our shader program
    glUseProgram(m_program);
    
    // Bind input texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTexture);
    GLint textureLocation = glGetUniformLocation(m_program, "videoTexture");
    if (textureLocation != -1) {
        glUniform1i(textureLocation, 0);
    }
    
    // Bind YUV texture if present
    if (context.uvTexture != 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, context.uvTexture);
        GLint uvTextureLocation = glGetUniformLocation(m_program, "uvTexture");
        if (uvTextureLocation != -1) {
            glUniform1i(uvTextureLocation, 1);
        }
    }
    
    // Set YUV flag
    GLint isYUVLocation = glGetUniformLocation(m_program, "isYUV");
    if (isYUVLocation != -1) {
        glUniform1i(isYUVLocation, context.isYUV ? 1 : 0);
    }
    
    // Update uniform buffer if needed
    if (m_uniformBuffer != 0 && m_uniformBufferDirty) {
        if (!UpdateUniformBuffer(context)) {
            LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to update uniform buffer");
            return false;
        }
    }
    
    // Get shared resources and render fullscreen quad
    OpenGLRenderPassResources* resources = OpenGLRenderPassResources::GetInstance();
    if (resources) {
        resources->RenderFullscreenQuad();
    } else {
        LOG_ERROR("OpenGLRenderPass '", m_name, "': Shared resources not available");
        return false;
    }
    
    // Unbind textures
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

void OpenGLSimpleRenderPass::UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) {
    bool changed = false;
    for (const auto& [name, value] : parameters) {
        if (m_parameters[name] != value) {
            m_parameters[name] = value;
            changed = true;
        }
    }
    
    if (changed) {
        PackParameters();
    }
}

std::string OpenGLSimpleRenderPass::GetVertexShaderSource() const {
    return R"(
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
}

std::string OpenGLSimpleRenderPass::GetFragmentShaderSource() const {
    return R"(
#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform bool isYUV;

void main()
{
    FragColor = texture(videoTexture, TexCoord);
}
)";
}

size_t OpenGLSimpleRenderPass::GetUniformBufferSize() const {
    return 0; // No uniform buffer by default
}

void OpenGLSimpleRenderPass::PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context) {
    // Default implementation does nothing
}

bool OpenGLSimpleRenderPass::LoadVertexShader(const std::string& shaderPath) {
    std::string source = LoadShaderFile(shaderPath);
    if (source.empty()) {
        return false;
    }
    return CompileShader(GL_VERTEX_SHADER, source, m_vertexShader);
}

bool OpenGLSimpleRenderPass::LoadFragmentShader(const std::string& shaderPath) {
    std::string source = LoadShaderFile(shaderPath);
    if (source.empty()) {
        return false;
    }
    return CompileShader(GL_FRAGMENT_SHADER, source, m_fragmentShader);
}

bool OpenGLSimpleRenderPass::LoadShadersFromSource() {
    std::string vertexSource = GetVertexShaderSource();
    std::string fragmentSource = GetFragmentShaderSource();
    
    if (vertexSource.empty() || fragmentSource.empty()) {
        return false;
    }
    
    if (!CompileShader(GL_VERTEX_SHADER, vertexSource, m_vertexShader)) {
        return false;
    }
    
    if (!CompileShader(GL_FRAGMENT_SHADER, fragmentSource, m_fragmentShader)) {
        return false;
    }
    
    return LinkProgram();
}

bool OpenGLSimpleRenderPass::CreateUniformBuffer(size_t size) {
    glGenBuffers(1, &m_uniformBuffer);
    if (m_uniformBuffer == 0) {
        return false;
    }
    
    glBindBuffer(GL_UNIFORM_BUFFER, m_uniformBuffer);
    glBufferData(GL_UNIFORM_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    return true;
}

bool OpenGLSimpleRenderPass::UpdateUniformBuffer(const OpenGLRenderPassContext& context) {
    if (m_uniformBuffer == 0 || m_uniformBufferData.empty()) {
        return false;
    }
    
    // Pack parameters into buffer
    PackUniformBuffer(m_uniformBufferData.data(), context);
    
    // Update GPU buffer
    glBindBuffer(GL_UNIFORM_BUFFER, m_uniformBuffer);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, m_uniformBufferData.size(), m_uniformBufferData.data());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    
    m_uniformBufferDirty = false;
    return true;
}

void OpenGLSimpleRenderPass::PackParameters() {
    // Mark buffer as dirty - actual packing happens in PackUniformBuffer
    m_uniformBufferDirty = true;
}

bool OpenGLSimpleRenderPass::CreateFullscreenQuad() {
    // Delegate to shared resources
    return InitializeSharedResources();
}

void OpenGLSimpleRenderPass::RenderFullscreenQuad() {
    // Delegate to shared resources
    OpenGLRenderPassResources* resources = OpenGLRenderPassResources::GetInstance();
    if (resources) {
        resources->RenderFullscreenQuad();
    }
}

bool OpenGLSimpleRenderPass::InitializeSharedResources() {
    // Get or initialize shared resources
    OpenGLRenderPassResources* resources = OpenGLRenderPassResources::GetInstance();
    return resources != nullptr;
}

bool OpenGLSimpleRenderPass::CompileShader(GLenum shaderType, const std::string& source, GLuint& shaderOut) {
    shaderOut = glCreateShader(shaderType);
    if (shaderOut == 0) {
        return false;
    }
    
    const char* sourceCStr = source.c_str();
    glShaderSource(shaderOut, 1, &sourceCStr, nullptr);
    glCompileShader(shaderOut);
    
    GLint success;
    glGetShaderiv(shaderOut, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shaderOut, 512, nullptr, infoLog);
        LOG_ERROR("OpenGLRenderPass '", m_name, "': Shader compilation failed: ", infoLog);
        glDeleteShader(shaderOut);
        shaderOut = 0;
        return false;
    }
    
    return true;
}

bool OpenGLSimpleRenderPass::LinkProgram() {
    if (m_vertexShader == 0 || m_fragmentShader == 0) {
        return false;
    }
    
    m_program = glCreateProgram();
    if (m_program == 0) {
        return false;
    }
    
    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_fragmentShader);
    glLinkProgram(m_program);
    
    GLint success;
    glGetProgramiv(m_program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(m_program, 512, nullptr, infoLog);
        LOG_ERROR("OpenGLRenderPass '", m_name, "': Program linking failed: ", infoLog);
        glDeleteProgram(m_program);
        m_program = 0;
        return false;
    }
    
    return true;
}

std::string OpenGLSimpleRenderPass::LoadShaderFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("OpenGLRenderPass '", m_name, "': Failed to open shader file: ", filename);
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}