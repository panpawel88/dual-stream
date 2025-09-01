#include "OpenGLRenderPassResources.h"
#include "core/Logger.h"

// Static instance
OpenGLRenderPassResources* OpenGLRenderPassResources::s_instance = nullptr;

OpenGLRenderPassResources* OpenGLRenderPassResources::GetInstance() {
    if (!s_instance) {
        s_instance = new OpenGLRenderPassResources();
        if (!s_instance->Initialize()) {
            delete s_instance;
            s_instance = nullptr;
            LOG_ERROR("Failed to initialize OpenGLRenderPassResources");
        }
    }
    return s_instance;
}

void OpenGLRenderPassResources::DestroyInstance() {
    if (s_instance) {
        s_instance->Cleanup();
        delete s_instance;
        s_instance = nullptr;
    }
}

OpenGLRenderPassResources::OpenGLRenderPassResources()
    : m_vao(0), m_vbo(0), m_ebo(0), m_linearSampler(0), m_pointSampler(0), m_initialized(false) {
}

OpenGLRenderPassResources::~OpenGLRenderPassResources() {
    Cleanup();
}

bool OpenGLRenderPassResources::Initialize() {
    if (m_initialized) {
        return true;
    }
    
    // Create fullscreen quad
    if (!CreateFullscreenQuad()) {
        LOG_ERROR("OpenGLRenderPassResources: Failed to create fullscreen quad");
        return false;
    }
    
    // Create samplers
    if (!CreateSamplers()) {
        LOG_ERROR("OpenGLRenderPassResources: Failed to create samplers");
        return false;
    }
    
    // Setup render states
    SetupRenderStates();
    
    m_initialized = true;
    LOG_INFO("OpenGLRenderPassResources initialized successfully");
    return true;
}

void OpenGLRenderPassResources::Cleanup() {
    if (m_linearSampler != 0) {
        glDeleteSamplers(1, &m_linearSampler);
        m_linearSampler = 0;
    }
    
    if (m_pointSampler != 0) {
        glDeleteSamplers(1, &m_pointSampler);
        m_pointSampler = 0;
    }
    
    if (m_ebo != 0) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }
    
    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    
    m_initialized = false;
}

void OpenGLRenderPassResources::RenderFullscreenQuad() {
    if (!m_initialized || m_vao == 0) {
        return;
    }
    
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

bool OpenGLRenderPassResources::CreateFullscreenQuad() {
    // Fullscreen quad vertices (position + texture coordinates)
    // Matched to main OpenGL renderer vertex order and texture coordinates
    float vertices[] = {
        // positions        // texture coords
        -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, // top left
         1.0f,  1.0f, 0.0f, 1.0f, 0.0f, // top right
         1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f  // bottom left
    };
    
    unsigned int indices[] = {
        0, 1, 2, // first triangle
        0, 2, 3  // second triangle
    };
    
    // Create VAO
    glGenVertexArrays(1, &m_vao);
    if (m_vao == 0) {
        return false;
    }
    
    // Create VBO
    glGenBuffers(1, &m_vbo);
    if (m_vbo == 0) {
        return false;
    }
    
    // Create EBO
    glGenBuffers(1, &m_ebo);
    if (m_ebo == 0) {
        return false;
    }
    
    // Bind VAO
    glBindVertexArray(m_vao);
    
    // Bind and fill VBO
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // Bind and fill EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Unbind VAO (optional)
    glBindVertexArray(0);
    
    return true;
}

bool OpenGLRenderPassResources::CreateSamplers() {
    // Create linear sampler
    glGenSamplers(1, &m_linearSampler);
    if (m_linearSampler == 0) {
        return false;
    }
    
    glSamplerParameteri(m_linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(m_linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(m_linearSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(m_linearSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Create point sampler
    glGenSamplers(1, &m_pointSampler);
    if (m_pointSampler == 0) {
        return false;
    }
    
    glSamplerParameteri(m_pointSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(m_pointSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(m_pointSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(m_pointSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    return true;
}

void OpenGLRenderPassResources::SetupRenderStates() {
    // Set default render states for render passes
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
}