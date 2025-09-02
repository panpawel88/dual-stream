#include "OpenGLToastRenderer.h"
#include "ui/ToastManager.h"
#include "core/Logger.h"
#include <algorithm>
#include <cmath>

// Simple 8x8 bitmap font data for basic ASCII characters (32-126)
// Each character is represented as 8 bytes (8x8 pixels)
const unsigned char FONT_DATA[][8] = {
    // Space (32)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // ! (33)
    {0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00},
    // " (34)
    {0x66, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00},
    // # (35)
    {0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00},
    // $ (36)
    {0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00},
    // % (37)
    {0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00},
    // & (38)
    {0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00},
    // ' (39)
    {0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00},
    // More characters... (truncated for brevity, would include full ASCII set)
};

// Shader sources
const char* OpenGLToastRenderer::TEXT_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

const char* OpenGLToastRenderer::TEXT_FRAGMENT_SHADER = R"(
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;
uniform float alpha;

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, alpha) * sampled;
}
)";

const char* OpenGLToastRenderer::BACKGROUND_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec2 position;
out vec2 fragPos;

uniform mat4 projection;
uniform vec2 bgPosition;
uniform vec2 bgSize;

void main() {
    vec2 scaledPos = position * bgSize + bgPosition;
    gl_Position = projection * vec4(scaledPos, 0.0, 1.0);
    fragPos = position; // 0-1 range for fragment shader
}
)";

const char* OpenGLToastRenderer::BACKGROUND_FRAGMENT_SHADER = R"(
#version 330 core
in vec2 fragPos;
out vec4 color;

uniform vec4 bgColor;
uniform float cornerRadius;
uniform vec2 bgSize;

void main() {
    // Calculate distance from edges for rounded corners
    vec2 pixelPos = fragPos * bgSize;
    vec2 fromCorner = max(vec2(0.0), abs(pixelPos - bgSize * 0.5) - (bgSize * 0.5 - cornerRadius));
    float distance = length(fromCorner) - cornerRadius;
    
    // Smooth alpha based on distance
    float alpha = 1.0 - smoothstep(0.0, 1.0, distance);
    color = vec4(bgColor.rgb, bgColor.a * alpha);
}
)";

OpenGLToastRenderer::OpenGLToastRenderer()
    : m_initialized(false)
    , m_VAO(0)
    , m_VBO(0)
    , m_textShader(0)
    , m_backgroundShader(0)
    , m_viewportWidth(0)
    , m_viewportHeight(0) {
}

OpenGLToastRenderer::~OpenGLToastRenderer() {
    Cleanup();
}

bool OpenGLToastRenderer::Initialize(const ToastConfig& config) {
    if (m_initialized) {
        Cleanup();
    }
    
    m_config = config;
    UpdateViewportDimensions();
    
    if (!CreateShaders()) {
        LOG_ERROR("Failed to create shaders for OpenGL toast rendering");
        return false;
    }
    
    if (!CreateOpenGLResources()) {
        LOG_ERROR("Failed to create OpenGL resources for toast rendering");
        return false;
    }
    
    if (!InitializeFont()) {
        LOG_ERROR("Failed to initialize font for toast rendering");
        return false;
    }
    
    m_initialized = true;
    LOG_DEBUG("OpenGLToastRenderer initialized successfully");
    return true;
}

void OpenGLToastRenderer::RenderToast(const ToastMessage& toast) {
    if (!m_initialized || toast.currentAlpha <= 0.0f) {
        return;
    }
    
    // Update viewport dimensions
    UpdateViewportDimensions();
    
    // Calculate toast position and dimensions
    float x, y, width, height;
    CalculateToastPosition(toast.text, &x, &y, &width, &height);
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Create orthographic projection matrix
    float projection[16] = {
        2.0f / m_viewportWidth, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f / m_viewportHeight, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    // Render background
    float bgAlpha = (m_config.backgroundColor.a / 255.0f) * toast.currentAlpha;
    RenderBackground(x, y, width, height, static_cast<float>(m_config.cornerRadius),
                    m_config.backgroundColor.r / 255.0f,
                    m_config.backgroundColor.g / 255.0f,
                    m_config.backgroundColor.b / 255.0f,
                    bgAlpha);
    
    // Render text
    float textX = x + m_config.padding;
    float textY = y + height * 0.5f; // Center vertically
    float textAlpha = (m_config.textColor.a / 255.0f) * toast.currentAlpha;
    
    RenderText(toast.text, textX, textY, 1.0f,
              m_config.textColor.r / 255.0f,
              m_config.textColor.g / 255.0f,
              m_config.textColor.b / 255.0f,
              textAlpha);
    
    glDisable(GL_BLEND);
}

void OpenGLToastRenderer::Cleanup() {
    // Clean up character textures
    for (auto& pair : m_characters) {
        if (pair.second.textureID != 0) {
            glDeleteTextures(1, &pair.second.textureID);
        }
    }
    m_characters.clear();
    
    // Clean up OpenGL resources
    if (m_VBO != 0) {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
    
    if (m_VAO != 0) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
    
    if (m_textShader != 0) {
        glDeleteProgram(m_textShader);
        m_textShader = 0;
    }
    
    if (m_backgroundShader != 0) {
        glDeleteProgram(m_backgroundShader);
        m_backgroundShader = 0;
    }
    
    m_initialized = false;
}

bool OpenGLToastRenderer::CreateShaders() {
    // Create text shader
    m_textShader = CreateShaderProgram(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER);
    if (m_textShader == 0) {
        LOG_ERROR("Failed to create text shader program");
        return false;
    }
    
    // Get uniform locations for text shader
    m_textProjectionLoc = glGetUniformLocation(m_textShader, "projection");
    m_textColorLoc = glGetUniformLocation(m_textShader, "textColor");
    m_textTextureLoc = glGetUniformLocation(m_textShader, "text");
    
    // Create background shader
    m_backgroundShader = CreateShaderProgram(BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER);
    if (m_backgroundShader == 0) {
        LOG_ERROR("Failed to create background shader program");
        return false;
    }
    
    // Get uniform locations for background shader
    m_bgProjectionLoc = glGetUniformLocation(m_backgroundShader, "projection");
    m_bgPositionLoc = glGetUniformLocation(m_backgroundShader, "bgPosition");
    m_bgSizeLoc = glGetUniformLocation(m_backgroundShader, "bgSize");
    m_bgColorLoc = glGetUniformLocation(m_backgroundShader, "bgColor");
    m_bgCornerRadiusLoc = glGetUniformLocation(m_backgroundShader, "cornerRadius");
    
    return true;
}

bool OpenGLToastRenderer::CreateOpenGLResources() {
    // Generate VAO and VBO
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    return true;
}

bool OpenGLToastRenderer::InitializeFont() {
    // Initialize basic ASCII characters (simplified implementation)
    // In a real implementation, you'd use FreeType or stb_truetype
    
    for (int i = 32; i < 127; i++) {  // Printable ASCII characters
        char c = static_cast<char>(i);
        GLuint texture = CreateCharacterTexture(c);
        
        if (texture != 0) {
            Character character;
            character.textureID = texture;
            character.sizeX = 8;  // 8x8 bitmap
            character.sizeY = 8;
            character.bearingX = 0;
            character.bearingY = 8;
            character.advance = 9;  // 8 pixels + 1 pixel spacing
            
            m_characters[c] = character;
        }
    }
    
    return !m_characters.empty();
}

GLuint OpenGLToastRenderer::CreateCharacterTexture(char c) {
    // Create a simple 8x8 texture for the character
    // This is a very simplified implementation
    unsigned char pixels[64]; // 8x8 pixels
    
    // Initialize to all zeros (transparent)
    std::fill(pixels, pixels + 64, 0);
    
    // For demonstration, create some basic character patterns
    if (c >= 'A' && c <= 'Z') {
        // Simple pattern for uppercase letters
        for (int i = 0; i < 64; i++) {
            int x = i % 8;
            int y = i / 8;
            // Create a simple rectangular pattern
            if (x >= 1 && x <= 6 && y >= 1 && y <= 6) {
                pixels[i] = 255; // White
            }
        }
    } else if (c >= 'a' && c <= 'z') {
        // Simple pattern for lowercase letters  
        for (int i = 0; i < 64; i++) {
            int x = i % 8;
            int y = i / 8;
            // Create a smaller rectangular pattern
            if (x >= 2 && x <= 5 && y >= 3 && y <= 6) {
                pixels[i] = 255; // White
            }
        }
    } else if (c >= '0' && c <= '9') {
        // Simple pattern for digits
        for (int i = 0; i < 64; i++) {
            int x = i % 8;
            int y = i / 8;
            // Create a digit-like pattern
            if ((x >= 2 && x <= 5 && (y == 2 || y == 6)) ||
                ((x == 2 || x == 5) && y >= 2 && y <= 6)) {
                pixels[i] = 255; // White
            }
        }
    } else {
        // For other characters, create a simple dot pattern
        pixels[28] = 255; // Center pixel
    }
    
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 8, 8, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glBindTexture(GL_TEXTURE_2D, 0);
    
    return texture;
}

void OpenGLToastRenderer::RenderText(const std::string& text, float x, float y, float scale, float r, float g, float b, float a) {
    glUseProgram(m_textShader);
    
    // Set uniforms
    float projection[16] = {
        2.0f / m_viewportWidth, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f / m_viewportHeight, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    glUniformMatrix4fv(m_textProjectionLoc, 1, GL_FALSE, projection);
    glUniform3f(m_textColorLoc, r, g, b);
    glUniform1f(glGetUniformLocation(m_textShader, "alpha"), a);
    
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_VAO);
    
    float currentX = x;
    
    // Render each character
    for (char c : text) {
        auto it = m_characters.find(c);
        if (it == m_characters.end()) {
            currentX += 9 * scale; // Default advance for missing characters
            continue;
        }
        
        const Character& ch = it->second;
        
        float xpos = currentX;
        float ypos = y - (ch.sizeY - ch.bearingY) * scale;
        
        float w = ch.sizeX * scale;
        float h = ch.sizeY * scale;
        
        // Update VBO for each character
        GLfloat vertices[6][4] = {
            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos,     ypos,       0.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 1.0f },
            
            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos + w, ypos,       1.0f, 1.0f },
            { xpos + w, ypos + h,   1.0f, 0.0f }
        };
        
        // Bind character texture
        glBindTexture(GL_TEXTURE_2D, ch.textureID);
        
        // Update VBO content
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        // Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        // Advance cursor for next glyph
        currentX += ch.advance * scale;
    }
    
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLToastRenderer::RenderBackground(float x, float y, float width, float height, float cornerRadius, float r, float g, float b, float a) {
    glUseProgram(m_backgroundShader);
    
    // Set uniforms
    float projection[16] = {
        2.0f / m_viewportWidth, 0.0f, 0.0f, -1.0f,
        0.0f, 2.0f / m_viewportHeight, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    glUniformMatrix4fv(m_bgProjectionLoc, 1, GL_FALSE, projection);
    glUniform2f(m_bgPositionLoc, x, y);
    glUniform2f(m_bgSizeLoc, width, height);
    glUniform4f(m_bgColorLoc, r, g, b, a);
    glUniform1f(m_bgCornerRadiusLoc, cornerRadius);
    
    // Quad vertices (0-1 range)
    GLfloat vertices[] = {
        0.0f, 1.0f,  // Top-left
        0.0f, 0.0f,  // Bottom-left
        1.0f, 0.0f,  // Bottom-right
        
        0.0f, 1.0f,  // Top-left
        1.0f, 0.0f,  // Bottom-right
        1.0f, 1.0f   // Top-right
    };
    
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLToastRenderer::CalculateTextDimensions(const std::string& text, float scale, float* width, float* height) {
    *width = 0.0f;
    *height = 0.0f;
    
    for (char c : text) {
        auto it = m_characters.find(c);
        if (it != m_characters.end()) {
            *width += it->second.advance * scale;
            *height = std::max(*height, static_cast<float>(it->second.sizeY * scale));
        } else {
            *width += 9 * scale; // Default advance
            *height = std::max(*height, 8.0f * scale); // Default height
        }
    }
}

void OpenGLToastRenderer::CalculateToastPosition(const std::string& text, float* x, float* y, float* width, float* height) {
    // Calculate text dimensions
    float textWidth, textHeight;
    CalculateTextDimensions(text, 1.0f, &textWidth, &textHeight);
    
    // Calculate toast dimensions
    *width = std::min(textWidth + 2 * m_config.padding, static_cast<float>(m_config.maxWidth));
    *height = textHeight + 2 * m_config.padding;
    
    // Calculate position based on configuration
    switch (m_config.position) {
        case ToastPosition::TOP_LEFT:
            *x = static_cast<float>(m_config.offsetX);
            *y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::TOP_CENTER:
            *x = (m_viewportWidth - *width) / 2 + m_config.offsetX;
            *y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::TOP_RIGHT:
            *x = m_viewportWidth - *width - m_config.offsetX;
            *y = static_cast<float>(m_config.offsetY);
            break;
        case ToastPosition::CENTER_LEFT:
            *x = static_cast<float>(m_config.offsetX);
            *y = (m_viewportHeight - *height) / 2 + m_config.offsetY;
            break;
        case ToastPosition::CENTER:
            *x = (m_viewportWidth - *width) / 2 + m_config.offsetX;
            *y = (m_viewportHeight - *height) / 2 + m_config.offsetY;
            break;
        case ToastPosition::CENTER_RIGHT:
            *x = m_viewportWidth - *width - m_config.offsetX;
            *y = (m_viewportHeight - *height) / 2 + m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_LEFT:
            *x = static_cast<float>(m_config.offsetX);
            *y = m_viewportHeight - *height - m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_CENTER:
            *x = (m_viewportWidth - *width) / 2 + m_config.offsetX;
            *y = m_viewportHeight - *height - m_config.offsetY;
            break;
        case ToastPosition::BOTTOM_RIGHT:
            *x = m_viewportWidth - *width - m_config.offsetX;
            *y = m_viewportHeight - *height - m_config.offsetY;
            break;
    }
}

void OpenGLToastRenderer::UpdateViewportDimensions() {
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    m_viewportWidth = viewport[2];
    m_viewportHeight = viewport[3];
}

GLuint OpenGLToastRenderer::CompileShader(const char* source, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        LOG_ERROR("Shader compilation failed: ", infoLog);
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint OpenGLToastRenderer::CreateShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = CompileShader(vertexSource, GL_VERTEX_SHADER);
    if (vertexShader == 0) return 0;
    
    GLuint fragmentShader = CompileShader(fragmentSource, GL_FRAGMENT_SHADER);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetProgramInfoLog(program, 1024, NULL, infoLog);
        LOG_ERROR("Shader program linking failed: ", infoLog);
        glDeleteProgram(program);
        program = 0;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}