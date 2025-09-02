#pragma once

#include "IToastRenderer.h"
#include "ui/ToastManager.h"
#include "OpenGLHeaders.h"
#include <unordered_map>
#include <string>

/**
 * Character data for bitmap font rendering
 */
struct Character {
    GLuint textureID;   // ID handle of the glyph texture
    int sizeX;         // Size of glyph
    int sizeY;
    int bearingX;      // Offset from baseline to left/top of glyph
    int bearingY;
    int advance;       // Offset to advance to next glyph
};

/**
 * OpenGL implementation of toast rendering using texture-based text.
 * Renders toast notifications as overlay using OpenGL shaders and bitmap fonts.
 */
class OpenGLToastRenderer : public IToastRenderer {
public:
    OpenGLToastRenderer();
    ~OpenGLToastRenderer();
    
    // IToastRenderer interface
    bool Initialize(const ToastConfig& config) override;
    void RenderToast(const ToastMessage& toast) override;
    void Cleanup() override;
    bool IsInitialized() const override { return m_initialized; }

private:
    bool m_initialized;
    
    // OpenGL resources
    GLuint m_VAO, m_VBO;
    GLuint m_textShader;
    GLuint m_backgroundShader;
    
    // Shader uniform locations
    GLint m_textProjectionLoc;
    GLint m_textColorLoc;
    GLint m_textTextureLoc;
    
    GLint m_bgProjectionLoc;
    GLint m_bgPositionLoc;
    GLint m_bgSizeLoc;
    GLint m_bgColorLoc;
    GLint m_bgCornerRadiusLoc;
    
    // Font character map (simplified bitmap font)
    std::unordered_map<char, Character> m_characters;
    
    // Configuration
    ToastConfig m_config;
    
    // Viewport dimensions
    int m_viewportWidth;
    int m_viewportHeight;
    
    /**
     * Create and compile shaders for text and background rendering
     */
    bool CreateShaders();
    
    /**
     * Create OpenGL resources (VAO, VBO)
     */
    bool CreateOpenGLResources();
    
    /**
     * Initialize a simple bitmap font (basic ASCII characters)
     */
    bool InitializeFont();
    
    /**
     * Create a simple texture for a character using basic bitmap
     */
    GLuint CreateCharacterTexture(char c);
    
    /**
     * Render text string at specified position
     */
    void RenderText(const std::string& text, float x, float y, float scale, float r, float g, float b, float a);
    
    /**
     * Render rounded rectangle background
     */
    void RenderBackground(float x, float y, float width, float height, float cornerRadius, float r, float g, float b, float a);
    
    /**
     * Calculate text dimensions for positioning
     */
    void CalculateTextDimensions(const std::string& text, float scale, float* width, float* height);
    
    /**
     * Calculate toast position based on configuration
     */
    void CalculateToastPosition(const std::string& text, float* x, float* y, float* width, float* height);
    
    /**
     * Update viewport dimensions
     */
    void UpdateViewportDimensions();
    
    /**
     * Compile shader from source
     */
    GLuint CompileShader(const char* source, GLenum shaderType);
    
    /**
     * Create shader program from vertex and fragment shaders
     */
    GLuint CreateShaderProgram(const char* vertexSource, const char* fragmentSource);
    
    // Shader source constants
    static const char* TEXT_VERTEX_SHADER;
    static const char* TEXT_FRAGMENT_SHADER;
    static const char* BACKGROUND_VERTEX_SHADER;
    static const char* BACKGROUND_FRAGMENT_SHADER;
};