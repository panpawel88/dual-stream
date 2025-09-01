#pragma once

#include "OpenGLRenderPass.h"
#include <string>
#include <map>
#include <vector>

/**
 * Simple OpenGL render pass implementation using vertex + fragment shaders
 */
class OpenGLSimpleRenderPass : public OpenGLRenderPass {
public:
    OpenGLSimpleRenderPass(const std::string& name) : OpenGLRenderPass(name), 
        m_program(0), m_vertexShader(0), m_fragmentShader(0), m_uniformBuffer(0),
        m_uniformBufferSize(0), m_uniformBufferDirty(false) {}
    virtual ~OpenGLSimpleRenderPass();

    // OpenGLRenderPass interface
    PassType GetType() const override { return PassType::Simple; }
    bool Initialize(const RenderPassConfig& config) override;
    void Cleanup() override;
    bool Execute(const OpenGLRenderPassContext& context,
                GLuint inputTexture,
                GLuint outputFramebuffer,
                GLuint outputTexture = 0) override;
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;

protected:
    // Virtual methods for derived classes to override
    virtual std::string GetVertexShaderSource() const;
    virtual std::string GetFragmentShaderSource() const;
    virtual size_t GetUniformBufferSize() const;
    virtual void PackUniformBuffer(uint8_t* buffer, const OpenGLRenderPassContext& context);
    
    // Shader loading
    bool LoadVertexShader(const std::string& shaderPath);
    bool LoadFragmentShader(const std::string& shaderPath);
    bool LoadShadersFromSource();
    
    // Uniform buffer management
    bool CreateUniformBuffer(size_t size);
    bool UpdateUniformBuffer(const OpenGLRenderPassContext& context);
    void PackParameters();
    
    // Rendering
    bool CreateFullscreenQuad();
    void RenderFullscreenQuad();
    bool InitializeSharedResources();

protected:
    // OpenGL resources
    GLuint m_program;
    GLuint m_vertexShader;
    GLuint m_fragmentShader;
    GLuint m_uniformBuffer;
    
    // Note: Geometry, samplers, and render states are now shared via OpenGLRenderPassResources
    
    // Parameter management
    std::map<std::string, RenderPassParameter> m_parameters;
    std::vector<uint8_t> m_uniformBufferData;
    size_t m_uniformBufferSize;
    bool m_uniformBufferDirty;
    
    // Shader paths
    std::string m_vertexShaderPath;
    std::string m_fragmentShaderPath;
    std::string m_shaderName; // For built-in shaders

private:
    bool CompileShader(GLenum shaderType, const std::string& source, GLuint& shaderOut);
    bool LinkProgram();
    std::string LoadShaderFile(const std::string& filename);
};

// Vertex structure for fullscreen quad
struct OpenGLRenderPassVertex {
    float position[3];  // x, y, z
    float texCoord[2];  // u, v
};