#pragma once

#include <string>
#include <unordered_map>

/**
 * Centralized shader source library for render passes
 * Provides common shaders and standardized shader sources
 */
class ShaderLibrary {
public:
    // Get common vertex shader (fullscreen quad)
    static std::string GetFullscreenQuadVertexShader();
    
    // Get built-in pixel shaders
    static std::string GetPassthroughPixelShader();
    static std::string GetMotionBlurPixelShader();
    static std::string GetYUVToRGBPixelShader();
    static std::string GetVignettePixelShader();
    static std::string GetSharpenPixelShader();
    static std::string GetSimpleBloomPixelShader();
    static std::string GetBloomExtractPixelShader();
    static std::string GetBloomBlurPixelShader();
    static std::string GetBloomCompositePixelShader();
    
    // Generic shader access by name
    static std::string GetPixelShaderByName(const std::string& name);
    
private:
    ShaderLibrary() = delete;
    
    // Common shader components
    static std::string GetCommonShaderStructures();
    static std::string GetCommonShaderFunctions();
};