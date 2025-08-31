#include "ShaderLibrary.h"

std::string ShaderLibrary::GetFullscreenQuadVertexShader() {
    return R"(
struct VS_INPUT {
    float3 pos : POSITION;
    float2 tex : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.pos = float4(input.pos, 1.0f);
    output.tex = input.tex;
    return output;
}
)";
}

std::string ShaderLibrary::GetCommonShaderStructures() {
    return R"(
struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};
)";
}

std::string ShaderLibrary::GetPassthroughPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

float4 main(PS_INPUT input) : SV_TARGET {
    return inputTexture.Sample(inputSampler, input.tex);
}
)";
}

std::string ShaderLibrary::GetMotionBlurPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer MotionBlurParams : register(b0) {
    float blurStrength;
    int sampleCount;
    float2 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 result = float4(0, 0, 0, 0);
    
    // Use the configured blur strength and sample count
    float2 blurDirection = float2(blurStrength * 0.01, 0); // Horizontal blur
    
    // Sample along the blur direction with proper offset distribution
    for (int i = 0; i < sampleCount; i++) {
        float offset = (float(i) / float(sampleCount - 1) - 0.5) * 2.0; // -1.0 to 1.0
        float2 sampleUV = saturate(input.tex + blurDirection * offset);
        result += inputTexture.Sample(inputSampler, sampleUV);
    }
    
    return result / float(sampleCount);
}
)";
}

std::string ShaderLibrary::GetYUVToRGBPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D yTexture : register(t0);   // Y plane
Texture2D uvTexture : register(t1);  // UV plane (for NV12, fallback to single Y for other formats)
SamplerState inputSampler : register(s0);

float4 main(PS_INPUT input) : SV_TARGET {
    // Sample Y (luminance)
    float y = yTexture.Sample(inputSampler, input.tex).r;
    
    // Sample UV (chrominance) - for NV12, UV is at half resolution
    // If UV texture is not available, default to grayscale
    float2 chroma = uvTexture.Sample(inputSampler, input.tex).rg;
    float u = chroma.r - 0.5;
    float v = chroma.g - 0.5;
    
    // Handle case where UV might be zero (fallback to grayscale)
    if (abs(u) < 0.001 && abs(v) < 0.001) {
        return float4(y, y, y, 1.0); // Grayscale fallback
    }
    
    // BT.709 YUV to RGB conversion (full range)
    float r = y + 1.402 * v;
    float g = y - 0.344 * u - 0.714 * v;
    float b = y + 1.772 * u;
    
    return float4(saturate(r), saturate(g), saturate(b), 1.0);
}
)";
}

std::string ShaderLibrary::GetVignettePixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer VignetteParams : register(b0) {
    float intensity;
    float radius;
    float feather;
    float centerX;
    float centerY;
    float aspectRatio;
    float2 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 color = inputTexture.Sample(inputSampler, input.tex);
    
    // Calculate distance from center with aspect ratio correction
    float2 center = float2(0.5 + centerX, 0.5 + centerY);
    float2 coords = input.tex - center;
    coords.x *= aspectRatio;
    float distance = length(coords);
    
    // Calculate vignette factor
    float vignette = 1.0 - smoothstep(radius, radius + feather, distance);
    vignette = lerp(1.0 - intensity, 1.0, vignette);
    
    return color * vignette;
}
)";
}

std::string ShaderLibrary::GetSharpenPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer SharpenParams : register(b0) {
    float sharpness;
    float radius;
    float threshold;
    float texelSizeX;
    float texelSizeY;
    float3 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float2 texelSize = float2(texelSizeX, texelSizeY);
    
    // Sample center pixel
    float4 center = inputTexture.Sample(inputSampler, input.tex);
    
    // Sample surrounding pixels for edge detection
    float4 top = inputTexture.Sample(inputSampler, input.tex + float2(0, -texelSize.y) * radius);
    float4 bottom = inputTexture.Sample(inputSampler, input.tex + float2(0, texelSize.y) * radius);
    float4 left = inputTexture.Sample(inputSampler, input.tex + float2(-texelSize.x, 0) * radius);
    float4 right = inputTexture.Sample(inputSampler, input.tex + float2(texelSize.x, 0) * radius);
    
    // Calculate edge strength
    float4 edge = abs(center - (top + bottom + left + right) * 0.25);
    float edgeStrength = dot(edge.rgb, float3(0.299, 0.587, 0.114));
    
    // Apply sharpening if edge strength exceeds threshold
    if (edgeStrength > threshold) {
        float4 sharpen = center + (center - (top + bottom + left + right) * 0.25) * sharpness;
        return float4(sharpen.rgb, center.a);
    }
    
    return center;
}
)";
}

std::string ShaderLibrary::GetBloomExtractPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer BloomParams : register(b0) {
    float threshold;
    float intensity;
    float2 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 color = inputTexture.Sample(inputSampler, input.tex);
    
    // Calculate luminance
    float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    
    // Extract bright areas above threshold
    float bloomFactor = max(0.0, luminance - threshold);
    float3 bloomColor = color.rgb * bloomFactor * intensity;
    
    return float4(bloomColor, 1.0);
}
)";
}

std::string ShaderLibrary::GetBloomBlurPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer BlurParams : register(b0) {
    float2 blurDirection;
    float blurRadius;
    int sampleCount;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 result = float4(0, 0, 0, 0);
    
    // Gaussian blur sampling
    for (int i = 0; i < sampleCount; i++) {
        float offset = (float(i) / float(sampleCount - 1) - 0.5) * 2.0 * blurRadius;
        float2 sampleUV = input.tex + blurDirection * offset;
        result += inputTexture.Sample(inputSampler, sampleUV);
    }
    
    return result / float(sampleCount);
}
)";
}

std::string ShaderLibrary::GetBloomCompositePixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D originalTexture : register(t0);
Texture2D bloomTexture : register(t1);
SamplerState inputSampler : register(s0);

cbuffer CompositeParams : register(b0) {
    float bloomStrength;
    float exposure;
    float2 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 original = originalTexture.Sample(inputSampler, input.tex);
    float4 bloom = bloomTexture.Sample(inputSampler, input.tex);
    
    // Combine original and bloom
    float3 result = original.rgb + bloom.rgb * bloomStrength;
    
    // Apply exposure
    result = 1.0 - exp(-result * exposure);
    
    return float4(result, original.a);
}
)";
}

std::string ShaderLibrary::GetSimpleBloomPixelShader() {
    return GetCommonShaderStructures() + R"(
Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

cbuffer BloomParams : register(b0) {
    float threshold;
    float intensity;
    float radius;
    float blendFactor;
    float texelSizeX;
    float texelSizeY;
    float2 padding;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float2 texCoord = input.tex;
    
    // Sample original pixel
    float4 original = inputTexture.Sample(inputSampler, texCoord);
    
    // Extract bright areas (simple threshold)
    float4 bright = original;
    float brightness = dot(bright.rgb, float3(0.299, 0.587, 0.114)); // Luminance
    if (brightness < threshold) {
        bright = float4(0, 0, 0, 0);
    } else {
        bright.rgb = (bright.rgb - threshold) / (1.0 - threshold);
    }
    
    // Simple box blur for bloom effect
    float4 bloom = float4(0, 0, 0, 0);
    float2 offset = float2(texelSizeX, texelSizeY) * radius;
    int samples = 0;
    
    // 5x5 sampling kernel for bloom
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float2 sampleCoord = texCoord + float2(x, y) * offset;
            float4 sampleColor = inputTexture.Sample(inputSampler, sampleCoord);
            
            // Extract bright areas from sample
            float sampleBrightness = dot(sampleColor.rgb, float3(0.299, 0.587, 0.114));
            if (sampleBrightness >= threshold) {
                sampleColor.rgb = (sampleColor.rgb - threshold) / (1.0 - threshold);
                bloom += sampleColor * intensity;
                samples++;
            }
        }
    }
    
    if (samples > 0) {
        bloom /= float(samples);
    }
    
    // Blend original with bloom effect
    float4 result = lerp(original, original + bloom, blendFactor);
    result.a = original.a; // Preserve alpha
    
    return saturate(result);
}
)";
}

std::string ShaderLibrary::GetPixelShaderByName(const std::string& name) {
    static std::unordered_map<std::string, std::string(*)()> shaderMap = {
        {"Passthrough", GetPassthroughPixelShader},
        {"MotionBlur", GetMotionBlurPixelShader},
        {"YUVToRGB", GetYUVToRGBPixelShader},
        {"Vignette", GetVignettePixelShader},
        {"Sharpen", GetSharpenPixelShader},
        {"Bloom", GetSimpleBloomPixelShader},
        {"BloomExtract", GetBloomExtractPixelShader},
        {"BloomBlur", GetBloomBlurPixelShader},
        {"BloomComposite", GetBloomCompositePixelShader}
    };
    
    auto it = shaderMap.find(name);
    if (it != shaderMap.end()) {
        return it->second();
    }
    
    return ""; // Shader not found
}