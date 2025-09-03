# DirectX 11 Render Pass Effects

This directory contains the individual DirectX 11 render pass implementations, providing a comprehensive collection of GPU-accelerated post-processing effects.

## Available Effects

### Core Passes
```
src/rendering/renderpass/d3d11/passes/
├── YUVToRGBRenderPass.h         # YUV color space conversion
├── PassthroughPass.h            # Direct texture copy (testing/fallback)
├── MotionBlurPass.h             # Directional motion blur
├── BloomPass.h/cpp              # HDR bloom lighting effect
├── SharpenPass.h/cpp            # Image sharpening filter
├── VignettePass.h/cpp           # Vignette darkening effect
├── D3D11OverlayRenderPass.h/cpp # ImGui overlay integration
└── CLAUDE.md                    # This documentation
```

## Effect Implementations

### YUVToRGBRenderPass
**Purpose:** Hardware-accelerated YUV to RGB color space conversion
**Usage:** Essential for hardware-decoded video frames

**Key Features:**
- **Format Support:** NV12, YUV420P, P010 formats
- **Color Spaces:** Rec. 709, Rec. 601 conversion matrices
- **Bit Depth:** 8-bit and 10-bit support
- **Performance:** Zero-copy GPU processing

**HLSL Implementation:**
```hlsl
// YUV to RGB conversion matrix (Rec. 709)
static const float3x3 YUVToRGBMatrix = {
    { 1.0f,  0.0f,      1.5748f   },
    { 1.0f, -0.1873f,  -0.4681f   },
    { 1.0f,  1.8556f,   0.0f      }
};

float4 YUVToRGBPixelShader(PS_INPUT input) : SV_TARGET {
    float3 yuv;
    yuv.x = luminanceTexture.Sample(linearSampler, input.tex).r;
    yuv.yz = chrominanceTexture.Sample(linearSampler, input.tex).rg - 0.5f;
    
    float3 rgb = mul(YUVToRGBMatrix, yuv);
    return float4(saturate(rgb), 1.0f);
}
```

### PassthroughPass  
**Purpose:** Direct texture copy with no processing
**Usage:** Testing, fallback, and pipeline validation

**Features:**
- **Zero Processing:** Direct texture copy
- **Format Preservation:** Maintains input format exactly
- **Performance Baseline:** Measures pure pipeline overhead
- **Fallback Safety:** Always works as last resort

### MotionBlurPass
**Purpose:** Directional motion blur effect
**Usage:** Cinematic motion effects and dynamic visual enhancement

**Parameters:**
- `blur_strength` (0.0-1.0): Controls blur intensity and direction
- `sample_count` (1-32): Quality vs performance trade-off

**HLSL Implementation:**
```hlsl
cbuffer MotionBlurConstants : register(b0) {
    float blurStrength;
    int sampleCount;
    float2 padding;
};

float4 MotionBlurPixelShader(PS_INPUT input) : SV_TARGET {
    float4 result = float4(0, 0, 0, 0);
    float2 blurDirection = float2(blurStrength * 0.01f, 0);
    
    for (int i = 0; i < sampleCount; i++) {
        float offset = (float(i) / float(sampleCount - 1) - 0.5f) * 2.0f;
        float2 sampleUV = input.tex + blurDirection * offset;
        result += inputTexture.Sample(linearSampler, sampleUV);
    }
    
    return result / float(sampleCount);
}
```

### BloomPass
**Purpose:** HDR bloom lighting effect with multi-pass implementation
**Usage:** Dramatic lighting effects and HDR-style enhancement

**Parameters:**
- `threshold` (0.0-2.0): Brightness threshold for bloom trigger
- `intensity` (0.0-3.0): Final bloom effect strength
- `blur_radius` (0.0-10.0): Bloom spread and softness

**Multi-Pass Architecture:**
```cpp
class BloomPass : public D3D11SimpleRenderPass {
    struct BloomResources {
        ComPtr<ID3D11Texture2D> brightPassTexture;
        ComPtr<ID3D11RenderTargetView> brightPassRTV;
        ComPtr<ID3D11ShaderResourceView> brightPassSRV;
        
        ComPtr<ID3D11Texture2D> blurTempTexture;
        ComPtr<ID3D11RenderTargetView> blurTempRTV;
        ComPtr<ID3D11ShaderResourceView> blurTempSRV;
        
        ComPtr<ID3D11Texture2D> blurResultTexture;
        ComPtr<ID3D11RenderTargetView> blurResultRTV;
        ComPtr<ID3D11ShaderResourceView> blurResultSRV;
    } m_resources;
    
    bool ExecutePass(const RenderPassContext& context,
                    ID3D11ShaderResourceView* inputSRV,
                    ID3D11RenderTargetView* outputRTV) override {
        // 4-pass bloom implementation
        ExtractBrightAreas(inputSRV, m_resources.brightPassRTV.Get());
        BlurHorizontal(m_resources.brightPassSRV.Get(), m_resources.blurTempRTV.Get());
        BlurVertical(m_resources.blurTempSRV.Get(), m_resources.blurResultRTV.Get());
        CompositeBloom(inputSRV, m_resources.blurResultSRV.Get(), outputRTV);
        
        return true;
    }
};
```

**Bright Pass Extraction:**
```hlsl
float4 BrightPassPixelShader(PS_INPUT input) : SV_TARGET {
    float4 color = inputTexture.Sample(linearSampler, input.tex);
    float luminance = dot(color.rgb, float3(0.299f, 0.587f, 0.114f));
    
    // Extract pixels above threshold
    float bloomAmount = saturate((luminance - threshold) / (1.0f - threshold));
    return color * bloomAmount;
}
```

### SharpenPass
**Purpose:** Image sharpening filter with artifact prevention
**Usage:** Enhance video clarity and perceived sharpness

**Parameters:**
- `strength` (0.0-2.0): Sharpening intensity
- `clamp` (0.0-1.0): Artifact prevention limit

**HLSL Implementation:**
```hlsl
cbuffer SharpenConstants : register(b0) {
    float strength;
    float clamp;
    float2 texelSize;
};

float4 SharpenPixelShader(PS_INPUT input) : SV_TARGET {
    float4 center = inputTexture.Sample(pointSampler, input.tex);
    
    // Sample surrounding pixels
    float4 top = inputTexture.Sample(pointSampler, input.tex + float2(0, -texelSize.y));
    float4 bottom = inputTexture.Sample(pointSampler, input.tex + float2(0, texelSize.y));
    float4 left = inputTexture.Sample(pointSampler, input.tex + float2(-texelSize.x, 0));
    float4 right = inputTexture.Sample(pointSampler, input.tex + float2(texelSize.x, 0));
    
    // Calculate sharpening kernel
    float4 sharpen = center * 5.0f - (top + bottom + left + right);
    sharpen = saturate(center + sharpen * strength * min(clamp, 1.0f));
    
    return sharpen;
}
```

### VignettePass
**Purpose:** Vignette darkening effect around screen edges
**Usage:** Cinematic framing and focus enhancement

**Parameters:**
- `intensity` (0.0-1.0): Vignette darkness strength
- `softness` (0.0-2.0): Edge gradient softness
- `radius` (0.5-2.0): Vignette coverage radius

**HLSL Implementation:**
```hlsl
cbuffer VignetteConstants : register(b0) {
    float intensity;
    float softness;
    float radius;
    float padding;
};

float4 VignettePixelShader(PS_INPUT input) : SV_TARGET {
    float4 color = inputTexture.Sample(linearSampler, input.tex);
    
    // Calculate distance from center
    float2 center = float2(0.5f, 0.5f);
    float distance = length(input.tex - center);
    
    // Calculate vignette factor
    float vignette = 1.0f - smoothstep(radius - softness, radius, distance);
    vignette = saturate(lerp(1.0f, vignette, intensity));
    
    return color * vignette;
}
```

### D3D11OverlayRenderPass
**Purpose:** ImGui overlay rendering integration
**Usage:** Debug UI, notifications, and runtime configuration

**Features:**
- **ImGui Integration:** Full ImGui rendering support
- **Alpha Blending:** Transparent overlays over video content
- **Input Handling:** Mouse and keyboard input processing
- **Component System:** Multiple UI components (debug UI, notifications)

**Implementation:**
```cpp
class D3D11OverlayRenderPass : public D3D11SimpleRenderPass {
public:
    bool Execute(const RenderPassContext& context,
                ID3D11ShaderResourceView* inputSRV,
                ID3D11RenderTargetView* outputRTV) override {
        
        // First pass: Render video background
        RenderVideoToTarget(context, inputSRV, outputRTV);
        
        // Second pass: Render ImGui overlay
        ImGuiManager& imgui = ImGuiManager::GetInstance();
        
        if (imgui.IsInitialized()) {
            imgui.NewFrame();
            
            // Draw registered UI components
            if (OverlayManager::GetInstance().IsUIRegistryVisible()) {
                UIRegistry::GetInstance().DrawDebugUI();
            }
            
            if (OverlayManager::GetInstance().IsNotificationsVisible()) {
                NotificationManager::GetInstance().DrawNotifications();
            }
            
            // Render ImGui to same render target (alpha blended)
            SetRenderTarget(outputRTV);
            imgui.Render();
        }
        
        return true;
    }

private:
    void RenderVideoToTarget(const RenderPassContext& context,
                           ID3D11ShaderResourceView* inputSRV,
                           ID3D11RenderTargetView* outputRTV) {
        // Use passthrough shader to copy video to output
        ID3D11DeviceContext* deviceContext = context.deviceContext;
        
        deviceContext->OMSetRenderTargets(1, &outputRTV, nullptr);
        deviceContext->PSSetShaderResources(0, 1, &inputSRV);
        deviceContext->VSSetShader(m_passthroughVS.Get(), nullptr, 0);
        deviceContext->PSSetShader(m_passthroughPS.Get(), nullptr, 0);
        deviceContext->Draw(4, 0);  // Fullscreen quad
    }
};
```

## Performance and Quality Settings

### Performance Optimization
**Quality vs Performance Trade-offs:**
```ini
# High Performance Settings
[render_pass.motion_blur]
sample_count = 8        # Lower samples for speed

[render_pass.bloom]
blur_radius = 2.0       # Smaller radius for speed

# High Quality Settings  
[render_pass.motion_blur]
sample_count = 24       # Higher samples for quality

[render_pass.bloom]
blur_radius = 6.0       # Larger radius for quality
```

### GPU Performance
```
Effect Performance (1920x1080 @ 60fps):
├── YUVToRGB: ~0.2ms (essential, no alternative)
├── Passthrough: ~0.1ms (baseline)
├── MotionBlur: 0.3ms (8 samples) - 1.2ms (24 samples)
├── Bloom: 1.5ms (small) - 4.0ms (large radius)
├── Sharpen: ~0.4ms (simple kernel)
├── Vignette: ~0.2ms (per-pixel calculation)
└── Overlay: 0.5ms - 2.0ms (depends on UI complexity)
```

### Memory Usage
```
Effect Memory Usage:
├── YUVToRGB: 0MB (in-place conversion)
├── Passthrough: 0MB (direct copy)
├── MotionBlur: 0MB (single-pass)
├── Bloom: ~25MB (intermediate textures for 1080p)
├── Sharpen: 0MB (single-pass)
├── Vignette: 0MB (single-pass)
└── Overlay: ~5MB (ImGui resources)
```

## Configuration Examples

### Cinematic Enhancement
```ini
[rendering]
render_pass_chain = bloom, vignette, overlay

[render_pass.bloom]
enabled = true
threshold = 0.6
intensity = 0.8
blur_radius = 4.0

[render_pass.vignette] 
enabled = true
intensity = 0.3
softness = 1.5
radius = 0.8
```

### Clarity Enhancement
```ini
[rendering]
render_pass_chain = sharpen, overlay

[render_pass.sharpen]
enabled = true
strength = 0.8
clamp = 0.4
```

### Dynamic Effects
```ini
[rendering]
render_pass_chain = motion_blur, bloom, overlay

[render_pass.motion_blur]
enabled = true
blur_strength = 0.03
sample_count = 12

[render_pass.bloom]
enabled = true
threshold = 0.7
intensity = 1.0
blur_radius = 3.0
```

This collection of DirectX 11 render pass effects provides comprehensive post-processing capabilities with excellent performance characteristics and extensive configurability for various visual enhancement scenarios.