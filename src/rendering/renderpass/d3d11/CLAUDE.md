# DirectX 11 Render Pass Implementation

This directory contains the DirectX 11-specific implementation of the render pass system, providing shader-based post-processing effects with comprehensive resource management.

## Architecture Overview

The DirectX 11 render pass system implements the abstract render pass interface using HLSL shaders and DirectX 11 resources:

```
src/rendering/renderpass/d3d11/
├── D3D11SimpleRenderPass.h/cpp      # Base class for shader-based passes
├── D3D11RenderPassResources.h/cpp   # Resource management and lifecycle
├── ShaderLibrary.h/cpp              # Built-in shader collection
├── passes/                          # Individual effect implementations
│   ├── YUVToRGBRenderPass.h         # YUV color space conversion
│   ├── PassthroughPass.h            # Direct texture copy
│   ├── MotionBlurPass.h             # Motion blur effect
│   ├── BloomPass.h/cpp              # HDR bloom effect
│   ├── SharpenPass.h/cpp            # Image sharpening
│   ├── VignettePass.h/cpp           # Vignette darkening effect
│   └── D3D11OverlayRenderPass.h/cpp # ImGui overlay integration
└── CLAUDE.md                        # This documentation
```

## Core Components

### D3D11SimpleRenderPass
**File:** `D3D11SimpleRenderPass.h/cpp`
**Purpose:** Base implementation for vertex + pixel shader render passes

**Key Features:**
- **Shader Management:** Built-in and external HLSL shader support
- **Constant Buffer:** Automatic parameter binding and constant buffer updates
- **Resource Creation:** Vertex buffers, input layouts, render states
- **Parameter System:** Type-safe parameter storage and GPU binding

**Base Class Architecture:**
```cpp
class D3D11SimpleRenderPass : public RenderPass {
public:
    D3D11SimpleRenderPass(const std::string& shaderName);
    
    bool Initialize(ID3D11Device* device, const RenderPassConfig& config) override;
    bool Execute(const RenderPassContext& context,
                ID3D11ShaderResourceView* inputSRV,
                ID3D11RenderTargetView* outputRTV) override;
    void UpdateParameters(const std::map<std::string, RenderPassParameter>& parameters) override;
    void Cleanup() override;

private:
    bool LoadShaders(ID3D11Device* device, const std::string& shaderName);
    bool CreateResources(ID3D11Device* device);
    void UpdateConstantBuffer(ID3D11DeviceContext* context);
    
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11Buffer> m_constantBuffer;
    ComPtr<ID3D11InputLayout> m_inputLayout;
    ComPtr<ID3D11SamplerState> m_samplerState;
};
```

### D3D11RenderPassResources
**File:** `D3D11RenderPassResources.h/cpp`
**Purpose:** Centralized resource management for render pass system

**Resource Management:**
```cpp
class D3D11RenderPassResources {
public:
    static D3D11RenderPassResources& GetInstance();
    
    // Shader management
    bool LoadShader(ID3D11Device* device, const std::string& name, 
                   ShaderType type, const std::string& source);
    ComPtr<ID3D11VertexShader> GetVertexShader(const std::string& name);
    ComPtr<ID3D11PixelShader> GetPixelShader(const std::string& name);
    
    // Resource creation utilities
    ComPtr<ID3D11Buffer> CreateVertexBuffer(ID3D11Device* device, const void* data, size_t size);
    ComPtr<ID3D11Buffer> CreateConstantBuffer(ID3D11Device* device, size_t size);
    ComPtr<ID3D11SamplerState> CreateLinearSampler(ID3D11Device* device);
    
    // Resource cleanup
    void Cleanup();

private:
    std::unordered_map<std::string, ComPtr<ID3D11VertexShader>> m_vertexShaders;
    std::unordered_map<std::string, ComPtr<ID3D11PixelShader>> m_pixelShaders;
    std::vector<ComPtr<ID3D11Buffer>> m_buffers;
    std::vector<ComPtr<ID3D11SamplerState>> m_samplerStates;
};
```

### ShaderLibrary
**File:** `ShaderLibrary.h/cpp`
**Purpose:** Built-in HLSL shader collection and management

**Built-in Shaders:**
```cpp
class ShaderLibrary {
public:
    static std::string GetVertexShader(const std::string& name);
    static std::string GetPixelShader(const std::string& name);
    static bool IsShaderBuiltIn(const std::string& name);
    
    // Available built-in shaders:
    // - "Passthrough"    - Direct texture copy
    // - "MotionBlur"     - Directional motion blur
    // - "Bloom"          - HDR bloom effect  
    // - "Sharpen"        - Image sharpening
    // - "Vignette"       - Vignette darkening
    // - "YUVToRGB"       - YUV color space conversion
};
```

**Example Built-in Shader (Motion Blur):**
```hlsl
// Motion Blur Pixel Shader
cbuffer MotionBlurConstants : register(b0) {
    float blurStrength;
    int sampleCount;
    float2 padding;
};

Texture2D inputTexture : register(t0);
SamplerState inputSampler : register(s0);

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    float4 result = float4(0, 0, 0, 0);
    float2 blurDirection = float2(blurStrength * 0.01, 0);
    
    for (int i = 0; i < sampleCount; i++) {
        float offset = (float(i) / float(sampleCount - 1) - 0.5) * 2.0;
        float2 sampleUV = input.tex + blurDirection * offset;
        result += inputTexture.Sample(inputSampler, sampleUV);
    }
    
    return result / float(sampleCount);
}
```

## Render Pass Implementations

### YUVToRGBRenderPass
**File:** `passes/YUVToRGBRenderPass.h`
**Purpose:** Hardware-accelerated YUV to RGB color space conversion

**Features:**
- **Format Support:** NV12, YUV420P input formats
- **Color Spaces:** Rec. 709, Rec. 601 conversion matrices  
- **Hardware Acceleration:** GPU shader-based conversion
- **Performance:** Zero-copy GPU processing

### PassthroughPass
**File:** `passes/PassthroughPass.h`
**Purpose:** Direct texture copy for testing and fallback

**Usage:**
- **Pipeline Testing:** Verify render pass pipeline functionality
- **Fallback Option:** When other passes fail or are disabled
- **Performance Baseline:** Measure pipeline overhead

### MotionBlurPass
**File:** `passes/MotionBlurPass.h`
**Purpose:** Directional motion blur effect

**Parameters:**
- `blur_strength` (float): Blur intensity and direction (0.0 - 1.0)
- `sample_count` (int): Number of blur samples for quality (1 - 32)

### BloomPass
**File:** `passes/BloomPass.h/cpp`
**Purpose:** HDR bloom lighting effect

**Parameters:**
- `threshold` (float): Brightness threshold for bloom (0.0 - 2.0)
- `intensity` (float): Bloom effect intensity (0.0 - 3.0)
- `blur_radius` (float): Bloom blur radius (0.0 - 10.0)

**Multi-Pass Implementation:**
```cpp
bool BloomPass::Execute(const RenderPassContext& context,
                       ID3D11ShaderResourceView* inputSRV,
                       ID3D11RenderTargetView* outputRTV) {
    // Pass 1: Extract bright areas
    ExecuteBrightPass(context, inputSRV, m_brightPassRTV.Get());
    
    // Pass 2: Horizontal blur
    ExecuteBlurPass(context, m_brightPassSRV.Get(), m_blurTempRTV.Get(), true);
    
    // Pass 3: Vertical blur
    ExecuteBlurPass(context, m_blurTempSRV.Get(), m_blurResultRTV.Get(), false);
    
    // Pass 4: Composite with original
    ExecuteCompositePass(context, inputSRV, m_blurResultSRV.Get(), outputRTV);
    
    return true;
}
```

### SharpenPass
**File:** `passes/SharpenPass.h/cpp`
**Purpose:** Image sharpening filter

**Parameters:**
- `strength` (float): Sharpening strength (0.0 - 2.0)
- `clamp` (float): Sharpening clamp to prevent artifacts (0.0 - 1.0)

### VignettePass
**File:** `passes/VignettePass.h/cpp`
**Purpose:** Vignette darkening effect around screen edges

**Parameters:**
- `intensity` (float): Vignette darkness intensity (0.0 - 1.0)
- `softness` (float): Edge softness (0.0 - 2.0)
- `radius` (float): Vignette radius (0.5 - 2.0)

### D3D11OverlayRenderPass
**File:** `passes/D3D11OverlayRenderPass.h/cpp`
**Purpose:** ImGui overlay rendering integration

**Features:**
- **ImGui Rendering:** Direct ImGui rendering onto video content
- **Transparency:** Alpha blending with video background
- **Input Integration:** Mouse and keyboard input handling
- **UI Components:** Debug UI, notifications, configuration panels

**Overlay Rendering:**
```cpp
bool D3D11OverlayRenderPass::Execute(const RenderPassContext& context,
                                    ID3D11ShaderResourceView* inputSRV,
                                    ID3D11RenderTargetView* outputRTV) {
    // First, render the input video to output
    RenderVideoBackground(context, inputSRV, outputRTV);
    
    // Then render ImGui overlay on top
    ImGuiManager& imgui = ImGuiManager::GetInstance();
    imgui.NewFrame();
    
    // Draw UI components
    if (OverlayManager::GetInstance().IsUIRegistryVisible()) {
        UIRegistry::GetInstance().DrawDebugUI();
    }
    
    if (OverlayManager::GetInstance().IsNotificationsVisible()) {
        NotificationManager::GetInstance().DrawNotifications();
    }
    
    // Render ImGui to the same render target
    ID3D11DeviceContext* deviceContext = GetDeviceContext();
    deviceContext->OMSetRenderTargets(1, &outputRTV, nullptr);
    imgui.Render();
    
    return true;
}
```

## Resource Management

### Efficient Resource Usage
**Resource Pooling:** Shared resources across render passes
- **Shader Caching:** Loaded shaders shared between pass instances
- **Buffer Reuse:** Common vertex and constant buffers reused  
- **Sampler State Sharing:** Standard sampler states shared
- **Texture Pooling:** Intermediate textures managed by pipeline

### Memory Management
**RAII Patterns:** Comprehensive resource cleanup
```cpp
// Automatic resource cleanup in destructors
D3D11SimpleRenderPass::~D3D11SimpleRenderPass() {
    Cleanup();  // Releases all ComPtr resources automatically
}

// Resource cleanup on device lost/reset
void OnDeviceLost() {
    D3D11RenderPassResources::GetInstance().Cleanup();
    // All ComPtr resources automatically released
}
```

### Performance Optimization
**Minimal State Changes:** Optimized rendering state management
```cpp
bool D3D11SimpleRenderPass::Execute(...) {
    // Set shaders (cached, minimal state changes)
    context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
    context->PSSetShader(m_pixelShader.Get(), nullptr, 0);
    
    // Update parameters only if changed
    if (m_parametersChanged) {
        UpdateConstantBuffer(context);
        m_parametersChanged = false;
    }
    
    // Draw with minimal overhead
    context->Draw(4, 0);  # Fullscreen quad
}
```

## Integration with Pipeline

### Pipeline Integration
**Seamless Integration:** Works with render pass pipeline system
```cpp
// Pipeline automatically manages D3D11 render passes
RenderPassPipeline pipeline;
pipeline.AddPass(std::make_unique<BloomPass>());
pipeline.AddPass(std::make_unique<SharpenPass>());
pipeline.AddPass(std::make_unique<D3D11OverlayRenderPass>());

// Execute entire pipeline
pipeline.Execute(context, inputTexture, outputTarget);
```

### Configuration Integration
**INI Configuration:** All passes configurable via INI files
```ini
[rendering]
enable_render_passes = true
render_pass_chain = bloom, sharpen, overlay

[render_pass.bloom]
enabled = true
shader = Bloom
threshold = 0.8
intensity = 1.2
blur_radius = 3.0

[render_pass.sharpen]
enabled = true
shader = Sharpen
strength = 0.7
clamp = 0.3
```

## Performance Characteristics

### GPU Performance
```
Typical Performance (1920x1080):
├── Passthrough: ~0.1ms
├── MotionBlur: ~0.8ms (16 samples)  
├── Bloom: ~2.5ms (multi-pass)
├── Sharpen: ~0.5ms
├── Vignette: ~0.3ms
├── Overlay: ~1.0ms (depends on UI complexity)
└── Pipeline Overhead: ~0.1ms
```

### Memory Usage
```
Resource Usage:
├── Shader Cache: ~50KB (all built-in shaders)
├── Vertex Buffers: ~1KB per pass
├── Constant Buffers: ~256 bytes per pass
├── Intermediate Textures: Managed by pipeline
└── Total Overhead: ~100KB base + texture memory
```

## Detailed Effect Documentation

For comprehensive information about individual DirectX 11 render pass effects:

### Individual Effects
- **[passes/CLAUDE.md](passes/CLAUDE.md)** - Complete documentation of all DirectX 11 render pass effects including HLSL shader implementations, parameter descriptions, and performance characteristics

This DirectX 11 render pass implementation provides high-performance, GPU-accelerated post-processing effects with comprehensive resource management and seamless integration with the render pass pipeline system.