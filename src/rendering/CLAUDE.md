# Rendering System

This directory implements a dual-backend rendering system supporting both DirectX 11 and OpenGL, with hardware acceleration and format conversion capabilities.

## Architecture Overview

The rendering system uses a clean abstraction pattern to support multiple graphics APIs:

```
src/rendering/
├── IRenderer.h                     # Abstract renderer interface
├── RendererFactory.h/cpp           # Factory for renderer creation
├── RenderTexture.h                 # Generic texture abstraction
├── TextureConverter.h/cpp          # Frame-to-texture conversion
├── D3D11Renderer.h/cpp             # DirectX 11 implementation  
├── OpenGLRenderer.h/cpp            # OpenGL implementation
├── CudaOpenGLInterop.h/cpp         # CUDA-OpenGL interoperability
├── CudaYuvConversion.cu/h          # CUDA YUV processing kernels
├── renderpass/                     # Comprehensive render pass system
│   ├── IRenderPass.h               # Render pass interface
│   ├── RenderPassPipeline.h/cpp    # Pipeline management and execution
│   ├── RenderPassConfig.h/cpp      # Configuration system integration
│   ├── OverlayRenderPass.h/cpp     # ImGui overlay rendering
│   ├── d3d11/                      # DirectX 11 render passes
│   │   ├── D3D11SimpleRenderPass.h/cpp      # Simple shader-based passes
│   │   ├── D3D11RenderPassResources.h/cpp   # Resource management
│   │   ├── ShaderLibrary.h/cpp              # Built-in shader library
│   │   └── passes/                          # Effect implementations
│   │       ├── YUVToRGBRenderPass.h         # YUV color conversion
│   │       ├── PassthroughPass.h            # Direct texture copy
│   │       ├── MotionBlurPass.h             # Motion blur effect
│   │       ├── BloomPass.h/cpp              # Bloom lighting effect
│   │       ├── SharpenPass.h/cpp            # Image sharpening
│   │       ├── VignettePass.h/cpp           # Vignette darkening
│   │       └── D3D11OverlayRenderPass.h/cpp # ImGui overlay integration
│   ├── opengl/                     # OpenGL render passes
│   │   ├── OpenGLSimpleRenderPass.h/cpp     # OpenGL shader passes
│   │   ├── OpenGLRenderPassResources.h/cpp  # OpenGL resource management
│   │   ├── OpenGLRenderPassPipeline.h/cpp   # OpenGL pipeline management
│   │   └── passes/                          # OpenGL effect implementations
│   │       ├── YUVToRGBRenderPass.h/cpp     # OpenGL YUV conversion
│   │       ├── PassthroughPass.h/cpp        # OpenGL passthrough
│   │       ├── MotionBlurPass.h/cpp         # OpenGL motion blur
│   │       ├── BloomPass.h/cpp              # OpenGL bloom effect
│   │       ├── SharpenPass.h/cpp            # OpenGL sharpening
│   │       ├── VignettePass.h/cpp           # OpenGL vignette
│   │       └── OpenGLOverlayRenderPass.h/cpp # OpenGL ImGui overlays
│   └── CLAUDE.md                   # Render pass system documentation
└── CLAUDE.md                       # This documentation
```

## Core Abstraction Layer

### IRenderer Interface
**File:** `IRenderer.h`
**Purpose:** Unified interface for all rendering backends

**Key Interface Methods:**
```cpp
class IRenderer {
public:
    virtual bool Initialize(HWND hwnd, int width, int height) = 0;
    virtual void Cleanup() = 0;
    virtual bool Present(const RenderTexture& texture) = 0;
    virtual bool Resize(int width, int height) = 0;
    virtual bool IsInitialized() const = 0;
    virtual RendererType GetRendererType() const = 0;
    virtual bool SupportsCudaInterop() const = 0;
};
```

**Renderer Types:**
```cpp
enum class RendererType {
    OpenGL,      // OpenGL 4.6 Core Profile with optional CUDA interop
    DirectX11    // DirectX 11 with D3D11VA hardware acceleration
};
```

### RendererFactory
**File:** `RendererFactory.h/cpp`
**Purpose:** Runtime renderer selection and creation

**Factory Logic:**
```cpp
std::unique_ptr<IRenderer> RendererFactory::CreateRenderer(RendererBackend preference) {
    // Try backends in order based on preference
    std::vector<RendererBackend> backendOrder;
    
    switch (preference) {
        case RendererBackend::DirectX11:
            backendOrder = {RendererBackend::DirectX11, RendererBackend::OpenGL};
            break;
        case RendererBackend::OpenGL:
            backendOrder = {RendererBackend::OpenGL, RendererBackend::DirectX11};
            break;
        case RendererBackend::Auto:
        default:
            backendOrder = {RendererBackend::DirectX11, RendererBackend::OpenGL};
            break;
    }
    
    // Try each backend and return the first successful one
}
```

**Runtime Configuration:**
- Both renderers always compiled and available
- Automatic fallback between backends based on system capabilities
- Runtime selection via RendererBackend enum

## Texture Abstraction System

### RenderTexture
**File:** `RenderTexture.h`
**Purpose:** Generic texture representation supporting multiple graphics APIs

**Supported Texture Types:**
```cpp
enum class TextureType {
    Software,   // CPU memory (uint8_t* data)
    D3D11,      // DirectX 11 texture (ID3D11Texture2D)
    OpenGL,     // OpenGL texture ID
    CUDA        // CUDA device memory for OpenGL interop
};

enum class TextureFormat {
    RGBA8,      // 8-bit RGBA
    BGRA8,      // 8-bit BGRA
    NV12,       // YUV NV12 format (hardware decoded)
    YUV420P     // YUV 420 planar
};
```

**Multi-Backend Texture Data:**
```cpp
struct RenderTexture {
    TextureType type;
    TextureFormat format;
    int width, height;
    bool isYUV; // Requires YUV→RGB shader conversion
    
    // Backend-specific data unions
    struct { const uint8_t* data; int pitch; } software;
    struct { ComPtr<ID3D11Texture2D> texture; DXGI_FORMAT dxgiFormat; } d3d11;
    struct { void* devicePtr; size_t pitch; void* glResource; } cuda;
    struct { unsigned int textureId; } opengl;
};
```

### TextureConverter
**File:** `TextureConverter.h/cpp`
**Purpose:** Converts decoded video frames to renderer-optimized textures

**Conversion Logic:**
```cpp
RenderTexture TextureConverter::ConvertFrame(const DecodedFrame& frame, IRenderer* renderer) {
    switch (renderer->GetRendererType()) {
        case RendererType::DirectX11:
            // Prefer D3D11 hardware texture, fallback to software
            if (frame.texture) return CreateD3D11Texture(frame);
            if (frame.data)    return CreateSoftwareTexture(frame);
            
        case RendererType::OpenGL:
            // Prefer CUDA interop, fallback to software
            if (renderer->SupportsCudaInterop() && frame.isHardwareCuda)
                return CreateCudaTexture(frame);
            if (frame.data) return CreateSoftwareTexture(frame);
    }
}
```

## Renderer Implementations

### DirectX 11 Renderer
**File:** `D3D11Renderer.h/cpp`
**Purpose:** DirectX 11 rendering with hardware video acceleration

**Key Features:**
- **D3D11VA Integration:** Direct hardware decoded texture rendering
- **YUV Shader Processing:** Hardware YUV→RGB conversion via pixel shaders
- **Format Support:** NV12, P010, BGRA8, RGBA8
- **Hardware Acceleration:** GPU-accelerated color space conversion

**Rendering Pipeline:**
```cpp
bool D3D11Renderer::Present(const RenderTexture& texture) {
    switch (texture.type) {
        case TextureType::D3D11:
            // Direct hardware texture rendering (optimal path)
            if (texture.isYUV) {
                return RenderYUVTexture(texture.d3d11.texture, texture.d3d11.dxgiFormat);
            } else {
                return RenderRGBTexture(texture.d3d11.texture);
            }
            
        case TextureType::Software:
            // CPU→GPU upload and rendering
            return RenderSoftwareTexture(texture.software.data, texture.width, texture.height);
    }
}
```

**YUV Processing:**
- **NV12 Format:** Two-plane YUV format (Y plane + interleaved UV)
- **Hardware Conversion:** GPU shader-based YUV→RGB conversion
- **Color Space:** Rec. 709 color space conversion matrices

### OpenGL Renderer  
**File:** `OpenGLRenderer.h/cpp`
**Purpose:** OpenGL 4.6 Core Profile rendering with CUDA interoperability

**Key Features:**
- **OpenGL 4.6 Core:** Modern OpenGL with programmable pipeline
- **CUDA Interop:** Direct CUDA memory→OpenGL texture mapping
- **YUV Shader Processing:** OpenGL shader-based color conversion
- **Software Fallback:** CPU memory upload for software decoded frames

**CUDA Interoperability:**
```cpp
bool OpenGLRenderer::Present(const RenderTexture& texture) {
    switch (texture.type) {
        case TextureType::CUDA:
            // Zero-copy CUDA→OpenGL rendering (optimal for hardware decode)
            return RenderCudaTexture(texture.cuda.devicePtr, texture.cuda.pitch, 
                                   texture.width, texture.height, texture.isYUV);
            
        case TextureType::Software:
            // CPU→GPU upload
            return RenderSoftwareTexture(texture.software.data, texture.width, texture.height);
    }
}
```

## Hardware Acceleration Integration

### CUDA-OpenGL Interoperability  
**File:** `CudaOpenGLInterop.h/cpp`
**Purpose:** Efficient CUDA decoded frame→OpenGL texture mapping

**Interop Process:**
1. **Resource Registration:** Register OpenGL texture for CUDA access
2. **Memory Mapping:** Map CUDA device memory to OpenGL texture
3. **Data Transfer:** Copy CUDA decoded frame to OpenGL texture memory
4. **Unmapping:** Release CUDA mapping, texture ready for rendering

**Benefits:**
- **Zero-Copy Operations:** Direct GPU memory sharing
- **Hardware Pipeline:** End-to-end GPU processing (decode→render)
- **Performance:** Eliminates CPU↔GPU memory transfers

### CUDA YUV Processing
**File:** `CudaYuvConversion.cu/h`  
**Purpose:** GPU-accelerated YUV→RGB color space conversion

**CUDA Kernel Features:**
- **Parallel Processing:** GPU thread-based YUV conversion
- **Color Space Conversion:** Rec. 709 YUV→RGB matrices
- **Format Support:** NV12, YUV420P input formats
- **Memory Efficiency:** Optimized memory access patterns

### D3D11VA Hardware Acceleration
**Integration:** DirectX 11 renderer with hardware decoded textures

**Hardware Pipeline:**
1. **Hardware Decode:** NVDEC→D3D11 texture (VideoDecoder)
2. **Format Detection:** Automatic YUV/RGB format identification
3. **Shader Processing:** Hardware YUV→RGB conversion if needed
4. **Direct Rendering:** Zero-copy rendering from hardware texture

## Format Handling and Conversion

### YUV Format Support
**Supported YUV Formats:**
- **NV12:** 4:2:0 semi-planar (Y plane + interleaved UV)
- **YUV420P:** 4:2:0 planar (separate Y, U, V planes)
- **P010:** 10-bit 4:2:0 semi-planar (high bit depth)

**Conversion Methods:**
- **Hardware Shaders:** GPU-based conversion (both D3D11 and OpenGL)
- **CUDA Kernels:** Parallel GPU conversion for CUDA interop
- **Software Conversion:** libswscale fallback (CPU-based)

### RGB Format Support
**Supported RGB Formats:**
- **RGBA8:** 8-bit RGBA (32-bit per pixel)
- **BGRA8:** 8-bit BGRA (32-bit per pixel, D3D11 preferred)
- **RGB8:** 8-bit RGB (24-bit per pixel, rare)

## Performance Characteristics

### Hardware Rendering Paths
**Optimal Performance (Hardware→Hardware):**
- **D3D11:** NVDEC decode → D3D11 texture → D3D11 render
- **OpenGL+CUDA:** NVDEC decode → CUDA memory → OpenGL interop

**CPU Fallback Path:**
- **Software:** CPU decode → RGB conversion → GPU upload → render

### Memory Usage Patterns
- **Hardware Path:** GPU-only memory (no CPU↔GPU transfers)
- **Software Path:** CPU memory + GPU upload (temporary double buffering)
- **Format Conversion:** In-place conversion when possible

### Frame Rate Optimization
- **Zero-Copy Operations:** Eliminates unnecessary memory transfers
- **Format-Aware Rendering:** Direct YUV rendering without RGB conversion
- **Resource Reuse:** Texture resource pooling for repeated frames

## Error Handling and Fallback

### Renderer Initialization Fallback
```cpp
// Application initialization with graceful fallback
auto renderer = RendererFactory::CreateRenderer();
if (!renderer->Initialize(window.GetHandle(), width, height)) {
    LOG_ERROR("Failed to initialize ", RendererFactory::GetRendererName(), " renderer");
    return 1; // Critical error - no fallback renderer
}
```

### Runtime Format Fallback
```cpp
// TextureConverter handles format mismatches gracefully
RenderTexture renderTexture = TextureConverter::ConvertFrame(frame, renderer.get());
if (!renderTexture.IsValid()) {
    renderTexture = TextureConverter::CreateNullTexture(); // Black frame fallback
}
renderer->Present(renderTexture);
```

### Resource Recovery
- **Device Loss Detection:** D3D11 device lost handling
- **Context Recovery:** OpenGL context loss recovery  
- **Memory Pressure:** Automatic resource cleanup under memory pressure

## Integration with Video System

### Frame Flow Integration
```cpp
// Enhanced rendering pipeline with render passes and overlays
DecodedFrame* currentFrame = videoManager.GetCurrentFrame();

// Convert video frame to render texture
RenderTexture renderTexture;
if (currentFrame && currentFrame->valid) {
    renderTexture = TextureConverter::ConvertFrame(*currentFrame, renderer.get());
} else {
    renderTexture = TextureConverter::CreateNullTexture(); // Black frame
}

// Present through render pass pipeline (includes post-processing and overlays)
renderer->Present(renderTexture);  // Now includes full pipeline processing
```

### Enhanced Rendering Pipeline
```
Video Frame → TextureConverter → RenderTexture
     ↓
IRenderer::Present()
     ↓
Render Pass Pipeline:
├── YUV to RGB Conversion (if needed)
├── Post-Processing Effects Chain
│   ├── Bloom Pass
│   ├── Sharpen Pass 
│   ├── Motion Blur Pass
│   └── Vignette Pass
├── ImGui Overlay Rendering
│   ├── Debug UI Components
│   ├── Notification System
│   └── Performance Overlays
└── Present to Swapchain
```

### Window Management Integration
```cpp
// Enhanced window resizing with render pass and overlay support
if (currentWidth != lastWindowWidth || currentHeight != lastWindowHeight) {
    if (!renderer->Resize(currentWidth, currentHeight)) {
        LOG_ERROR("Failed to resize renderer");
        break; // Critical rendering error
    }
    
    // Update ImGui for overlay rendering
    ImGuiManager::GetInstance().HandleResize(currentWidth, currentHeight);
    
    lastWindowWidth = currentWidth;
    lastWindowHeight = currentHeight;
}
```

## Advanced Render Pass System

### Comprehensive Post-Processing Pipeline
**Major Architecture Enhancement:** The rendering system now includes a sophisticated render pass architecture supporting multiple post-processing effects with both DirectX 11 and OpenGL implementations.

**Available Effects:**
- **YUV to RGB Conversion:** Hardware-accelerated color space conversion
- **Passthrough:** Direct texture copy for testing and fallback
- **Motion Blur:** Directional blur effects with configurable intensity  
- **Bloom:** HDR lighting bloom effects with threshold control
- **Sharpen:** Image sharpening with adjustable strength
- **Vignette:** Subtle darkening effects around edges
- **ImGui Overlay:** Seamless UI overlay rendering integration

### Dual-Backend Render Pass Support
**Complete Parity:** All render passes implemented for both graphics APIs
- **DirectX 11 Passes:** Full HLSL shader implementations with constant buffer management
- **OpenGL Passes:** Complete GLSL shader implementations with uniform management
- **Automatic Selection:** Render passes automatically use appropriate backend
- **Resource Management:** Backend-specific resource pools and lifecycle management

### Configuration-Driven Pipeline
**INI Integration:** All render passes configurable via existing INI system
```ini
[rendering]
enable_render_passes = true
render_pass_chain = bloom, sharpen, vignette

[render_pass.bloom]
enabled = true
threshold = 0.8
intensity = 1.2

[render_pass.sharpen] 
enabled = true
strength = 0.5
```

### ImGui Overlay Integration
**Seamless UI Rendering:** Advanced overlay system with full UI capabilities
- **Debug UI:** Runtime configuration and performance monitoring
- **Notification System:** Toast notifications with configurable styling
- **Component Registry:** Dynamic UI component registration and management
- **Input Integration:** Full mouse and keyboard support within overlays

### Performance Optimization
**Zero-Overhead Design:** Render passes add no performance cost when disabled
- **Lazy Allocation:** Resources created only when passes are enabled
- **Ping-Pong Texturing:** Efficient intermediate texture management
- **Resource Pooling:** Automatic texture reuse and memory management
- **Bypass Support:** Direct passthrough when pipeline is disabled

For detailed information about the render pass system, see **[renderpass/CLAUDE.md](renderpass/CLAUDE.md)**

This enhanced rendering system provides a robust, high-performance foundation for video display with automatic hardware acceleration, graceful software fallback, comprehensive post-processing effects, and modern UI overlay capabilities across multiple graphics APIs.