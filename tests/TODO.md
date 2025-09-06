# DualStream Video Player - Testing System TODO

This document outlines the remaining tasks to complete the comprehensive testing system for the dual-stream video player.

## 📊 Current Status Summary

**✅ WORKING:** Basic test system compiles and runs successfully
- Simple functionality tests: 3/3 passing (100%)
- Build targets: `simple_test`, `run_basic_tests`, `run_all_tests`, `run_performance_tests`
- Location: `build/bin/Release/simple_test.exe`

**✅ WORKING:** JSON configuration system fully restored
- JSON parsing: nlohmann/json v3.11.3 integrated via FetchContent
- Test configuration loading: 4 test suites, 10 individual tests parsed successfully
- Localized dependency: JSON library isolated to tests directory only
- Working executables: `json_test.exe`, `json_config_test.exe`

**✅ WORKING:** Full video testing system restored with modular architecture
- TestRunner.cpp, FrameValidator.cpp, SwitchingValidator.cpp, PerformanceBenchmark.cpp fully operational
- Re-enabled in CMakeLists.txt after modular refactoring
- All video system dependencies resolved through core library linkage
- Working executable: `build/bin/Release/test_runner.exe`

**🎯 COMPLETED:** Full test system restored through modular architecture refactoring

## ✅ COMPLETED - Modular Architecture Refactoring

### New Build Architecture
The codebase has been successfully refactored into a clean modular architecture:

1. **dual_stream_core (Static Library)**
   - Contains all business logic: video processing, rendering, camera system, UI components
   - All source files except main.cpp moved to this library
   - Proper dependency management with FFmpeg, CUDA, OpenCV, ImGui
   - Shared by both main application and test system

2. **dual_stream.exe (Main Application)**
   - Thin executable containing only main.cpp entry point
   - Links with dual_stream_core static library
   - All functionality preserved, no behavioral changes

3. **test_runner.exe (Test Application)**
   - Links with dual_stream_core static library  
   - Full access to video system components (VideoManager, IRenderer, etc.)
   - No longer needs forward declarations or header workarounds
   - All test files (TestRunner, FrameValidator, SwitchingValidator, PerformanceBenchmark) operational

### Benefits Achieved
- **Resolved all compilation issues**: Test system compiles and links successfully
- **Clean separation**: Library code separated from application entry points
- **Better testability**: Tests can access all video system components directly
- **Maintainability**: Single source of truth for all business logic
- **Faster builds**: Core library only rebuilds when business logic changes

### Immediate Next Steps to Restore Full Test System (COMPLETED ✅)
1. **Phase 1: Resolve dependencies** ✅ 
   - ~~Add FFmpeg include paths to test CMakeLists.txt~~
   - ~~Link test executables with main application libraries~~
   - **COMPLETED:** Core library provides all dependencies automatically

2. **Phase 2: Restore original test files** ✅
   - ~~Uncomment TestRunner.cpp, FrameValidator.cpp, SwitchingValidator.cpp, PerformanceBenchmark.cpp in CMakeLists.txt~~
   - ~~Fix remaining compilation issues with proper includes and linking~~
   - **COMPLETED:** All test files operational and linking with core library

3. **Phase 3: API Integration** ✅
   - ~~Fix DecodedFrame API mismatch (linesize vs pitch)~~
   - ~~Update function signatures to match current implementation~~
   - ~~Replace forward declarations with direct header includes~~
   - **COMPLETED:** All API compatibility issues resolved

## ⚠️ HISTORICAL - Previous Compilation & Dependency Issues (RESOLVED ✅)

### Build System & Dependencies
- [x] **Resolve JSON configuration dependency** ✅
  - ~~Current: jsoncpp dependency removed, hardcoded test config used~~
  - ~~Issue: `find_package(jsoncpp CONFIG REQUIRED)` failed - jsoncpp not available in vcpkg setup~~
  - ~~Workaround: Removed JSON parsing, using hardcoded test configurations in `test_runner_main.cpp`~~
  - **COMPLETED:** Integrated nlohmann/json v3.11.3 via FetchContent in tests/CMakeLists.txt
  - **VERIFIED:** Full JSON parsing from test_config.json working (4 test suites, 10 tests)
  - **LOCALIZED:** JSON dependency isolated to tests directory only, main app unaffected

- [x] **Fix complex test system compilation** ✅ **COMPLETED **
  - ~~Current: Full test system (TestRunner.cpp, FrameValidator.cpp, etc.) commented out~~
  - ~~Issue: Dependencies on VideoManager, IRenderer, FFmpeg headers causing compilation failures~~
  - ~~Workaround: Created simple_test.cpp with basic functionality tests only~~
  - **COMPLETED:** Created dual_stream_core static library providing all dependencies
  - **RESULT:** All test files (TestRunner.cpp, FrameValidator.cpp, SwitchingValidator.cpp, PerformanceBenchmark.cpp) fully operational

- [x] **Fix FFmpeg header dependencies in test files** ✅ **COMPLETED **
  - ~~Current: Tests that include `../src/video/VideoManager.h` fail to compile~~
  - ~~Issue: `libavformat/avformat.h: No such file or directory` - FFmpeg headers not in test include path~~
  - ~~Workaround: Disabled affected test files in CMakeLists.txt~~
  - **COMPLETED:** Core library provides all FFmpeg dependencies automatically
  - **RESULT:** All video system headers accessible through core library linkage

- [x] **Resolve video system integration issues** ✅ **COMPLETED **
  - ~~Current: TestRunner attempts to initialize actual VideoManager, IRenderer components~~
  - ~~Issue: Complex dependency chain requires full application initialization~~
  - ~~Workaround: Using simple_test.cpp with no video system dependencies~~
  - **COMPLETED:** Direct access to all video system components through core library
  - **RESULT:** Full test system ready for comprehensive video system testing

### Syntax & Code Issues (Fixed)
- [x] **Fixed FrameValidator.h syntax error** ✅
  - Issue: `bool ValidateBouncing BallPattern` - missing identifier
  - Fixed: Changed to `bool ValidateBouncingBallPattern`
  
- [x] **Fixed missing thread header** ✅
  - Issue: `std::thread` not available in PerformanceBenchmark.h
  - Fixed: Added `#include <thread>` to PerformanceBenchmark.h

- [x] **Fixed CMake target dependencies** ✅
  - Issue: Main CMakeLists.txt referenced non-existent `test_runner` target
  - Fixed: Updated to use `simple_test` target instead

### Current Working State
- ✅ **Simple test system compiles and runs successfully**
  - Executable: `build/bin/Release/simple_test.exe`
  - Tests: Basic functionality, performance metrics, memory operations
  - Status: 3/3 tests passing (100% success rate)
  - Build targets: `simple_test`, `run_basic_tests`, `run_all_tests`, `run_performance_tests`

- ✅ **JSON configuration system fully operational**
  - Library: nlohmann/json v3.11.3 via FetchContent (localized to tests/ directory)
  - Configuration: test_config.json parsing with 4 test suites, 10 individual tests
  - Executables: `build/bin/Release/json_test.exe`, `build/bin/Release/json_config_test.exe`
  - Implementation: `test_runner_main.cpp` with comprehensive JSON parsing and validation
  - Status: All JSON functionality tests passing (100% success rate)

### ✅ COMPLETED - Full Test System Restoration

**All phases completed through modular architecture refactoring:**

1. **Phase 1: Resolve dependencies** ✅ **COMPLETED**
   - ~~Add FFmpeg include paths to test CMakeLists.txt~~
   - ~~Link test executables with main application libraries~~
   - **COMPLETED:** Created dual_stream_core static library providing automatic dependency resolution

2. **Phase 2: Restore original test files** ✅ **COMPLETED**
   - ~~Uncomment TestRunner.cpp, FrameValidator.cpp, SwitchingValidator.cpp, PerformanceBenchmark.cpp in CMakeLists.txt~~
   - ~~Fix remaining compilation issues with proper includes and linking~~
   - **COMPLETED:** All test files operational and linking with core library

3. **Phase 3: JSON configuration system** ✅ **COMPLETED**
   - ~~Install jsoncpp through vcpkg or implement lightweight JSON parser~~
   - **COMPLETED:** Flexible test configuration from test_config.json restored
   - **COMPLETED:** Full JSON parsing functionality implemented with nlohmann/json

## 🔴 High Priority - Core Testing Functionality

### Frame Validation System
- [ ] **Implement full frame extraction from rendered output**
  - Add methods to IRenderer to capture current framebuffer to CPU memory
  - Support both DirectX 11 and OpenGL frame capture
  - Handle different pixel formats (RGBA, BGRA, YUV) correctly
  - Optimize capture performance to minimize test overhead

- [ ] **Add OCR integration for accurate frame number detection**  
  - Integrate Tesseract OCR or similar library for text recognition
  - Implement robust text extraction from frame overlay regions
  - Handle different font sizes and styles across test videos
  - Add fallback pattern matching for OCR failures
  - Validate extracted text against expected frame number patterns

### Video System Integration  
- [ ] **Add video system integration to TestRunner**
  - Link TestRunner with actual VideoManager, IRenderer, and Window components
  - Create test-specific initialization without full UI dependencies
  - Support offscreen rendering for CI environments
  - Implement proper cleanup and resource management between tests

- [ ] **Complete SwitchingValidator with real video system integration**
  - Implement actual video loading and playback control
  - Add real-time switching trigger simulation (keyboard events)
  - Measure actual switching latency with high-precision timing
  - Validate post-switch frame accuracy using frame number extraction
  - Test all three algorithms (immediate, keyframe-sync, predecoded) with real scenarios

## 🟡 Medium Priority - Advanced Testing Features

### Performance Monitoring
- [ ] **Implement actual performance monitoring with system resource tracking**
  - Add Windows Performance Counters integration for CPU/GPU usage
  - Monitor memory usage patterns (process memory, GPU memory)
  - Track frame delivery timing and consistency
  - Implement percentile calculations (P95, P99) for latency metrics
  - Add frame drop detection and analysis

- [ ] **Implement hardware acceleration path testing**
  - Validate NVDEC hardware decoding is being used when available
  - Test D3D11VA and CUDA interop paths separately
  - Verify graceful fallback to software decoding
  - Measure performance differences between hardware/software paths
  - Test different GPU vendors and driver versions

### Visual Regression Testing
- [ ] **Create visual regression testing with golden reference images**
  - Generate golden reference frames for each test video at key timestamps
  - Implement perceptual image comparison (SSIM, PSNR)
  - Handle acceptable variations due to hardware differences
  - Store golden images in version control with proper organization
  - Add tools for updating golden images when video generation changes

## 🟢 Lower Priority - Quality & Robustness

### Error Handling & Edge Cases
- [ ] **Add comprehensive error handling and edge case testing**
  - Test with corrupted video files
  - Test with missing video files
  - Test with unsupported video formats
  - Test resource exhaustion scenarios (out of memory, full disk)
  - Test rapid switching edge cases
  - Validate graceful degradation under stress

- [ ] **Add memory leak detection and profiling**
  - Integrate Application Verifier or similar tools
  - Monitor memory usage over extended test runs
  - Detect resource leaks (textures, file handles, etc.)
  - Add stress tests with thousands of switches
  - Profile memory allocation patterns per algorithm

### CI/CD Integration
- [ ] **Create CI/CD integration scripts**
  - Create GitHub Actions workflow for automated testing
  - Add test result reporting with pass/fail status
  - Generate test coverage reports
  - Archive test artifacts (logs, screenshots, performance data)
  - Set up performance regression detection
  - Add notification system for test failures

## 🔧 Technical Infrastructure Improvements

### Test Framework Enhancements
- [ ] **Add parallel test execution support**
  - Implement test isolation to enable parallel runs
  - Add test dependency management
  - Support for distributed testing across multiple machines
  - Load balancing for performance tests

- [ ] **Enhance test configuration system**
  - Add test parameter validation
  - Support for environment-specific configurations
  - Add test filtering by tags, duration, or resource requirements
  - Implement test retry logic for flaky tests

- [ ] **Improve test reporting and analytics**
  - Create HTML test report generation
  - Add performance trend analysis over time
  - Implement test result comparison between builds
  - Create dashboards for monitoring test health

### Documentation & Maintenance
- [ ] **Add comprehensive test documentation**
  - Document test writing guidelines
  - Add troubleshooting guide for common test failures
  - Create developer onboarding guide for testing system
  - Document performance baseline expectations

- [ ] **Create test maintenance tools**
  - Add tools for bulk test video regeneration
  - Create utilities for test data cleanup
  - Implement test result archival system
  - Add performance baseline update tools

## 🎯 Specialized Testing Areas

### Algorithm-Specific Testing
- [ ] **Keyframe-Sync Algorithm Testing**
  - Validate switches occur exactly at keyframes
  - Test different GOP sizes and structures
  - Verify temporal alignment accuracy
  - Test edge cases near video boundaries

- [ ] **Predecoded Algorithm Testing**  
  - Validate simultaneous dual-stream decoding
  - Test memory usage scaling with video resolution
  - Verify zero-latency switching accuracy
  - Test synchronization between parallel streams

- [ ] **Immediate Algorithm Testing**
  - Measure seek operation latency
  - Validate frame accuracy after seeks
  - Test with different video container formats
  - Verify playback resume timing

### Cross-Platform Testing
- [ ] **Multi-GPU Testing**
  - Test NVIDIA, AMD, and Intel GPU compatibility
  - Validate hardware acceleration across vendors
  - Test multi-monitor scenarios
  - Verify performance scaling with GPU capabilities

- [ ] **Resolution and Format Matrix Testing**
  - Automated testing across all supported resolutions (720p to 8K)
  - Test different codecs (H.264, H.265, AV1)
  - Validate different container formats (MP4, MKV)
  - Test HDR and color space handling

## 🚀 Future Enhancements

### Advanced Features Testing
- [ ] **Camera System Integration Testing**
  - Test face detection accuracy with test datasets
  - Validate automatic switching based on face detection
  - Test camera input edge cases and failures
  - Benchmark computer vision performance impact

- [ ] **Render Pass Pipeline Testing**
  - Validate all post-processing effects
  - Test render pass chain configuration
  - Verify performance impact of effect combinations
  - Test effect parameter ranges and edge values

- [ ] **UI System Testing**
  - Test ImGui overlay functionality
  - Validate notification system behavior
  - Test window management edge cases
  - Verify input handling across different scenarios

---

## Implementation Priority

**✅ FOUNDATIONAL PHASES COMPLETED:**

1. **Phase 1**: ~~JSON configuration system~~ ✅ **COMPLETED** 
2. **Phase 2**: ~~Video system dependency resolution (enable TestRunner, FrameValidator compilation)~~ ✅ **COMPLETED** 

**🎯 NEXT PHASES FOR COMPREHENSIVE TESTING:**

3. **Phase 3**: Core frame extraction and OCR integration (enables basic frame validation)
4. **Phase 4**: Video system integration (enables realistic testing scenarios)
5. **Phase 5**: Performance monitoring (enables comprehensive benchmarking)
6. **Phase 6**: Visual regression and error handling (ensures robustness)
7. **Phase 7**: CI/CD integration (enables automated testing)

**Foundation Complete:** The modular architecture refactoring has successfully completed Phases 1-2, establishing a solid foundation where all test system components (TestRunner, FrameValidator, SwitchingValidator, PerformanceBenchmark) are fully operational and ready for development.

Each remaining phase builds upon this foundation, creating a progressively more capable and comprehensive testing system that validates all aspects of the dual-stream video player's functionality.