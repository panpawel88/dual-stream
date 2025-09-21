@echo off
REM DualStream Video Player Build Script for Windows
REM Requires: Visual Studio 2022, vcpkg toolchain configured in .env.local

setlocal EnableDelayedExpansion

echo ========================================
echo DualStream Video Player Build Script
echo ========================================

REM Check if .env.local exists
if not exist ".env.local" (
    echo ERROR: .env.local file not found!
    echo Please create .env.local with: VCPKG_TOOLCHAIN_FILE=^<path-to-vcpkg^>/scripts/buildsystems/vcpkg.cmake
    exit /b 1
)

REM Read vcpkg toolchain path from .env.local
for /f "tokens=2 delims==" %%a in ('findstr "VCPKG_TOOLCHAIN_FILE" .env.local') do set VCPKG_TOOLCHAIN=%%a

if "%VCPKG_TOOLCHAIN%"=="" (
    echo ERROR: VCPKG_TOOLCHAIN_FILE not found in .env.local
    echo Please add: VCPKG_TOOLCHAIN_FILE=^<path-to-vcpkg^>/scripts/buildsystems/vcpkg.cmake
    exit /b 1
)

echo Using vcpkg toolchain: %VCPKG_TOOLCHAIN%

REM Parse command line arguments
set BUILD_CONFIG=Release
set CLEAN_BUILD=false
set FORCE_CMAKE=false
set ENABLE_TRACY=false

:parse_args
if "%1"=="" goto end_parse
if /i "%1"=="--debug" set BUILD_CONFIG=Debug
if /i "%1"=="--clean" set CLEAN_BUILD=true
if /i "%1"=="--cmake" set FORCE_CMAKE=true
if /i "%1"=="--tracy" set ENABLE_TRACY=true
shift
goto parse_args
:end_parse

echo Build configuration: %BUILD_CONFIG%

REM Clean build directory if requested
if "%CLEAN_BUILD%"=="true" (
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    mkdir build
    set FORCE_CMAKE=true
    echo Clean completed.
)

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Check if CMake configuration is needed
set NEED_CMAKE=false
if "%FORCE_CMAKE%"=="true" set NEED_CMAKE=true
if not exist "build\CMakeCache.txt" set NEED_CMAKE=true
if not exist "build\dual_stream.sln" set NEED_CMAKE=true

REM Check if CMakeLists.txt is newer than cache (CMake files changed)
if exist "build\CMakeCache.txt" (
    for %%i in ("CMakeLists.txt") do set CMAKE_TIME=%%~ti
    for %%i in ("build\CMakeCache.txt") do set CACHE_TIME=%%~ti
    REM Simple timestamp comparison - if CMakeLists.txt is newer, reconfigure
    if "!CMAKE_TIME!" GTR "!CACHE_TIME!" set NEED_CMAKE=true
)

if "%NEED_CMAKE%"=="true" (
    echo ========================================
    echo Configuring CMake...
    echo ========================================

    cd build

    REM Build CMake command with optional Tracy support
    set CMAKE_CMD=cmake -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%"
    if "%ENABLE_TRACY%"=="true" (
        echo Enabling Tracy profiler support...
        set CMAKE_CMD=!CMAKE_CMD! -DENABLE_TRACY=ON
    )
    set CMAKE_CMD=!CMAKE_CMD! ..

    echo Running: !CMAKE_CMD!
    !CMAKE_CMD!

    if errorlevel 1 (
        echo ERROR: CMake configuration failed!
        cd ..
        exit /b 1
    )
    cd ..
) else (
    echo CMake configuration up to date, skipping...
)

echo ========================================
echo Building project...
echo ========================================

REM Build the project
cmake --build build --config %BUILD_CONFIG%

if errorlevel 1 (
    echo ERROR: Build failed!
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo Executable: build\bin\%BUILD_CONFIG%\dual_stream.exe
echo.
echo Usage examples:
echo   build\bin\%BUILD_CONFIG%\dual_stream.exe video1.mp4 video2.mp4
echo   build\bin\%BUILD_CONFIG%\dual_stream.exe video1.mp4 video2.mp4 --algorithm predecoded
echo   build\bin\%BUILD_CONFIG%\dual_stream.exe video1.mp4 video2.mp4 --speed 1.5 --debug
echo.
echo Build script options:
echo   --debug     Build in Debug configuration (default: Release)
echo   --clean     Clean build directory before building
echo   --cmake     Force CMake reconfiguration
echo   --tracy     Enable Tracy profiler support (-DENABLE_TRACY=ON)
echo ========================================