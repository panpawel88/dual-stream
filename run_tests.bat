@echo off
echo ========================================
echo DualStream Video Player Test Runner
echo ========================================
echo.

REM Check if test videos exist, generate them if needed
if not exist "test_videos\video1_red_square.mp4" (
    echo Test videos not found, generating them first...
    call generate_test_videos.bat
)

REM Build the project including tests
echo Building project with test system...
call build.bat --cmake

if %errorlevel% neq 0 (
    echo ERROR: Build failed
    exit /b 1
)

REM Check if test runner was built
if not exist "build\bin\Release\test_runner.exe" (
    echo ERROR: Test runner executable not found at build\bin\Release\test_runner.exe
    echo Make sure ENABLE_TESTS is ON in CMakeLists.txt
    exit /b 1
)

echo.
echo ========================================
echo Running Basic Frame Validation Tests
echo ========================================
echo.

build\bin\Release\test_runner.exe --suite frame_validation_basic --verbose

echo.
echo ========================================
echo Running Switching Accuracy Tests
echo ========================================
echo.

build\bin\Release\test_runner.exe --suite switching_accuracy --verbose

echo.
echo ========================================
echo Running Performance Benchmarks
echo ========================================
echo.

build\bin\Release\test_runner.exe --suite performance_benchmarks --verbose

echo.
echo ========================================
echo Test Summary
echo ========================================
echo.

if exist "test_results.json" (
    echo Test results saved to test_results.json
    echo.
    echo To run specific tests:
    echo   build\bin\Release\test_runner.exe --suite frame_validation_basic
    echo   build\bin\Release\test_runner.exe --test hd_30fps_performance
    echo   build\bin\Release\test_runner.exe --help
) else (
    echo WARNING: test_results.json not found - tests may have failed
)

echo.
pause