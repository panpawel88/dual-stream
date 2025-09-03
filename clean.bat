@echo off
REM DualStream Video Player Clean Script
REM Removes all build artifacts and forces fresh CMake configuration

echo ========================================
echo DualStream Video Player Clean Script
echo ========================================

if exist build (
    echo Removing build directory...
    rmdir /s /q build
    echo Build directory removed.
) else (
    echo Build directory not found, nothing to clean.
)

echo ========================================
echo Clean completed!
echo ========================================
echo Run "build.bat" to rebuild the project from scratch.
echo ========================================