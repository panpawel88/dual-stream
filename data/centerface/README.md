# CenterFace Face Detection Model

## Overview

CenterFace is a modern face detection algorithm that treats face detection as a center point estimation problem. It's been successfully integrated into the DualStream Video Player's face detection system.

## Implementation Status

✅ **COMPLETED**
- Added `CENTERFACE` algorithm to `FaceDetectionAlgorithm` enum
- Implemented `DetectFacesCenterFace()` method with full preprocessing and postprocessing
- Added CenterFace-specific configuration parameters
- Updated configuration parsing to handle "centerface" algorithm string
- Added default configuration values in Config.cpp
- Created dedicated preprocessing method `PreprocessFrameCenterFace()`
- Implemented complex postprocessing with Non-Maximum Suppression
- Full integration with existing video switching system

## Configuration

The CenterFace algorithm can be configured in the `default.ini` file or through the global configuration system:

```ini
[face_detection]
algorithm = centerface
centerface_input_width = 640
centerface_input_height = 640
centerface_score_threshold = 0.5
centerface_nms_threshold = 0.2
```

## Model Requirements

✅ **AUTOMATIC DOWNLOAD**: The CenterFace ONNX model is automatically downloaded during build!

1. **Model File**: `centerface.onnx` (7.5MB)
2. **Location**: Automatically placed in runtime directory (`data/centerface/centerface.onnx`)
3. **Source**: Downloaded from official Star-Clouds/CenterFace repository on GitHub
4. **Build Integration**: Handled by CMakeLists.txt just like Haar cascades and YuNet models

## Usage

Once the model is available, CenterFace can be used by:

1. **Configuration**: Set `face_detection.algorithm = "centerface"` in config
2. **Command Line**: Run with face detection trigger: `--trigger=face-detection`
3. **Camera Integration**: Automatically works with camera system

## Technical Details

- **Input Size**: 640x640 pixels (configurable)
- **Outputs**: 4 tensors (heatmap, scale, offset, landmarks)
- **Postprocessing**: Grid-based decoding with NMS filtering
- **Performance**: Modern DNN-based approach with good accuracy/speed balance
- **Integration**: Fully compatible with existing switching logic and preview system

## Model Architecture

CenterFace outputs 4 tensors:
1. **Heatmap** (1,1,H/4,W/4): Face center confidence scores
2. **Scale** (1,2,H/4,W/4): Bounding box width and height scales
3. **Offset** (1,2,H/4,W/4): Center point offsets
4. **Landmarks** (1,10,H/4,W/4): Facial landmark coordinates (5 points)

The implementation uses only the first 3 tensors for face detection and bounding box generation.