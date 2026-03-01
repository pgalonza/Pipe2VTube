# MediaPipe to VTube Studio Integration Plugin

## Overview

The project implements a plugin for VTube Studio that integrates MediaPipe via WebSocket API to control Live2D avatars using webcam data. This solution provides real-time facial tracking and expression mapping to create immersive avatar experiences.

> [!NOTE]
> This project was developed using advanced coding tools and optimization techniques.

## Key Features

- [x] FacePositionX
- [x] FacePositionY
- [x] FacePositionZ
- [x] FaceAngleX
- [x] FaceAngleY
- [x] FaceAngleZ
- [x] MouthSmile
- [x] MouthOpen
- [x] Brows
- [ ] MousePositionX
- [ ] MousePositionY
- [x] TongueOut
- [x] EyeOpenLeft
- [x] EyeOpenRight
- [x] EyeLeftX
- [x] EyeLeftY
- [x] EyeRightX
- [x] EyeRightY
- [x] CheekPuff
- [x] BrowLeftY
- [x] BrowRightY
- [x] MouthX
- [ ] FaceAngry

## Prerequisites

* Python 3.14+
* VTube Studio installed and running
* Webcam with at least 720p resolution recommended
* FaceLandmarker model file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pgalonza/Pipe2VTube
cd Pipe2VTube
```

2. Download FaceLandmarker model:
```bash
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

3. Install dependencies:
```bash
uv sync --extra ui
```

## Usage

1. Run the application:
```bash
uv run python -m src.main
```

2. For first-time use, the system will automatically run eye calibration:
   - Keep your eyes OPEN when prompted
   - Follow instructions to close your eyes when prompted
   - Calibration data is saved for future use

## Configuration Options

- `--host`         VTube Studio host (default: localhost)
- `--port`         WebSocket port (default: 8001)
- `--camera`       Camera device ID (default: 0)
- `--fps`          Camera frames per second (default: 30)
- `--debug`        Enable debug mode with face landmarks visualization
- `--no-vtube`     Run in standalone debug mode without VTube Studio
- `--calibrate`    Force eye calibration even if calibration file exists

## Advanced Features

### Eye Calibration
The system automatically calibrates eye tracking for optimal accuracy:
- Measures eye openness in both open and closed states
- Creates personalized thresholds for each user
- Saves calibration data for future sessions
- Provides visual feedback during calibration process

### Position Calibration
Automatically calibrates face position tracking:
- Establishes neutral head position as reference point
- Tracks movements relative to initial position
- Automatically resets calibration when face is not detected

### Performance Optimization
- Optimized parameter mapping with change detection
- Parameter batching to reduce API overhead
- Real-time performance monitoring
- Adaptive frame rate control

## Troubleshooting

### Common Issues

1. **VTube Studio connection failed**:
   - Ensure VTube Studio is running
   - Check that API access is enabled in VTube Studio
   - Verify host and port settings

2. **No face detected**:
   - Check camera permissions
   - Ensure proper lighting conditions
   - Verify camera is not covered

3. **Poor tracking quality**:
   - Ensure good lighting conditions
   - Position face in center of frame
   - Clean camera lens

4. **Calibration issues**:
   - Use `--calibrate` flag to force recalibration
   - Follow visual instructions during calibration
   - Keep head still during calibration process

### Performance Tips

- Use a high-quality webcam for best results
- Ensure adequate lighting (avoid backlighting)
- Position camera at eye level
- Maintain consistent distance from camera

## Development Tools

This project was developed using:

- Gigacode (logic)
- Source Craft Assistant (optimization and refactoring)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
