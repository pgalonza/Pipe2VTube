# MediaPipe to VTude Studio Integration plugin

## Overview

The project implements a plugin for VTube Studio that integrates MediaPipe via WebSocket API to control Live2D avatars using webcam data.

> [!NOTE]
> My first project done using vibcoding

## Key Features

* **Head Tracking** - Detects and transmits head rotation angles
* **Facial Expressions** - Tracks and controls:
  * Mouth movements
  * Smile detection
  * Eyelid movements
  * Eyebrow movements

## Prerequisites

* Python 3.14+
* VTube Studio installed and running
* Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/live2d-avatar-controller.git
```

2. Download FaceLandmarker

```bash
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

3. Install dependencies:
```bash
uv sync
```

## Usage

1. Run the script:
```bash
uv run python -m src.main
```

## Configuration Options

- --host         VTube Studio host (default: localhost)
- --port         WebSocket port (default: 8001)
- --camera       Camera device ID (default: 0)
- --fps          Camera frames per second (default: 30)
- --debug        Enable debug mode with face landmarks visualization
- --no-vtube     Run in standalone debug mode without VTube Studio connection

## Development Tools

This project was developed using:

- Gigacode (logic)
- Source Craft Assistant (optimization and refactoring)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
