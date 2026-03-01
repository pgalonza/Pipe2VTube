# AGENTS.md: Enhancing Solution for Live2D Model Control in VTube Studio

**Role**: Experienced Python developer, specialist in computer vision integration with virtual avatars.

**Task**: To enhance the existing solution for controlling **Live2D model in VTube Studio** via webcam face tracking using MediaPipe.

## Enhancement Objective

To ensure stable operation of the face tracking system and data transmission to VTube Studio via API for Live2D model control.

## Technology Stack

- **Python** — primary implementation language.
- **OpenCV** — handling webcam video stream.
- **MediaPipe** — face key points detection and tracking. The new version lacks solutions.
  *Documentation*: `./mediapipe-docs`.
- **VTube Studio** — Live2D model control.
  *Documentation*: `./vitube-studio-docs`.
- **uv** — dependency manager, virtual environments, and launch tool.
  *Dependency versions*: `./pyproject.toml`

## Enhancement Requirements

1. **Integration with VTube Studio**
   - Transmit to API:
     - head rotation angles;
     - parameters for facial expressions, eyes, mouth, eyebrows;
     - additional events via VTube Studio Event API.
   - Implemented MouthX parameter for horizontal mouth movement tracking
   - Conversion of parameters from MediaPipe to VTube Studio:
     - Parameters used in VTube Studio should be transmitted as standard parameters.
     - Other parameters, such as MediaPipe, should be transmitted as additional (custom) parameters.
   - Add API error handling (connection loss, incorrect parameters).

2. **Project Structure**  
    - `src/camera.py` — video capture and preprocessing.
    - `src/facetracker.py` — face analysis with MediaPipe.
    - `src/vtube_client.py` — interaction with VTube Studio API.
    - `src/main.py` — entry point, module coordination.
    - `src/parameter_mapper.py` — mapping parameters from MediaPipe to VTube Studio.
    - `src/eye_calibrator.py` — eye calibration.
    - `src/optimized_parameter_mapper.py` — optimized parameter mapping.
    - `src/performance_monitor.py` - performance monitoring.
    - `src/pipeline.py` — video processing pipeline.
    - `pyproject.toml` with up-to-date dependencies.

## Launch Procedure

1. Start the application:
   ```bash
   uv run python src/main.py
