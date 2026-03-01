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

## Project Architecture

### Overview
The system follows a modular pipeline architecture with clearly defined responsibilities for each component:

1. **Video Capture Layer** (`src/camera.py`) - Handles webcam initialization and frame capture
2. **Face Tracking Layer** (`src/facetracker.py`) - Processes video frames using MediaPipe to extract facial landmarks and blendshapes
3. **Data Processing Layer** (`src/parameter_mapper.py`, `src/optimized_parameter_mapper.py`) - Transforms MediaPipe data into VTube Studio parameters with optimization
4. **Communication Layer** (`src/vtube_client.py`) - Manages WebSocket connection and data transmission to VTube Studio
5. **Calibration Layer** (`src/eye_calibrator.py`, `src/position_calibrator.py`) - Handles automatic calibration of tracking parameters
6. **Monitoring Layer** (`src/performance_monitor.py`) - Tracks and reports system performance metrics
7. **Orchestration Layer** (`src/pipeline.py`, `src/main.py`) - Coordinates all components and manages data flow

### Data Flow
```
Camera Input → Face Tracking → Parameter Mapping → VTube Studio API
     ↓              ↓               ↓                   ↓
  Frame      Landmarks/     VTube Studio     WebSocket Connection
             Blendshapes    Parameters       with Authentication
```

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
     - `src/position_calibrator.py` — position calibration.
     - `pyproject.toml` with up-to-date dependencies.

## Module Responsibilities

### src/camera.py
- Initialize and manage camera resources
- Capture video frames with proper error handling
- Provide frame preprocessing (horizontal flip for mirror effect)
- Ensure proper resource cleanup

### src/facetracker.py
- Process video frames using MediaPipe FaceLandmarker
- Extract facial landmarks, blendshapes, and transformation matrices
- Calculate head pose (pitch, yaw, roll) and position (x, y, z)
- Coordinate with calibration modules for accurate tracking

### src/parameter_mapper.py
- Transform MediaPipe data into VTube Studio parameters
- Handle mapping of blendshapes to VTube Studio parameters
- Calculate eye openness from landmarks with calibration
- Implement eye direction tracking
- Calculate MouthX parameter for horizontal mouth movement

### src/optimized_parameter_mapper.py
- Optimize parameter transmission to reduce API overhead
- Implement change detection to only send significant parameter changes
- Batch parameters for efficient transmission
- Cache computations to avoid redundant processing

### src/vtube_client.py
- Manage WebSocket connection to VTube Studio
- Handle authentication and token management
- Create custom parameters in VTube Studio
- Inject parameter data with retry mechanisms
- Handle connection errors and reconnection logic

### src/eye_calibrator.py
- Automatically calibrate eye tracking parameters
- Collect data on eye openness in open and closed states
- Provide visual feedback during calibration
- Save and load calibration data
- Calibrate eye direction tracking

### src/position_calibrator.py
- Calibrate face position tracking
- Establish neutral head position as reference point
- Track movements relative to initial position
- Automatically reset calibration when needed

### src/performance_monitor.py
- Monitor system performance metrics
- Track frame processing times
- Monitor parameter injection times
- Record API errors and reconnection events
- Provide benchmarking utilities

### src/pipeline.py
- Implement asynchronous pipeline for video processing
- Coordinate data flow between all modules
- Handle timing control and frame rate management
- Manage debug visualization

### src/main.py
- Entry point for the application
- Parse command line arguments
- Initialize all components
- Start the processing pipeline
- Handle application shutdown

## API Documentation

### VTube Studio Client API
- `connect()` - Establish WebSocket connection to VTube Studio
- `authenticate()` - Authenticate with VTube Studio API
- `create_parameter()` - Create custom parameters in VTube Studio
- `inject_parameters()` - Send parameter data to VTube Studio
- `reconnect()` - Handle reconnection logic

### Parameter Mapper API
- `transform_mediapipe_to_vtubestudio()` - Transform MediaPipe data to VTube Studio parameters
- `MEDIPIPE_TO_VTUBE` - Mapping dictionary from MediaPipe blendshapes to VTube Studio parameters
- `STANDARD_VTS_PARAMS` - Set of standard VTube Studio parameters
- `POSE_MAPPING` - Mapping from MediaPipe pose data to VTube Studio parameters
- `POSITION_MAPPING` - Mapping from MediaPipe position data to VTube Studio parameters

### Eye Calibrator API
- `start_calibration()` - Begin eye calibration process
- `update()` - Update calibration data with current landmarks
- `get_thresholds()` - Get current calibration thresholds
- `calibrate_eye_direction()` - Calibrate eye direction tracking
- `calculate_calibration_quality()` - Calculate quality metrics for calibration

## Performance Optimization Guidelines

1. **Parameter Optimization**
   - Use change detection to only send significant parameter changes
   - Batch parameters to reduce API overhead
   - Cache computations to avoid redundant processing
   - Prioritize critical parameters for transmission

2. **Memory Management**
   - Use context managers for proper resource cleanup
   - Implement garbage collection in long-running processes
   - Minimize memory allocations in hot paths

3. **Timing Control**
   - Implement adaptive frame rate control
   - Use timing control to maintain consistent frame rate
   - Skip frames when VTube Studio is busy

## Testing Procedures

1. **Unit Testing**
   - Test individual modules with isolated test cases
   - Use mock objects for external dependencies
   - Verify parameter transformations and mappings
   - Test edge cases and error conditions

2. **Integration Testing**
   - Test data flow between modules
   - Verify parameter transmission to VTube Studio
   - Test calibration processes
   - Validate performance optimization features

3. **Performance Testing**
   - Benchmark parameter processing performance
   - Monitor frame processing times
   - Verify real-time performance requirements
   - Test memory usage and resource consumption

## Coding Standards

1. **Code Organization**
   - Follow modular design principles
   - Maintain clear separation of concerns
   - Use descriptive variable and function names
   - Document complex algorithms and logic

2. **Error Handling**
   - Implement proper exception handling
   - Provide meaningful error messages
   - Handle edge cases gracefully
   - Log errors for debugging purposes

3. **Performance Considerations**
   - Minimize computational overhead
   - Use efficient data structures
   - Avoid unnecessary computations
   - Profile code for performance bottlenecks

## Launch Procedure

1. Start the application:
   ```bash
   uv run python -m src.main
   ```

2. For development and testing:
   ```bash
   # Run with debug visualization
   uv run python -m src.main --debug
   
   # Run without VTube Studio connection for standalone testing
   uv run python -m src.main --no-vtube
   
   # Force eye calibration
   uv run python -m src.main --calibrate
   ```

## Development Environment Setup

1. Install Python 3.14+
2. Install uv package manager
3. Clone the repository
4. Install dependencies with `uv sync --extra ui`
5. Download FaceLandmarker model
6. Install VTube Studio and enable API access

## Contributing Guidelines

1. Follow the established code structure and patterns
2. Write unit tests for new functionality
3. Document new features and changes
4. Submit pull requests for review
5. Follow semantic versioning for releases
