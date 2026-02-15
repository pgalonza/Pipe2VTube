"""
Main entry point for MediaPipe + VTube Studio integration.
Coordinates the face tracking pipeline.
"""

import asyncio
import logging
import argparse

# Import local modules
from src.facetracker import FaceTracker
from src.vtube_client import VTubeStudioClient
from src.pipeline import capture_stage, tracking_stage, mapping_stage, injection_stage
from src.parameter_mapper import MEDIPIPE_TO_VTUBE, STANDARD_VTS_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main(host: str = "localhost", port: int = 8001, camera_id: int = 0, fps: int = 30):
    """
    Main application loop.

    Args:
        host: VTube Studio host.
        port: VTube Studio WebSocket port.
        camera_id: Camera device ID.
        fps: Target FPS for camera.
    """
    # Initialize components
    tracker = FaceTracker()
    
    # Run eye calibration
    from src.eye_calibrator import calibrator
    from src.eye_calibrator import EyeCalibrationHelper
    calibration_helper = EyeCalibrationHelper(calibrator)
    if not calibration_helper.run_calibration():
        logger.error("Калибровка не удалась, продолжение с настройками по умолчанию.")
    
    logger.info("Старт основной части программы")
    
    # Initialize VTube Studio client only if --no-vtube is not specified
    client = None
    if not args.no_vtube:
        client = VTubeStudioClient(host=host, port=port)
        # Connect to VTube Studio
        if not await client.connect():
            logger.error("Failed to connect to VTube Studio. Make sure VTube Studio is running and API access is enabled.")
            return
        
        if not await client.authenticate():
            logger.error("Failed to authenticate with VTube Studio. Check plugin permissions.")
            return
        
        # TODO need to refactoring add custom parameters
        # Create custom parameters in VTube Studio if they don't exist
        # Create custom parameters in VTube Studio if they don't exist
        # Use central list of standard VTube Studio parameters from parameter_mapper
        for vts_param_name in [v for k, v in MEDIPIPE_TO_VTUBE.items() if v not in STANDARD_VTS_PARAMS]:
            success = await client.create_parameter(vts_param_name)
            if not success:
                logger.warning(f"Failed to create parameter {vts_param_name}. It may already exist.")
            
        # Add delay after creating parameters to ensure they are registered
        await asyncio.sleep(1.0)  # Small delay to ensure connection is stable
    
    logger.info("Starting face tracking pipeline...")
    
    # Create the pipeline stages
    frame_stream = capture_stage(device_id=camera_id, fps=fps)
    tracking_stream = tracking_stage(frame_stream, tracker, draw_landmarks=args.debug)
    mapping_stream = mapping_stage(tracking_stream)
    
    # Run the injection stage
    try:
        await injection_stage(mapping_stream, client, fps, draw_landmarks=args.debug)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Unexpected error in main loop: %s", e, exc_info=True)
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception as e:
                logger.error("Error during client cleanup: %s", e)

async def async_generator_wrapper(sync_gen):
    """Wrapper to use synchronous generator in async context."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Run next() in thread pool to avoid blocking
            item = await loop.run_in_executor(None, lambda: next(sync_gen, (False, None)))
            if item == (False, None):
                break
            yield item
    except StopIteration:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live2D avatar control via webcam")
    parser.add_argument("--host", type=str, default="localhost", help="VTube Studio host")
    parser.add_argument("--port", type=int, default=8001, help="VTube Studio WebSocket port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with face landmarks visualization")
    parser.add_argument("--no-vtube", action="store_true", help="Disable VTube Studio connection and run in standalone debug mode")
    
    args = parser.parse_args()
    
    # Run the async main function
    try:
        asyncio.run(main(args.host, args.port, args.camera, args.fps))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        
    # Ensure clean shutdown
    import sys
    sys.exit(0)  # Already closed in main's finally block
