"""
Pipeline module for MediaPipe + VTube Studio integration.
Implements an asynchronous pipeline to process video frames through stages:
1. Frame capture
2. Face tracking and data extraction
3. Parameter mapping
4. Injection to VTube Studio
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.camera import generate_frames
from src.facetracker import FaceTracker
from src.vtube_client import VTubeStudioClient
from src.optimized_parameter_mapper import optimized_mapper

logger = logging.getLogger(__name__)


async def capture_stage(device_id: int, fps: int) -> AsyncGenerator[Tuple[bool, Optional[np.ndarray]], None]:
    """
    Stage 1: Capture video frames from the camera.

    Yields:
        Tuple of (success, frame) where frame is None if not successful.
    """
    frame_generator = generate_frames(device_id=device_id, fps=fps)
    async for success, frame in async_generator_wrapper(frame_generator):
        yield success, frame


async def tracking_stage(
    frame_stream: AsyncGenerator[Tuple[bool, Optional[np.ndarray]], None],
    tracker: FaceTracker,
    draw_landmarks: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stage 2: Process frames for face tracking and extract data.

    Args:
        frame_stream: Async generator of frames from capture_stage.
        tracker: FaceTracker instance.
        draw_landmarks: If True, draw landmarks on frames for debug.

    Yields:
        Dictionary with face data including 'face_data', 'display_frame', 'success'.
    """
    async for success, frame in frame_stream:
        if not success or frame is None:
            yield {"success": False, "face_data": None, "display_frame": None}
            continue
        
        # Process frame for face tracking
        face_data, display_frame = tracker.process_frame(frame, draw_landmarks=draw_landmarks)
        
        yield {
            "success": True,
            "face_data": face_data,
            "display_frame": display_frame
        }


async def mapping_stage(
    tracking_stream: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stage 3: Transform MediaPipe data into VTube Studio parameters.

    Args:
        tracking_stream: Async generator from tracking_stage.
    Yields:
        Dictionary with 'parameters', 'face_found', 'display_frame'.
    """
    async for item in tracking_stream:
        if not item["success"]:
            yield {
                "parameters": {},
                "face_found": False,
                "display_frame": item["display_frame"]
            }
            continue
            
        face_data = item["face_data"]
        display_frame = item["display_frame"]
        
        if face_data is None:
            yield {
                "parameters": {},
                "face_found": False,
                "display_frame": display_frame
            }
        else:
            # Extract parameters for VTube Studio using optimized mapper
            # Prepare MediaPipe data for the optimized mapper
            mediapipe_data = {
                "landmarks": face_data.get("landmarks"),
                "blendshapes": face_data.get("blendshapes"),
                "pose": face_data.get("pose")
            }
            
            # Transform and optimize parameters
            parameters = optimized_mapper.transform_and_optimize(mediapipe_data)
            
            yield {
                "parameters": parameters,
                "face_found": True,
                "display_frame": display_frame
            }


async def injection_stage(
    mapping_stream: AsyncGenerator[Dict[str, Any], None],
    client: Optional[VTubeStudioClient],
    fps: int,
    draw_landmarks: bool = False,
    enable_memory_optimization: bool = True
):
    """
    Stage 4: Inject parameters to VTube Studio.

    Args:
        mapping_stream: Async generator from mapping_stage.
        client: VTubeStudioClient instance or None.
        fps: Target FPS for camera (used for timing).
        draw_landmarks: If True, show debug window with landmarks.
    """
    # Import performance monitor
    from src.performance_monitor import performance_monitor
    
    # Timing control
    target_frame_time = 1.0 / fps if fps > 0 else 0.016  # Target time per frame
    last_injection_time = 0.0
    
    # Adaptive frame rate control
    injection_times = []  # Keep last 10 injection times for performance monitoring
    max_injection_times = 10
    frame_skip_counter = 0
    max_frame_skip = 3  # Maximum frames to skip when VTube Studio is busy
    
    frame_counter = 0
    gc_interval = 300  # Run garbage collection every 300 frames
    last_gc_time = 0.0
    gc_interval_time = 30.0  # Run garbage collection every 30 seconds minimum
    
    async for item in mapping_stream:
        # Start performance monitoring for this frame
        performance_monitor.start_frame()
        
        parameters = item["parameters"]
        face_found = item["face_found"]
        display_frame = item["display_frame"]
        frame_counter += 1
        
        # Memory optimization - periodic garbage collection
        if enable_memory_optimization and frame_counter % gc_interval == 0:
            import gc
            current_time = asyncio.get_event_loop().time()
            # Run garbage collection if enough time has passed
            if (current_time - last_gc_time) > gc_interval_time:
                collected = gc.collect()
                last_gc_time = current_time
                logger.debug("Memory optimization: Collected %d objects", collected)
        
        # Show debug window if draw_landmarks is requested
        if draw_landmarks and display_frame is not None:
            cv2.imshow('Face Tracking Debug', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Skip injection if client is None
        if client is None:
            await asyncio.sleep(0)  # Allow other tasks to run
            continue
            
        # Adaptive frame skipping based on VTube Studio performance
        # Calculate average injection time to determine if we should skip frames
        if injection_times:
            avg_injection_time = sum(injection_times) / len(injection_times)
            # If average injection time is too high, skip frames
            if avg_injection_time > target_frame_time * 2:
                frame_skip_counter += 1
                if frame_skip_counter <= max_frame_skip:
                    logger.debug("Skipping frame due to high injection time: %.3fs", avg_injection_time)
                    # Record skipped frame in performance monitor
                    performance_monitor.end_frame(skipped=True)
                    await asyncio.sleep(0)  # Allow other tasks to run
                    continue
            else:
                frame_skip_counter = 0  # Reset skip counter when performance is good
        else:
            frame_skip_counter = 0  # Reset skip counter
            
        # Timing control - ensure we don't inject too frequently
        current_time = asyncio.get_event_loop().time()
        elapsed_since_last = current_time - last_injection_time
        
        # If we're running too fast, add a small delay
        if elapsed_since_last < target_frame_time:
            sleep_time = target_frame_time - elapsed_since_last
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Update last injection time
        last_injection_time = asyncio.get_event_loop().time()
        
        # Measure injection time for adaptive control
        injection_start_time = asyncio.get_event_loop().time()
        
        # Ensure parameters is not empty when face is found
        if face_found and not parameters:
            logger.warning("No parameters to inject, but face is found. Sending minimal parameters to avoid API error.")
            # Send minimal default parameters to avoid API error
            parameters = {"FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0, "MouthOpen": 0.0, "EyeOpenLeft": 0.5, "EyeOpenRight": 0.5}
            face_found = True  # Ensure face is still reported as found
        
        success = await client.inject_parameters(parameters, face_found=face_found)
        
        # Calculate and store injection time for adaptive control
        injection_end_time = asyncio.get_event_loop().time()
        injection_time = injection_end_time - injection_start_time
        
        # Keep only the last N injection times for performance monitoring
        injection_times.append(injection_time)
        if len(injection_times) > max_injection_times:
            injection_times.pop(0)
        
        # Record injection time in performance monitor
        performance_monitor.record_injection_time(injection_time)
        
        # Log performance metrics periodically
        if len(injection_times) % 5 == 0 and injection_times:
            avg_time = sum(injection_times) / len(injection_times)
            logger.debug("Average injection time: %.3fs (target: %.3fs)", avg_time, target_frame_time)
        
        if not success:
            logger.warning("Failed to inject parameters to VTube Studio")
            performance_monitor.record_api_error()
        
        # End performance monitoring for this frame
        performance_monitor.end_frame(skipped=False)
        
        # Log performance metrics periodically
        if performance_monitor.should_log():
            performance_monitor.log_performance()


async def async_generator_wrapper(sync_gen):
    """Wrapper to use synchronous generator in async context."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Run next() in thread pool to avoid blocking
            try:
                item = await loop.run_in_executor(None, next, sync_gen)
            except StopIteration:
                break
            yield item
    except Exception as e:
        logger.error("Error in async_generator_wrapper: %s", e)
        raise