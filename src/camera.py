"""
Module for capturing video from webcam with preprocessing.
"""

import cv2
import logging
from typing import Generator, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Context manager for camera resources to ensure proper cleanup.
    """
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, 
                 fps: int = 30):
        """
        Initialize camera manager.
        
        Args:
            device_id: Camera device ID.
            width: Frame width.
            height: Frame height.
            fps: Target FPS for camera.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def __enter__(self):
        """Initialize and return the camera capture object."""
        try:
            # Open the camera
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return None

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set preferred codec to MJPEG for better performance
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # Verify and log actual camera capabilities
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
            actual_fourcc_str = ''.join([
                chr((int(actual_fourcc) >> 8 * i) & 0xFF) 
                for i in range(4)
            ])
            logger.info(
                "Camera actual settings: %dx%d @ %.1f FPS, Codec: %s",
                actual_width, actual_height, actual_fps, actual_fourcc_str
            )

            logger.info(
                "Camera initialized: %dx%d @ %d FPS",
                self.width, self.height, self.fps
            )
            return self.cap
            
        except Exception as e:
            logger.error("Error initializing camera: %s", e)
            if self.cap:
                self.release()
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera resources."""
        self.release()
        
    def release(self):
        """Release camera resources."""
        if self.cap:
            try:
                self.cap.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error("Error releasing camera: %s", e)
            finally:
                self.cap = None


def generate_frames(device_id: int = 0, width: int = 640, height: int = 480, 
                    fps: int = 30) -> Generator[Tuple[bool, Optional[Any]], None, None]:
    """
    Generator that yields preprocessed video frames from the webcam.
    The camera is opened and closed on-demand for each session.

    Args:
        device_id: Camera device ID.
        width: Frame width.
        height: Frame height.
        fps: Target FPS for camera.

    Yields:
        Tuple of (success, frame) where frame is None if not successful.
    """
    with CameraManager(device_id, width, height, fps) as cap:
        if cap is None:
            logger.error("Failed to initialize camera")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    yield False, None
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                yield True, frame
        except Exception as e:
            logger.error("Error during frame capture: %s", e)
            yield False, None
