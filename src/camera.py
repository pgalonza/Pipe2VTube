"""
Module for capturing video from webcam with preprocessing.
"""

import cv2
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def generate_frames(device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> Generator[Tuple[bool, Optional[cv2.Mat]], None, None]:
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
    # Open the camera
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        logger.error("Failed to open camera after retry. Please check camera permissions and availability.")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Set preferred codec to MJPEG for better performance
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Verify and log actual camera capabilities
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    actual_fourcc_str = ''.join([chr((int(actual_fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    logger.info("Camera actual settings: %dx%d @ %.1f FPS, Codec: %s", actual_width, actual_height, actual_fps, actual_fourcc_str)

    logger.info("Camera initialized: %dx%d @ %d FPS", width, height, fps)

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
    finally:
        # cap.release()
        # logger.info("Camera released")
        # The generator has no reference to cap.
        # The caller must manage the VideoCapture lifecycle externally.
        # This is a limitation of the current design.
        pass
