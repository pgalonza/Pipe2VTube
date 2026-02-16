"""
This module is the single source of truth for eye calibration logic.
All helper functions have been moved from calibration_helper.py.
"""

from typing import Dict
import logging
import numpy as np
import time
import asyncio
import json
import os
logger = logging.getLogger(__name__)


class EyeCalibrator:
    """
    Class for automatic calibration of eye opening thresholds.
    Collects data on the distance between eyelids in open and closed
    eye states.
    """
    def __init__(self):
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_duration = 3.0  # секунды
        self.open_distances_left = []
        self.open_distances_right = []
        self.min_distances_left = []
        self.min_distances_right = []
        self.is_open_phase = True  # True: собираем max, False: собираем min
        self._load_calibration_data()

    def start_calibration(self):
        """Start the calibration process"""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.open_distances_left.clear()
        self.open_distances_right.clear()
        self.min_distances_left.clear()
        self.min_distances_right.clear()
        self.is_open_phase = True
        logger.info("Eye calibration process started. Keep your eyes open.")

    def update(self, landmarks: list) -> bool:
        """
        Update calibration data based on current landmarks.
        Returns True when calibration is complete.
        
        Args:
            landmarks: List of face landmarks from MediaPipe.

        Returns:
            bool: True if calibration is complete.
        """
        if not self.is_calibrating:
            return False

        current_time = time.time()
        elapsed = current_time - (self.calibration_start_time or 0)

        # Determine distances
        # MediaPipe right eye -> model left eye
        dist_left = _calculate_eye_openness(landmarks, (386, 374))
        # MediaPipe left eye -> model right eye
        dist_right = _calculate_eye_openness(landmarks, (159, 145))

        # Always add data if calibration is active
        self.open_distances_left.append(dist_left)
        self.open_distances_right.append(dist_right)

        if not self.is_open_phase:
            self.min_distances_left.append(dist_left)
            self.min_distances_right.append(dist_right)

        # Check if calibration is complete by timer
        if (elapsed >= self.calibration_duration and
                self.calibration_start_time is not None):
            if self.is_open_phase:
                # Switch to closed eyes phase
                self.is_open_phase = False
                self.calibration_start_time = current_time
                logger.info("Now close your eyes for minimum calibration.")
                return False
            else:
                # Complete calibration
                self._finalize_calibration()
                return True

        return False

    def _finalize_calibration(self):
        """Finalize calibration: calculate thresholds"""
        # Calculate average values
        max_left = np.mean(self.open_distances_left) if self.open_distances_left else 0.038
        min_left = np.mean(self.min_distances_left) if self.min_distances_left else 0.012
        max_right = np.mean(self.open_distances_right) if self.open_distances_right else 0.038
        min_right = np.mean(self.min_distances_right) if self.min_distances_right else 0.012

        # Apply multiplier for reliability
        max_left *= 1.1
        max_right *= 1.1

        # Save as attributes
        self.EYE_OPEN_CALIBRATED_MAX_LEFT = max_left
        self.EYE_OPEN_CALIBRATED_MIN_LEFT = min_left
        self.EYE_OPEN_CALIBRATED_MAX_RIGHT = max_right
        self.EYE_OPEN_CALIBRATED_MIN_RIGHT = min_right

        logger.info("Calibration completed")
        logger.info("  Left eye: MAX=%.4f, MIN=%.4f" %
            (self.EYE_OPEN_CALIBRATED_MAX_LEFT, self.EYE_OPEN_CALIBRATED_MIN_LEFT))
        logger.info("  Right eye: MAX=%.4f, MIN=%.4f" %
            (self.EYE_OPEN_CALIBRATED_MAX_RIGHT, self.EYE_OPEN_CALIBRATED_MIN_RIGHT))

        self.is_calibrating = False

    def _load_calibration_data(self):
        """Load calibration data from file if it exists."""
        try:
            if os.path.exists("eye_calibration.json"):
                with open("eye_calibration.json", "r") as f:
                    data = json.load(f)
                    self.EYE_OPEN_CALIBRATED_MAX_LEFT = data["left_max"]
                    self.EYE_OPEN_CALIBRATED_MIN_LEFT = data["left_min"]
                    self.EYE_OPEN_CALIBRATED_MAX_RIGHT = data["right_max"]
                    self.EYE_OPEN_CALIBRATED_MIN_RIGHT = data["right_min"]
                    logger.info("Loaded previous calibration data from file")
        except Exception as e:
            logger.warning(f"Could not load calibration data: {e}")

    def _save_calibration_data(self):
        """Save calibration data to file."""
        try:
            data = {
                "left_max": float(self.EYE_OPEN_CALIBRATED_MAX_LEFT),
                "left_min": float(self.EYE_OPEN_CALIBRATED_MIN_LEFT),
                "right_max": float(self.EYE_OPEN_CALIBRATED_MAX_RIGHT),
                "right_min": float(self.EYE_OPEN_CALIBRATED_MIN_RIGHT)
            }
            with open("eye_calibration.json", "w") as f:
                json.dump(data, f)
            logger.info("Saved calibration data to file")
        except Exception as e:
            logger.warning(f"Could not save calibration data: {e}")

    def get_thresholds(self) -> Dict[str, float]:
        """Get current calibration thresholds.
        
        Returns:
            Dict: Dictionary with threshold values for left and right eyes.
            Returns default values if calibration has not been performed.
        """
        if not self.is_calibrating:
            try:
                if hasattr(self, 'EYE_OPEN_CALIBRATED_MAX_LEFT'):
                    return {
                        'left_max': float(self.EYE_OPEN_CALIBRATED_MAX_LEFT),
                        'left_min': float(self.EYE_OPEN_CALIBRATED_MIN_LEFT),
                        'right_max': float(self.EYE_OPEN_CALIBRATED_MAX_RIGHT),
                        'right_min': float(self.EYE_OPEN_CALIBRATED_MIN_RIGHT)
                    }
                else:
                    logger.warning("Calibration values not initialized, using default values.")
                    return {
                        'left_max': 0.038,
                        'left_min': 0.012,
                        'right_max': 0.038,
                        'right_min': 0.012
                    }
            except Exception as e:
                logger.error(f"Error getting calibration values: {e}")
                return {
                    'left_max': 0.038,
                    'left_min': 0.012,
                    'right_max': 0.038,
                    'right_min': 0.012
                }
        return {
            'left_max': 0.038,
            'left_min': 0.012,
            'right_max': 0.038,
            'right_min': 0.012
        }


calibrator = EyeCalibrator()


def _calculate_eye_openness(landmarks: list, indices: tuple) -> float:
    """
    Calculate eye openness from two points (upper/lower).
    
    Args:
        landmarks: List of MediaPipe points (NormalizedLandmarkList)
        indices: (idx_upper, idx_lower) - indices of upper and lower eyelid points
    
    Returns:
        float: Distance between points, representing eye openness
    """
    idx_upper, idx_lower = indices
    try:
        upper = landmarks[idx_upper]
        lower = landmarks[idx_lower]
        # Extract x, y, z coordinates from NormalizedLandmark
        upper_coord = np.array([upper.x, upper.y, upper.z])
        lower_coord = np.array([lower.x, lower.y, lower.z])
        distance = np.linalg.norm(upper_coord - lower_coord)
        return float(distance)
    except (IndexError, AttributeError):
        return 0.0


def get_default_eye_thresholds() -> Dict[str, float]:
    """Get default thresholds for eye openness.
    
    Returns:
        Dict[str, float]: Dictionary with default thresholds for normalization.
    """
    return {
        'left_max': 0.038,
        'left_min': 0.012,
        'right_max': 0.038,
        'right_min': 0.012
    }


class EyeCalibrationHelper:
    """
    Helper class for managing the eye calibration process with user interaction.
    """
    def __init__(self, calibrator=None):
        """
        Initialize the calibration helper.
        
        Args:
            calibrator: Instance of EyeCalibrator to use for calibration.
                       If None, uses the global default instance from eye_calibrator module.
        """
        if calibrator is None:
            calibrator = globals()['calibrator']  # Use the global default instance
        self.calibrator = calibrator
        # We need a tracker instance to process frames during calibration
        from src.facetracker import FaceTracker
        self.tracker = FaceTracker()

    async def run_calibration(self) -> bool:
        """
        Run the complete eye calibration process with user guidance.
        The camera is activated only during the calibration process.
        
        Returns:
            bool: True if calibration completed successfully, False otherwise.
        """
        try:
            # Initial prompt
            logger.info("Starting automatic eye calibration...")
            # Start camera for calibration
            from src.camera import generate_frames
            frame_generator = generate_frames(device_id=0, width=640, height=480, fps=30)
            
            # Open eyes phase
            logger.info("Keep your eyes OPEN for automatic calibration...")
            input()
            self.calibrator.start_calibration()
            
            # Capture frames for 3 seconds for open eyes
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    success, frame = next(frame_generator, (False, None))
                    if success and frame is not None:
                        # Extract landmarks from frame
                        import cv2
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        import mediapipe as mp
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        detection_result = self.tracker.detector.detect(image)
                        if detection_result.face_landmarks:
                            landmarks = detection_result.face_landmarks[0]
                            self.calibrator.update(landmarks)
                    await asyncio.sleep(1/30)  # Simulate FPS
                except StopIteration:
                    break
            
            # Closed eyes phase
            logger.info("Keep your eyes CLOSED for automatic calibration...")
            input()
            # Switch to closed eyes phase
            self.calibrator.is_open_phase = False
            self.calibrator.calibration_start_time = time.time()
            
            # Capture frames for 3 seconds for closed eyes
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    success, frame = next(frame_generator, (False, None))
                    if success and frame is not None:
                        # Extract landmarks from frame
                        import cv2
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        import mediapipe as mp
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        detection_result = self.tracker.detector.detect(image)
                        if detection_result.face_landmarks:
                            landmarks = detection_result.face_landmarks[0]
                            self.calibrator.update(landmarks)
                    await asyncio.sleep(1/30)  # Simulate FPS
                except StopIteration:
                    break
            
            # Finalize calibration
            logger.info("Eye calibration completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration process: {e}")
            return False
        
