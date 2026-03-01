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
import mediapipe as mp

logger = logging.getLogger(__name__)

# Try to import cv2, but handle case where it's not available
try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV (cv2) not available, visual feedback will be disabled")


class VisualFeedback:
    """
    Class for providing visual feedback during eye calibration.
    """
    def __init__(self, window_name="Eye Calibration"):
        self.window_name = window_name
        if cv2 is not None:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
    def show_instruction(self, text: str, frame_height: int = 480, frame_width: int = 640):
        """Display instruction text during calibration."""
        if cv2 is None:
            return
        # Create a black image for instructions
        instruction_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # Add text to the frame
        cv2.putText(instruction_frame, text, (50, frame_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "Press any key to continue...",
                   (50, frame_height // 2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(self.window_name, instruction_frame)
        cv2.waitKey(1)
        
    def show_progress(self, phase: str, progress: float, frame_height: int = 480, frame_width: int = 640):
        """Display calibration progress."""
        if cv2 is None:
            return
        progress_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Add phase text
        cv2.putText(progress_frame, f"Calibrating {phase} eyes", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw progress bar
        bar_width = 400
        bar_height = 30
        bar_x = (frame_width - bar_width) // 2
        bar_y = frame_height // 2
        
        # Background of progress bar
        cv2.rectangle(progress_frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progress fill
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            cv2.rectangle(progress_frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         (0, 255, 0), -1)
        
        # Progress percentage
        cv2.putText(progress_frame, f"{int(progress * 100)}%",
                   (frame_width // 2 - 30, frame_height // 2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, progress_frame)
        cv2.waitKey(1)
        
    def show_eye_state(self, is_open: bool, frame_height: int = 480, frame_width: int = 640):
        """Display current eye state."""
        if cv2 is None:
            return
        state_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Add eye state text
        state_text = "Eyes OPEN" if is_open else "Eyes CLOSED"
        color = (0, 255, 0) if is_open else (0, 0, 255)
        cv2.putText(state_frame, state_text, (50, frame_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        cv2.imshow(self.window_name, state_frame)
        cv2.waitKey(1)
        
    def close(self):
        """Close the visual feedback window."""
        if cv2 is not None:
            cv2.destroyWindow(self.window_name)


class CalibrationQualityChecker:
    """
    Class for checking the quality of eye calibration.
    """
    def __init__(self):
        self.min_valid_range = 0.01  # Minimum range for valid calibration
        self.min_sample_count = 10  # Minimum number of samples for valid calibration
        
    def calculate_metrics(self, calibrator) -> Dict[str, float]:
        """
        Calculate quality metrics for the calibration.
        
        Args:
            calibrator: EyeCalibrator instance
            
        Returns:
            Dictionary with quality metrics
        """
        # Calculate range of values
        left_range = calibrator.EYE_OPEN_CALIBRATED_MAX_LEFT - calibrator.EYE_OPEN_CALIBRATED_MIN_LEFT
        right_range = calibrator.EYE_OPEN_CALIBRATED_MAX_RIGHT - calibrator.EYE_OPEN_CALIBRATED_MIN_RIGHT
        
        # Calculate sample counts
        open_sample_count = len(calibrator.open_distances_left)
        closed_sample_count = len(calibrator.min_distances_left)
        min_sample_count = min(open_sample_count, closed_sample_count)
        
        # Calculate stability (inverse of standard deviation)
        left_open_std = np.std(calibrator.open_distances_left) if calibrator.open_distances_left else 0
        left_closed_std = np.std(calibrator.min_distances_left) if calibrator.min_distances_left else 0
        right_open_std = np.std(calibrator.open_distances_right) if calibrator.open_distances_right else 0
        right_closed_std = np.std(calibrator.min_distances_right) if calibrator.min_distances_right else 0
        
        # Average stability (lower std = higher stability)
        avg_stability = 1.0 / (1.0 + (left_open_std + left_closed_std + right_open_std + right_closed_std) / 4.0)
        
        return {
            'left_range': float(left_range),
            'right_range': float(right_range),
            'sample_count': float(min_sample_count),
            'stability': float(avg_stability),
            'left_open_std': float(left_open_std),
            'left_closed_std': float(left_closed_std),
            'right_open_std': float(right_open_std),
            'right_closed_std': float(right_closed_std)
        }
        
    def is_calibration_valid(self, metrics: Dict[str, float]) -> bool:
        """
        Check if calibration is valid based on quality metrics.
        
        Args:
            metrics: Dictionary with quality metrics
            
        Returns:
            True if calibration is valid, False otherwise
        """
        # Check if range is sufficient
        if metrics['left_range'] < self.min_valid_range or metrics['right_range'] < self.min_valid_range:
            return False
            
        # Check if we have enough samples
        if metrics['sample_count'] < self.min_sample_count:
            return False
            
        # Check if stability is sufficient (at least 0.5 on our scale)
        if metrics['stability'] < 0.5:
            return False
            
        return True
        
    def suggest_recalibration(self, metrics: Dict[str, float]) -> str:
        """
        Provide suggestions for recalibration based on quality metrics.
        
        Args:
            metrics: Dictionary with quality metrics
            
        Returns:
            String with suggestion for improvement
        """
        suggestions = []
        
        if metrics['left_range'] < self.min_valid_range or metrics['right_range'] < self.min_valid_range:
            suggestions.append("Range of eye movement is too small. Try opening and closing eyes more fully.")
            
        if metrics['sample_count'] < self.min_sample_count:
            suggestions.append("Not enough samples collected. Keep eyes steady during calibration.")
            
        if metrics['stability'] < 0.5:
            suggestions.append("Eye measurements are unstable. Keep head still during calibration.")
            
        if not suggestions:
            return "Calibration quality is good."
            
        return " ".join(suggestions)


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
        self.eye_state_history = []  # History of eye states for stability analysis
        self.visual_feedback = None
        self.quality_checker = CalibrationQualityChecker()
        
        # Initialize calibration attributes with default values
        self.EYE_OPEN_CALIBRATED_MAX_LEFT = 0.038
        self.EYE_OPEN_CALIBRATED_MIN_LEFT = 0.012
        self.EYE_OPEN_CALIBRATED_MAX_RIGHT = 0.038
        self.EYE_OPEN_CALIBRATED_MIN_RIGHT = 0.012
        
        # Initialize eye direction calibration attributes with default values
        self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X = -0.5
        self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X = 0.5
        self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y = -0.5
        self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y = 0.5
        self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X = -0.5
        self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X = 0.5
        self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y = -0.5
        self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y = 0.5
        
        self._load_calibration_data()
        
    def start_calibration(self):
        """Start the calibration process"""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.open_distances_left.clear()
        self.open_distances_right.clear()
        self.min_distances_left.clear()
        self.min_distances_right.clear()
        self.eye_state_history.clear()
        self.is_open_phase = True
        
        # Initialize visual feedback
        # Initialize visual feedback
        if self.visual_feedback is None:
            self.visual_feedback = VisualFeedback()
            
        logger.info("Eye calibration process started. Keep your eyes open.")
        if self.visual_feedback:
            self.visual_feedback.show_instruction("Keep your eyes OPEN for calibration")
        
    def detect_eye_state(self, landmarks: list) -> bool:
        """
        Automatically detect if eyes are open or closed with hysteresis.
        
        Args:
            landmarks: List of face landmarks from MediaPipe.
            
        Returns:
            True if eyes are open, False if closed
        """
        # Calculate eye openness for both eyes
        left_eye_openness = _calculate_eye_openness(landmarks, (386, 374))  # Right eye in MediaPipe
        right_eye_openness = _calculate_eye_openness(landmarks, (159, 145))  # Left eye in MediaPipe
        
        # Get current thresholds
        thresholds = self.get_thresholds()
        
        # Calculate normalized openness (0.0 = closed, 1.0 = open)
        left_normalized = (left_eye_openness - thresholds['left_min']) / (thresholds['left_max'] - thresholds['left_min'])
        right_normalized = (right_eye_openness - thresholds['right_min']) / (thresholds['right_max'] - thresholds['right_min'])
        
        # Average both eyes
        avg_openness = (left_normalized + right_normalized) / 2.0
        
        # Add hysteresis to prevent rapid state changes during blinking
        if hasattr(self, '_last_eye_state'):
            if self._last_eye_state and avg_openness < 0.3:  # Transition to closed
                eye_state = False
            elif not self._last_eye_state and avg_openness > 0.7:  # Transition to open
                eye_state = True
            else:
                eye_state = self._last_eye_state
        else:
            eye_state = avg_openness > 0.5
        
        self._last_eye_state = eye_state
        return eye_state
        
    def is_eye_state_stable(self, current_state: bool, window_size: int = 10) -> bool:
        """
        Check if the eye state is stable over time.
        
        Args:
            current_state: Current eye state (True = open, False = closed)
            window_size: Number of previous states to consider
            
        Returns:
            True if state is stable, False otherwise
        """
        # Add current state to history
        self.eye_state_history.append(current_state)
        
        # Keep only the last window_size states
        if len(self.eye_state_history) > window_size:
            self.eye_state_history = self.eye_state_history[-window_size:]
            
        # Need at least 5 samples to determine stability
        if len(self.eye_state_history) < 5:
            return False
            
        # Check if all recent states are the same
        return all(state == current_state for state in self.eye_state_history[-5:])
        
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
        
        # Automatically detect eye state
        current_eye_state = self.detect_eye_state(landmarks)
        
        # Show current eye state in visual feedback
        if self.visual_feedback:
            self.visual_feedback.show_eye_state(current_eye_state)
        
        # Check if eye state is stable
        if self.is_eye_state_stable(current_eye_state):
            # Determine distances
            # MediaPipe right eye -> model left eye
            dist_left = _calculate_eye_openness(landmarks, (386, 374))
            # MediaPipe left eye -> model right eye
            dist_right = _calculate_eye_openness(landmarks, (159, 145))

            # Add data based on current phase and eye state
            if self.is_open_phase and current_eye_state:
                # Collecting open eye data
                self.open_distances_left.append(dist_left)
                self.open_distances_right.append(dist_right)
            elif not self.is_open_phase and not current_eye_state:
                # Collecting closed eye data
                self.min_distances_left.append(dist_left)
                self.min_distances_right.append(dist_right)
                
                # Show progress for closed eye phase
                if self.visual_feedback:
                    progress = min(elapsed / self.calibration_duration, 1.0)
                    self.visual_feedback.show_progress("closed", progress)

            # Show progress for open eye phase
            if self.is_open_phase and self.visual_feedback:
                progress = min(elapsed / self.calibration_duration, 1.0)
                self.visual_feedback.show_progress("open", progress)

        # Check if calibration is complete by timer
        if (elapsed >= self.calibration_duration and
                self.calibration_start_time is not None):
            if self.is_open_phase:
                # Switch to closed eyes phase
                self.is_open_phase = False
                self.calibration_start_time = current_time
                self.eye_state_history.clear()  # Clear history for new phase
                logger.info("Now close your eyes for minimum calibration.")
                if self.visual_feedback:
                    self.visual_feedback.show_instruction("Now CLOSE your eyes")
                return False
            else:
                # Complete calibration
                self._finalize_calibration()
                return True

        # Update eye direction calibration data
        self.update_eye_direction_calibration(landmarks, self.is_calibrating)

        return False
        
    def calibrate_eye_direction(self, landmarks: list) -> Dict[str, float]:
        """
        Calibrate eye direction tracking using current landmarks.
        
        Args:
            landmarks: List of face landmarks from MediaPipe.
            
        Returns:
            Dict: Dictionary with calibrated eye direction values.
        """
        # MediaPipe Face Mesh landmark indices for eye tracking
        # Left eye (from viewer's perspective, right eye in MediaPipe)
        left_pupil = 468
        left_eye_left_corner = 33
        left_eye_right_corner = 133
        left_eye_top = 159
        left_eye_bottom = 145
        
        # Right eye (from viewer's perspective, left eye in MediaPipe)
        right_pupil = 473
        right_eye_left_corner = 362
        right_eye_right_corner = 263
        right_eye_top = 386
        right_eye_bottom = 374
        
        # Check if all required landmarks are available
        required_landmarks = [
            left_pupil, left_eye_left_corner, left_eye_right_corner, left_eye_top, left_eye_bottom,
            right_pupil, right_eye_left_corner, right_eye_right_corner, right_eye_top, right_eye_bottom
        ]
        
        # Verify all landmarks exist
        if not all(0 <= idx < len(landmarks) for idx in required_landmarks):
            # Return default values if landmarks are missing
            return {
                "left_x": 0.0,
                "left_y": 0.0,
                "right_x": 0.0,
                "right_y": 0.0
            }
        
        try:
            # Calculate eye centers
            left_eye_center_x = (landmarks[left_eye_left_corner].x + landmarks[left_eye_right_corner].x) / 2
            left_eye_center_y = (landmarks[left_eye_left_corner].y + landmarks[left_eye_right_corner].y) / 2
            
            right_eye_center_x = (landmarks[right_eye_left_corner].x + landmarks[right_eye_right_corner].x) / 2
            right_eye_center_y = (landmarks[right_eye_left_corner].y + landmarks[right_eye_right_corner].y) / 2
            
            # Calculate eye dimensions
            left_eye_width = abs(landmarks[left_eye_right_corner].x - landmarks[left_eye_left_corner].x)
            left_eye_height = abs(landmarks[left_eye_top].y - landmarks[left_eye_bottom].y)
            
            right_eye_width = abs(landmarks[right_eye_right_corner].x - landmarks[right_eye_left_corner].x)
            right_eye_height = abs(landmarks[right_eye_top].y - landmarks[right_eye_bottom].y)
            
            # Calculate raw gaze directions
            left_gaze_x = (landmarks[left_pupil].x - left_eye_center_x) / left_eye_width if left_eye_width > 0 else 0
            left_gaze_y = (landmarks[left_pupil].y - left_eye_center_y) / left_eye_height if left_eye_height > 0 else 0
            
            right_gaze_x = (landmarks[right_pupil].x - right_eye_center_x) / right_eye_width if right_eye_width > 0 else 0
            right_gaze_y = (landmarks[right_pupil].y - right_eye_center_y) / right_eye_height if right_eye_height > 0 else 0
            
            # Apply calibration normalization
            left_gaze_x_calibrated = self._normalize_eye_direction(
                left_gaze_x,
                self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X,
                self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X
            )
            left_gaze_y_calibrated = self._normalize_eye_direction(
                left_gaze_y,
                self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y,
                self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y
            )
            right_gaze_x_calibrated = self._normalize_eye_direction(
                right_gaze_x,
                self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X,
                self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X
            )
            right_gaze_y_calibrated = self._normalize_eye_direction(
                right_gaze_y,
                self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y,
                self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y
            )
            
            return {
                "left_x": left_gaze_x_calibrated,
                "left_y": left_gaze_y_calibrated,
                "right_x": right_gaze_x_calibrated,
                "right_y": right_gaze_y_calibrated
            }
        except Exception as e:
            logger.warning(f"Error calibrating eye direction: {e}")
            # Return default values on error
            return {
                "left_x": 0.0,
                "left_y": 0.0,
                "right_x": 0.0,
                "right_y": 0.0
            }
    
    def update_eye_direction_calibration(self, landmarks: list, is_calibrating: bool = False):
        """
        Update eye direction calibration data based on current landmarks.
        
        Args:
            landmarks: List of face landmarks from MediaPipe.
            is_calibrating: Whether we're currently calibrating.
        """
        # MediaPipe Face Mesh landmark indices for eye tracking
        # Left eye (from viewer's perspective, right eye in MediaPipe)
        left_pupil = 468
        left_eye_left_corner = 33
        left_eye_right_corner = 133
        left_eye_top = 159
        left_eye_bottom = 145
        
        # Right eye (from viewer's perspective, left eye in MediaPipe)
        right_pupil = 473
        right_eye_left_corner = 362
        right_eye_right_corner = 263
        right_eye_top = 386
        right_eye_bottom = 374
        
        # Check if all required landmarks are available
        required_landmarks = [
            left_pupil, left_eye_left_corner, left_eye_right_corner, left_eye_top, left_eye_bottom,
            right_pupil, right_eye_left_corner, right_eye_right_corner, right_eye_top, right_eye_bottom
        ]
        
        # Verify all landmarks exist
        if not all(0 <= idx < len(landmarks) for idx in required_landmarks):
            return
        
        try:
            # Calculate eye centers
            left_eye_center_x = (landmarks[left_eye_left_corner].x + landmarks[left_eye_right_corner].x) / 2
            left_eye_center_y = (landmarks[left_eye_left_corner].y + landmarks[left_eye_right_corner].y) / 2
            
            right_eye_center_x = (landmarks[right_eye_left_corner].x + landmarks[right_eye_right_corner].x) / 2
            right_eye_center_y = (landmarks[right_eye_left_corner].y + landmarks[right_eye_right_corner].y) / 2
            
            # Calculate eye dimensions
            left_eye_width = abs(landmarks[left_eye_right_corner].x - landmarks[left_eye_left_corner].x)
            left_eye_height = abs(landmarks[left_eye_top].y - landmarks[left_eye_bottom].y)
            
            right_eye_width = abs(landmarks[right_eye_right_corner].x - landmarks[right_eye_left_corner].x)
            right_eye_height = abs(landmarks[right_eye_top].y - landmarks[right_eye_bottom].y)
            
            # Calculate raw gaze directions
            left_gaze_x = (landmarks[left_pupil].x - left_eye_center_x) / left_eye_width if left_eye_width > 0 else 0
            left_gaze_y = (landmarks[left_pupil].y - left_eye_center_y) / left_eye_height if left_eye_height > 0 else 0
            
            right_gaze_x = (landmarks[right_pupil].x - right_eye_center_x) / right_eye_width if right_eye_width > 0 else 0
            right_gaze_y = (landmarks[right_pupil].y - right_eye_center_y) / right_eye_height if right_eye_height > 0 else 0
            
            # Update calibration data if calibrating
            if is_calibrating:
                # Update min/max values for left eye X direction
                if left_gaze_x < self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X:
                    self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X = left_gaze_x
                if left_gaze_x > self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X:
                    self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X = left_gaze_x
                
                # Update min/max values for left eye Y direction
                if left_gaze_y < self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y:
                    self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y = left_gaze_y
                if left_gaze_y > self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y:
                    self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y = left_gaze_y
                
                # Update min/max values for right eye X direction
                if right_gaze_x < self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X:
                    self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X = right_gaze_x
                if right_gaze_x > self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X:
                    self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X = right_gaze_x
                
                # Update min/max values for right eye Y direction
                if right_gaze_y < self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y:
                    self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y = right_gaze_y
                if right_gaze_y > self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y:
                    self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y = right_gaze_y
        except Exception as e:
            logger.warning(f"Error updating eye direction calibration: {e}")
    
    def _normalize_eye_direction(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize eye direction value to [-1, 1] range using calibration data.
        
        Args:
            value: Raw eye direction value
            min_val: Minimum calibrated value
            max_val: Maximum calibrated value
            
        Returns:
            float: Normalized value in [-1, 1] range
        """
        if max_val - min_val == 0:
            return 0.0
        
        # Normalize to [0, 1] range first
        normalized = (value - min_val) / (max_val - min_val)
        # Convert to [-1, 1] range
        return max(-1.0, min(1.0, normalized * 2 - 1))
        
    def calculate_calibration_quality(self) -> Dict[str, float]:
        """
        Calculate quality metrics for the current calibration.
        
        Returns:
            Dictionary with quality metrics
        """
        return self.quality_checker.calculate_metrics(self)
        
    def needs_recalibration(self) -> bool:
        """
        Check if calibration needs to be redone based on quality metrics.
        
        Returns:
            True if recalibration is needed, False otherwise
        """
        if not hasattr(self, 'EYE_OPEN_CALIBRATED_MAX_LEFT'):
            return True
            
        metrics = self.calculate_calibration_quality()
        return not self.quality_checker.is_calibration_valid(metrics)

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
            
        # Calculate and log quality metrics
        metrics = self.calculate_calibration_quality()
        logger.info("Calibration quality metrics:")
        logger.info("  Left range: %.4f, Right range: %.4f" % (metrics['left_range'], metrics['right_range']))
        logger.info("  Sample count: %d, Stability: %.2f" % (int(metrics['sample_count']), metrics['stability']))
        
        # Check if calibration is valid
        if not self.quality_checker.is_calibration_valid(metrics):
            suggestions = self.quality_checker.suggest_recalibration(metrics)
            logger.warning("Calibration quality is low: %s", suggestions)
            
        # Save calibration data
        self._save_calibration_data()
        
        # Close visual feedback
        if self.visual_feedback:
            self.visual_feedback.close()
            self.visual_feedback = None

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
                    
                    # Load eye direction calibration data if available
                    self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X = data.get("left_dir_min_x", -0.5)
                    self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X = data.get("left_dir_max_x", 0.5)
                    self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y = data.get("left_dir_min_y", -0.5)
                    self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y = data.get("left_dir_max_y", 0.5)
                    self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X = data.get("right_dir_min_x", -0.5)
                    self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X = data.get("right_dir_max_x", 0.5)
                    self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y = data.get("right_dir_min_y", -0.5)
                    self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y = data.get("right_dir_max_y", 0.5)
                    logger.info("Loaded previous calibration data from file")
        except Exception as e:
            logger.warning(f"Could not load calibration data: {e}")
            # Ensure default values are set even if loading fails
            self.EYE_OPEN_CALIBRATED_MAX_LEFT = 0.038
            self.EYE_OPEN_CALIBRATED_MIN_LEFT = 0.012
            self.EYE_OPEN_CALIBRATED_MAX_RIGHT = 0.038
            self.EYE_OPEN_CALIBRATED_MIN_RIGHT = 0.012
            
            # Default values for eye direction calibration
            self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X = -0.5
            self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X = 0.5
            self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y = -0.5
            self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y = 0.5
            self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X = -0.5
            self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X = 0.5
            self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y = -0.5
            self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y = 0.5

    def _save_calibration_data(self):
        """Save calibration data to file."""
        try:
            data = {
                "left_max": float(self.EYE_OPEN_CALIBRATED_MAX_LEFT),
                "left_min": float(self.EYE_OPEN_CALIBRATED_MIN_LEFT),
                "right_max": float(self.EYE_OPEN_CALIBRATED_MAX_RIGHT),
                "right_min": float(self.EYE_OPEN_CALIBRATED_MIN_RIGHT),
                "left_dir_min_x": float(self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X),
                "left_dir_max_x": float(self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X),
                "left_dir_min_y": float(self.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y),
                "left_dir_max_y": float(self.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y),
                "right_dir_min_x": float(self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X),
                "right_dir_max_x": float(self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X),
                "right_dir_min_y": float(self.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y),
                "right_dir_max_y": float(self.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y)
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
        Run the complete eye calibration process with manual confirmation.
        The camera is activated only during the calibration process.
        
        Returns:
            bool: True if calibration completed successfully, False otherwise.
        """
        try:
            # Initial prompt
            logger.info("Starting manual eye calibration...")
            
            # Start camera for calibration
            from src.camera import generate_frames
            frame_generator = generate_frames(device_id=0, width=640, height=480, fps=30)
            
            # Initialize visual feedback
            if self.calibrator.visual_feedback is None:
                self.calibrator.visual_feedback = VisualFeedback()
            
            # Show instruction for open eyes
            self.calibrator.visual_feedback.show_instruction("Keep your eyes OPEN and press any key to start calibration")
            if cv2 is not None:
                cv2.waitKey(0)  # Wait for user input
            
            # Start calibration for open eyes
            self.calibrator.is_calibrating = True
            self.calibrator.is_open_phase = True
            self.calibrator.open_distances_left.clear()
            self.calibrator.open_distances_right.clear()
            self.calibrator.min_distances_left.clear()
            self.calibrator.min_distances_right.clear()
            self.calibrator.calibration_start_time = time.time()
            
            # Collect data for open eyes for 3 seconds
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    success, frame = next(frame_generator, (False, None))
                    if success and frame is not None:
                        # Extract landmarks from frame
                        if cv2 is not None:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            # Fallback if cv2 is not available
                            rgb_frame = frame
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        detection_result = self.tracker.detector.detect(image)
                        if detection_result.face_landmarks:
                            landmarks = detection_result.face_landmarks[0]
                            # Determine distances
                            dist_left = _calculate_eye_openness(landmarks, (386, 374))
                            dist_right = _calculate_eye_openness(landmarks, (159, 145))
                            # Add data
                            self.calibrator.open_distances_left.append(dist_left)
                            self.calibrator.open_distances_right.append(dist_right)
                            
                            # Show progress
                            elapsed = time.time() - start_time
                            progress = min(elapsed / 3.0, 1.0)
                            if self.calibrator.visual_feedback:
                                self.calibrator.visual_feedback.show_progress("open", progress)
                    await asyncio.sleep(1/30)  # Simulate FPS
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error during open eye calibration: {e}")
                    break
            
            # Show instruction for closed eyes
            self.calibrator.visual_feedback.show_instruction("Keep your eyes CLOSED and press any key to continue calibration")
            if cv2 is not None:
                cv2.waitKey(0)  # Wait for user input
            
            # Collect data for closed eyes for 3 seconds
            self.calibrator.is_open_phase = False
            self.calibrator.open_distances_left.clear()
            self.calibrator.open_distances_right.clear()
            self.calibrator.min_distances_left.clear()
            self.calibrator.min_distances_right.clear()
            self.calibrator.calibration_start_time = time.time()
            
            start_time = time.time()
            while time.time() - start_time < 3.0:
                try:
                    success, frame = next(frame_generator, (False, None))
                    if success and frame is not None:
                        # Extract landmarks from frame
                        if cv2 is not None:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            # Fallback if cv2 is not available
                            rgb_frame = frame
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        detection_result = self.tracker.detector.detect(image)
                        if detection_result.face_landmarks:
                            landmarks = detection_result.face_landmarks[0]
                            # Determine distances
                            dist_left = _calculate_eye_openness(landmarks, (386, 374))
                            dist_right = _calculate_eye_openness(landmarks, (159, 145))
                            # Add data
                            self.calibrator.min_distances_left.append(dist_left)
                            self.calibrator.min_distances_right.append(dist_right)
                            
                            # Show progress
                            elapsed = time.time() - start_time
                            progress = min(elapsed / 3.0, 1.0)
                            if self.calibrator.visual_feedback:
                                self.calibrator.visual_feedback.show_progress("closed", progress)
                    await asyncio.sleep(1/30)  # Simulate FPS
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error during closed eye calibration: {e}")
                    break
            
            # Finalize calibration
            self.calibrator._finalize_calibration()
            self.calibrator.is_calibrating = False
            
            logger.info("Eye calibration completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration process: {e}")
            # Close visual feedback if error occurs
            if self.calibrator.visual_feedback:
                self.calibrator.visual_feedback.close()
                self.calibrator.visual_feedback = None
            # Reset calibration state
            self.calibrator.is_calibrating = False
            return False
