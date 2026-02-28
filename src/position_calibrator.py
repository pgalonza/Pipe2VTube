"""
Module for position calibration logic.
Handles calibration of face position for Live2D control.
"""
from typing import Dict, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)


class PositionCalibrator:
    """
    Class for automatic calibration of face position.
    Makes position values relative to an initial neutral position.
    """
    
    def __init__(self):
        # Position calibration state
        self._calibration_position: Optional[Dict[str, float]] = None
        self._is_calibrated = False
        self._last_face_detected_time = 0.0
        # Reset calibration if no face detected for 1 second
        self._calibration_timeout = 1.0
        
        # Load saved calibration data if available
        self._load_calibration_data()
        
    def calibrate_position(self, position: Dict[str, float]) -> None:
        """
        Calibrate the initial position as the neutral position.
        
        Args:
            position: Dictionary with position_x, position_y,
                position_z values.
        """
        self._calibration_position = {
            "position_x": float(position["position_x"]),
            "position_y": float(position["position_y"]),
            "position_z": float(position["position_z"])
        }
        self._is_calibrated = True
        
        # Save calibration data
        self._save_calibration_data()
        
        logger.debug(
            "Position calibrated: x=%.4f, y=%.4f, z=%.4f" %
            (position["position_x"], position["position_y"],
             position["position_z"])
        )
    
    def reset_calibration(self) -> None:
        """
        Reset the position calibration.
        """
        self._calibration_position = None
        self._is_calibrated = False
        logger.debug("Position calibration reset")
    
    def apply_calibration(
        self,
        position: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply calibration to position data to make it relative to the
        initial position.
        
        Args:
            position: Dictionary with position_x, position_y,
                position_z values.
            
        Returns:
            Dictionary with calibrated position values.
        """
        if not self._is_calibrated or self._calibration_position is None:
            # If not calibrated yet, use current position as calibration
            self.calibrate_position(position)
            return {"position_x": 0.0, "position_y": 0.0, "position_z": 0.0}
        
        # Calculate relative position
        calibrated_position = {}
        for axis in ["position_x", "position_y", "position_z"]:
            calibrated_position[axis] = (
                float(position[axis]) - self._calibration_position[axis])
        
        return calibrated_position
    
    def update_last_face_time(self, current_time: float) -> None:
        """
        Update the last face detected time.
        
        Args:
            current_time: Current time in seconds.
        """
        self._last_face_detected_time = current_time
    
    def check_calibration_timeout(self, current_time: float) -> bool:
        """
        Check if calibration should be reset due to timeout.
        
        Args:
            current_time: Current time in seconds.
            
        Returns:
            True if calibration was reset, False otherwise.
        """
        if not self._is_calibrated:
            return False
            
        time_since_last_face = current_time - self._last_face_detected_time
        if time_since_last_face > self._calibration_timeout:
            self.reset_calibration()
            return True
        return False
    
    def _load_calibration_data(self) -> None:
        """Load calibration data from file if it exists."""
        try:
            if os.path.exists("position_calibration.json"):
                with open("position_calibration.json", "r") as f:
                    data = json.load(f)
                    self._calibration_position = {
                        "position_x": float(data["position_x"]),
                        "position_y": float(data["position_y"]),
                        "position_z": float(data["position_z"])
                    }
                    self._is_calibrated = bool(data["is_calibrated"])
                    logger.info(
                        "Loaded previous position calibration data from file"
                    )
        except Exception as e:
            logger.warning(f"Could not load position calibration data: {e}")
    
    def _save_calibration_data(self) -> None:
        """Save calibration data to file."""
        try:
            if self._calibration_position is not None:
                data = {
                    "position_x": float(
                        self._calibration_position["position_x"]
                    ),
                    "position_y": float(
                        self._calibration_position["position_y"]
                    ),
                    "position_z": float(
                        self._calibration_position["position_z"]
                    ),
                    "is_calibrated": bool(self._is_calibrated)
                }
                with open("position_calibration.json", "w") as f:
                    json.dump(data, f)
                logger.info("Saved position calibration data to file")
        except Exception as e:
            logger.warning(f"Could not save position calibration data: {e}")
    
    @property
    def is_calibrated(self) -> bool:
        """Return whether position is calibrated."""
        return self._is_calibrated
    
    @property
    def calibration_position(self) -> Optional[Dict[str, float]]:
        """Return the calibration position."""
        return self._calibration_position


# Global instance for position calibration
position_calibrator = PositionCalibrator()