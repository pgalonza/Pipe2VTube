"""
Test for eye direction tracking functionality.
"""
import unittest
from unittest.mock import Mock
from src.parameter_mapper import transform_mediapipe_to_vtubestudio
from src.eye_calibrator import calibrator


class TestEyeDirectionTracking(unittest.TestCase):
    """Test cases for eye direction tracking."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset calibrator to default values
        calibrator.EYE_OPEN_CALIBRATED_MAX_LEFT = 0.038
        calibrator.EYE_OPEN_CALIBRATED_MIN_LEFT = 0.012
        calibrator.EYE_OPEN_CALIBRATED_MAX_RIGHT = 0.038
        calibrator.EYE_OPEN_CALIBRATED_MIN_RIGHT = 0.012
        
        # Set default values for eye direction calibration
        calibrator.EYE_DIRECTION_CALIBRATED_MIN_LEFT_X = -0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MAX_LEFT_X = 0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MIN_LEFT_Y = -0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MAX_LEFT_Y = 0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_X = -0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_X = 0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MIN_RIGHT_Y = -0.5
        calibrator.EYE_DIRECTION_CALIBRATED_MAX_RIGHT_Y = 0.5

    def test_eye_direction_parameters_present(self):
        """Test that eye direction parameters are present in output."""
        # Create mock landmarks with basic structure
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        
        # Create list of 478 landmarks (FaceMesh full set)
        mock_landmarks = [mock_landmark] * 478
        
        # Test data with landmarks
        mediapipe_data = {
            "landmarks": mock_landmarks,
            "blendshapes": {},
            "pose": None
        }
        
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Check that eye direction parameters are present
        self.assertIn("EyeLeftX", result)
        self.assertIn("EyeLeftY", result)
        self.assertIn("EyeRightX", result)
        self.assertIn("EyeRightY", result)
        
        # Check that values are in valid range [-1, 1]
        self.assertGreaterEqual(result["EyeLeftX"], -1.0)
        self.assertLessEqual(result["EyeLeftX"], 1.0)
        self.assertGreaterEqual(result["EyeLeftY"], -1.0)
        self.assertLessEqual(result["EyeLeftY"], 1.0)
        self.assertGreaterEqual(result["EyeRightX"], -1.0)
        self.assertLessEqual(result["EyeRightX"], 1.0)
        self.assertGreaterEqual(result["EyeRightY"], -1.0)
        self.assertLessEqual(result["EyeRightY"], 1.0)

    def test_eye_direction_with_extreme_values(self):
        """Test eye direction calculation with extreme landmark values."""
        # Create landmarks with extreme values to test normalization
        mock_landmarks = []
        for i in range(478):
            landmark = Mock()
            # Left half 0.0, right half 1.0
            landmark.x = 0.0 if i < 239 else 1.0
            # Alternate 0.0 and 1.0
            landmark.y = 0.0 if i % 2 == 0 else 1.0
            landmark.z = 0.0
            mock_landmarks.append(landmark)
        
        mediapipe_data = {
            "landmarks": mock_landmarks,
            "blendshapes": {},
            "pose": None
        }
        
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Check that eye direction parameters are present
        self.assertIn("EyeLeftX", result)
        self.assertIn("EyeLeftY", result)
        self.assertIn("EyeRightX", result)
        self.assertIn("EyeRightY", result)
        
        # Check that values are in valid range [-1, 1]
        self.assertGreaterEqual(result["EyeLeftX"], -1.0)
        self.assertLessEqual(result["EyeLeftX"], 1.0)
        self.assertGreaterEqual(result["EyeLeftY"], -1.0)
        self.assertLessEqual(result["EyeLeftY"], 1.0)
        self.assertGreaterEqual(result["EyeRightX"], -1.0)
        self.assertLessEqual(result["EyeRightX"], 1.0)
        self.assertGreaterEqual(result["EyeRightY"], -1.0)
        self.assertLessEqual(result["EyeRightY"], 1.0)

    def test_eye_direction_with_insufficient_landmarks(self):
        """Test eye direction with insufficient landmarks."""
        # Create only 10 landmarks (not enough for eye tracking)
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmarks = [mock_landmark] * 10
        
        mediapipe_data = {
            "landmarks": mock_landmarks,
            "blendshapes": {},
            "pose": None
        }
        
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Should still have eye direction parameters but with default values
        self.assertIn("EyeLeftX", result)
        self.assertIn("EyeLeftY", result)
        self.assertIn("EyeRightX", result)
        self.assertIn("EyeRightY", result)


if __name__ == '__main__':
    unittest.main()