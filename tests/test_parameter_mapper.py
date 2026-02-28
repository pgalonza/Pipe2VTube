"""
Tests for parameter_mapper.py
"""

import unittest
from src.parameter_mapper import transform_mediapipe_to_vtubestudio


class TestParameterMapper(unittest.TestCase):
    """
    Test suite for parameter_mapper.py
    """

    def test_transform_empty_data(self):
        """Test with empty input data."""
        result = transform_mediapipe_to_vtubestudio({})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_transform_only_pose(self):
        """Test transformation with only pose data."""
        mediapipe_data = {
            "pose": {
                "pitch": 15.0,
                "yaw": -10.0,
                "roll": 5.0
            }
        }
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        self.assertIn("FaceAngleX", result)
        self.assertIn("FaceAngleY", result)
        self.assertIn("FaceAngleZ", result)
        
        self.assertAlmostEqual(result["FaceAngleX"], 15.0)
        self.assertAlmostEqual(result["FaceAngleY"], -10.0)
        self.assertAlmostEqual(result["FaceAngleZ"], 5.0)

    def test_transform_only_blendshapes(self):
        """Test transformation with only blendshapes."""
        mediapipe_data = {
            "blendshapes": {
                "happy": 0.8,
                "jawOpen": 0.3,
                "unknown_morph": 0.5
            }
        }
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Standard mapped parameters
        self.assertIn("MouthSmile", result)
        # Happy value 0.8 gets transformed to 0.5 + (0.8 * 0.5) = 0.9
        self.assertAlmostEqual(result["MouthSmile"], 0.9)
        
        self.assertIn("MouthOpen", result)
        self.assertAlmostEqual(result["MouthOpen"], 0.3)
        
        # Custom parameter
        self.assertIn("custom_unknown_morph", result)
        # Custom values get normalized to [0, 1] range, so 0.5 stays 0.5
        self.assertAlmostEqual(result["custom_unknown_morph"], 0.5)

    def test_transform_with_custom_data(self):
        """Test transformation with custom non-standard data."""
        mediapipe_data = {
            "custom_value": 0.8,
            "another_custom": 0.3
        }
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        self.assertIn("custom_custom_value", result)
        self.assertIn("custom_another_custom", result)
        
        # Custom values get normalized to [0, 1] range
        self.assertAlmostEqual(result["custom_custom_value"], 0.8)
        self.assertAlmostEqual(result["custom_another_custom"], 0.3)

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        # Test with non-dict input
        with self.assertRaises(ValueError):
            transform_mediapipe_to_vtubestudio("not a dict")

        # Test with invalid pose data
        result = transform_mediapipe_to_vtubestudio({"pose": "invalid"})
        self.assertIsInstance(result, dict)
        
        # Test with invalid blendshapes data
        result = transform_mediapipe_to_vtubestudio({"blendshapes": "invalid"})
        self.assertIsInstance(result, dict)

    def test_individual_brow_tracking(self):
        """Test individual brow tracking with separate left and right brow calculations."""
        mediapipe_data = {
            "blendshapes": {
                "browDownLeft": 0.6,
                "browOuterUpLeft": 0.3,
                "browDownRight": 0.4,
                "browOuterUpRight": 0.5,
                "browInnerUp": 0.7
            }
        }
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Check that individual brow parameters are present
        self.assertIn("BrowLeftY", result)
        self.assertIn("BrowRightY", result)
        self.assertIn("BrowInnerUp", result)
        
        # Check that the combined Brows parameter is present
        self.assertIn("Brows", result)
        
        # Check value ranges (should be clamped to [0, 1])
        self.assertGreaterEqual(result["BrowLeftY"], 0.0)
        self.assertLessEqual(result["BrowLeftY"], 1.0)
        self.assertGreaterEqual(result["BrowRightY"], 0.0)
        self.assertLessEqual(result["BrowRightY"], 1.0)
        self.assertGreaterEqual(result["BrowInnerUp"], 0.0)
        self.assertLessEqual(result["BrowInnerUp"], 1.0)

    def test_brow_tracking_edge_cases(self):
        """Test brow tracking with edge cases like missing blendshape data."""
        # Test with only left brow data
        mediapipe_data_left = {
            "blendshapes": {
                "browDownLeft": 0.5,
                "browOuterUpLeft": 0.3
            }
        }
        result_left = transform_mediapipe_to_vtubestudio(mediapipe_data_left)
        self.assertIn("BrowLeftY", result_left)
        self.assertNotIn("BrowRightY", result_left)
        
        # Test with only right brow data
        mediapipe_data_right = {
            "blendshapes": {
                "browDownRight": 0.4,
                "browOuterUpRight": 0.6
            }
        }
        result_right = transform_mediapipe_to_vtubestudio(mediapipe_data_right)
        self.assertNotIn("BrowLeftY", result_right)
        self.assertIn("BrowRightY", result_right)
        
        # Test with no brow data
        mediapipe_data_none = {
            "blendshapes": {
                "jawOpen": 0.5
            }
        }
        result_none = transform_mediapipe_to_vtubestudio(mediapipe_data_none)
        self.assertNotIn("BrowLeftY", result_none)
        self.assertNotIn("BrowRightY", result_none)
        self.assertNotIn("BrowInnerUp", result_none)


if __name__ == "__main__":
    unittest.main()
