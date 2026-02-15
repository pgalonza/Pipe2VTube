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
        
        self.assertIn("FaceAngleY", result)
        self.assertIn("FaceAngleX", result)
        self.assertIn("FaceAngleZ", result)
        
        self.assertAlmostEqual(result["FaceAngleY"], 15.0)
        self.assertAlmostEqual(result["FaceAngleX"], -10.0)
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
        self.assertAlmostEqual(result["MouthSmile"], 0.8)
        
        self.assertIn("MouthOpen", result)
        self.assertAlmostEqual(result["MouthOpen"], 0.3)
        
        # Custom parameter
        self.assertIn("custom_unknown_morph", result)
        self.assertAlmostEqual(result["custom_unknown_morph"], 0.5)

    def test_transform_with_custom_data(self):
        """Test transformation with custom non-standard data."""
        mediapipe_data = {
            "custom_value": 42.0,
            "another_custom": 3.14
        }
        result = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        self.assertIn("custom_custom_value", result)
        self.assertIn("custom_another_custom", result)
        
        self.assertAlmostEqual(result["custom_custom_value"], 42.0)
        self.assertAlmostEqual(result["custom_another_custom"], 3.14)

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


if __name__ == "__main__":
    unittest.main()
