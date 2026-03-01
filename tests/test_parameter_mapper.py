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

    def test_mouth_x_calculation(self):
        """Test MouthX calculation from mouthLeft and mouthRight blendshapes."""
        # Test case 1: Mouth centered (both left and right values are equal)
        mediapipe_data_center = {
            "blendshapes": {
                "mouthLeft": 0.3,
                "mouthRight": 0.3
            }
        }
        result_center = transform_mediapipe_to_vtubestudio(mediapipe_data_center)
        self.assertIn("MouthX", result_center)
        # When left and right are equal, MouthX should be 0.5 (center)
        self.assertAlmostEqual(result_center["MouthX"], 0.5, places=3)

        # Test case 2: Mouth moved to the right
        mediapipe_data_right = {
            "blendshapes": {
                "mouthLeft": 0.1,
                "mouthRight": 0.7
            }
        }
        result_right = transform_mediapipe_to_vtubestudio(mediapipe_data_right)
        self.assertIn("MouthX", result_right)
        # MouthX should be > 0.5 when mouth is moved to the right
        self.assertGreater(result_right["MouthX"], 0.5)

        # Test case 3: Mouth moved to the left
        mediapipe_data_left = {
            "blendshapes": {
                "mouthLeft": 0.7,
                "mouthRight": 0.1
            }
        }
        result_left = transform_mediapipe_to_vtubestudio(mediapipe_data_left)
        self.assertIn("MouthX", result_left)
        # MouthX should be < 0.5 when mouth is moved to the left
        self.assertLess(result_left["MouthX"], 0.5)

        # Test case 4: Missing blendshapes (should not crash)
        mediapipe_data_missing = {
            "blendshapes": {
                "jawOpen": 0.5
            }
        }
        result_missing = transform_mediapipe_to_vtubestudio(mediapipe_data_missing)
        # MouthX should not be present when blendshapes are missing
        self.assertNotIn("MouthX", result_missing)

    def test_tongue_out_mapping(self):
        """Test TongueOut parameter mapping from MediaPipe blendshapes."""
        # Test case 1: Tongue fully out
        mediapipe_data_full = {
            "blendshapes": {
                "tongueOut": 1.0
            }
        }
        result_full = transform_mediapipe_to_vtubestudio(mediapipe_data_full)
        self.assertIn("TongueOut", result_full)
        self.assertAlmostEqual(result_full["TongueOut"], 1.0)

        # Test case 2: Tongue partially out
        mediapipe_data_partial = {
            "blendshapes": {
                "tongueOut": 0.5
            }
        }
        result_partial = transform_mediapipe_to_vtubestudio(mediapipe_data_partial)
        self.assertIn("TongueOut", result_partial)
        self.assertAlmostEqual(result_partial["TongueOut"], 0.5)

        # Test case 3: Tongue retracted
        mediapipe_data_retracted = {
            "blendshapes": {
                "tongueOut": 0.0
            }
        }
        result_retracted = transform_mediapipe_to_vtubestudio(mediapipe_data_retracted)
        self.assertIn("TongueOut", result_retracted)
        self.assertAlmostEqual(result_retracted["TongueOut"], 0.0)

        # Test case 4: Missing tongue blendshape
        mediapipe_data_missing = {
            "blendshapes": {
                "jawOpen": 0.5
            }
        }
        result_missing = transform_mediapipe_to_vtubestudio(mediapipe_data_missing)
        # TongueOut should not be present when blendshape is missing
        self.assertNotIn("TongueOut", result_missing)


if __name__ == "__main__":
    unittest.main()
