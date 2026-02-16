"""
Tests for optimized_parameter_mapper.py
"""

import unittest
import time
from src.optimized_parameter_mapper import OptimizedParameterMapper


class TestOptimizedParameterMapper(unittest.TestCase):
    """
    Test suite for optimized_parameter_mapper.py
    """

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = OptimizedParameterMapper(
            change_threshold=0.01,
            batch_window=0.1,  # 100ms for testing
            cache_ttl=0.5,
            max_batch_size=10
        )

    def test_initialization(self):
        """Test that the mapper initializes correctly."""
        self.assertEqual(self.mapper.change_threshold, 0.01)
        self.assertEqual(self.mapper.batch_window, 0.1)
        self.assertEqual(self.mapper.max_batch_size, 10)
        self.assertEqual(self.mapper.cache_ttl, 0.5)

    def test_significant_change_detection(self):
        """Test significant change detection."""
        # Small change should not be significant
        self.assertFalse(self.mapper._is_significant_change(
            "MouthOpen", 0.5, 0.505))
        
        # Large change should be significant
        self.assertTrue(self.mapper._is_significant_change(
            "MouthOpen", 0.5, 0.6))
        
        # Critical parameters should always be significant
        self.assertTrue(self.mapper._is_significant_change(
            "FaceAngleX", 0.5, 0.505))

    def test_should_send_parameter(self):
        """Test parameter sending logic."""
        # First time should always send
        self.assertTrue(self.mapper._should_send_parameter("MouthOpen", 0.5))
        
        # Small change should not send
        self.assertFalse(
            self.mapper._should_send_parameter("MouthOpen", 0.505))
        
        # Large change should send
        self.assertTrue(
            self.mapper._should_send_parameter("MouthOpen", 0.6))

    def test_cache_operations(self):
        """Test cache operations."""
        # Set a value in cache
        self.mapper._cache_set("test_key", "test_value")
        
        # Get the value back
        self.assertEqual(self.mapper._cache_get("test_key"), "test_value")
        
        # Wait for cache to expire
        time.sleep(0.6)
        
        # Should return None for expired cache
        self.assertIsNone(self.mapper._cache_get("test_key"))

    def test_transform_and_optimize(self):
        """Test the main transformation and optimization function."""
        # Test data
        test_data = {
            "pose": {
                "pitch": 15.0,
                "yaw": -10.0,
                "roll": 5.0
            },
            "blendshapes": {
                "jawOpen": 0.8,
                "mouthSmileLeft": 0.6,
                "mouthSmileRight": 0.7,
                "eyeBlinkLeft": 0.2,
                "eyeBlinkRight": 0.3
            }
        }
        
        # Call the transformation function
        result = self.mapper.transform_and_optimize(test_data)
        
        # Should return a dictionary
        self.assertIsInstance(result, dict)
        
        # Wait for batch window to expire
        time.sleep(0.11)
        
        # Call again - should return a batch since time has passed
        result = self.mapper.transform_and_optimize(test_data)
        # Should have parameters now
        self.assertIsInstance(result, dict)

    def test_batch_flushing(self):
        """Test batch flushing."""
        # Add some parameters to batch
        test_data = {
            "pose": {"pitch": 15.0},
            "blendshapes": {"jawOpen": 0.8}
        }
        
        # Process data to add to batch
        self.mapper.transform_and_optimize(test_data)
        
        # Flush batch
        flushed = self.mapper.flush_batch()
        
        # Should have parameters
        self.assertIsInstance(flushed, dict)
        # Should be empty now since we flushed
        self.assertEqual(self.mapper.flush_batch(), {})


if __name__ == "__main__":
    unittest.main()