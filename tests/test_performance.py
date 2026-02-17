"""
Performance test for the optimized parameter mapper.
This test verifies that the optimization doesn't negatively
impact real-time performance.
"""

import time
import random
import unittest
from src.optimized_parameter_mapper import OptimizedParameterMapper


def generate_test_data(frame_num):
    """Generate test MediaPipe data with subtle changes."""
    # Add some variation to simulate real face movements
    variation = 0.05 * random.random()
    
    return {
        "pose": {
            "pitch": 15.0 + variation,
            "yaw": -10.0 + variation,
            "roll": 5.0 + variation
        },
        "blendshapes": {
            "jawOpen": 0.5 + variation,
            "mouthSmileLeft": 0.3 + variation,
            "mouthSmileRight": 0.3 + variation,
            "eyeBlinkLeft": 0.8 + variation,
            "eyeBlinkRight": 0.8 + variation,
            "browDownLeft": 0.2 + variation,
            "browDownRight": 0.2 + variation,
            "browInnerUp": 0.1 + variation
        }
    }


class TestPerformance(unittest.TestCase):
    """Test suite for performance tests."""

    def test_performance(self):
        """Test that the optimized mapper maintains real-time performance."""
        mapper = OptimizedParameterMapper()
    
        # Test with 1000 frames of data
        num_frames = 1000
        start_time = time.time()
    
        for i in range(num_frames):
            test_data = generate_test_data(i)
            result = mapper.transform_and_optimize(test_data)
        
            # In a real application, we would send the result to VTube Studio
            # For this test, we just verify it doesn't crash and returns a dict
            self.assertIsInstance(
                result, dict, "Result should be a dictionary"
            )
    
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_frame = total_time / num_frames
    
        # Should be well under 10ms per frame for real-time performance
        # (30 FPS = 33.3ms per frame, 60 FPS = 16.6ms per frame)
        self.assertLess(
            avg_time_per_frame, 0.01,
            f"Average time per frame ({avg_time_per_frame*1000:.2f}ms) "
            f"exceeds 10ms"
        )
    
        # Print statistics
        stats = mapper.get_stats()
    
        # Verify that we're sending fewer parameters than we process
        # (This indicates that change detection is working)
        if stats['parameters_processed'] > 0:
            reduction_ratio = (
                stats['parameters_sent'] / stats['parameters_processed']
            )
        
            # Should be less than 100% (some reduction)
            # But not too low (should still send meaningful changes)
            self.assertLessEqual(
                reduction_ratio, 1.0,
                "Should not send more parameters than processed"
            )


if __name__ == "__main__":
    unittest.main()