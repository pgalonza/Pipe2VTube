"""
Tests for eye_calibrator.py
"""

import unittest
from src.eye_calibrator import calibrator


class TestCalibrator(unittest.TestCase):
    """Test suite for eye_calibrator.py"""

    def test_calibrator_attributes(self):
        """Test the calibration attributes."""

        # Test that calibrator has the expected attributes
        self.assertTrue(hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_LEFT'))
        self.assertTrue(hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_LEFT'))
        self.assertTrue(hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MAX_RIGHT'))
        self.assertTrue(hasattr(calibrator, 'EYE_OPEN_CALIBRATED_MIN_RIGHT'))

        # Test that thresholds can be retrieved
        thresholds = calibrator.get_thresholds()
        self.assertIsInstance(thresholds, dict)
        self.assertIn('left_max', thresholds)
        self.assertIn('left_min', thresholds)
        self.assertIn('right_max', thresholds)
        self.assertIn('right_min', thresholds)


if __name__ == '__main__':
    unittest.main()