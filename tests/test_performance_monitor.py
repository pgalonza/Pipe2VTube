"""
Tests for the performance monitor module.
"""

import time
import unittest
from src.performance_monitor import PerformanceMonitor, BenchmarkRunner


class TestPerformanceMonitor(unittest.TestCase):
    """Test suite for performance_monitor.py"""

    def test_performance_monitor_basic(self):
        """Test basic functionality of PerformanceMonitor."""
        monitor = PerformanceMonitor()
    
        # Test initial state
        stats = monitor.get_detailed_stats()
        self.assertEqual(stats["frame_count"], 0)
        self.assertEqual(stats["api_errors"], 0)
        self.assertEqual(stats["reconnections"], 0)
    
        # Test frame timing
        monitor.start_frame()
        time.sleep(0.01)  # 10ms delay
        monitor.end_frame(skipped=False)
    
        stats = monitor.get_detailed_stats()
        self.assertEqual(stats["frame_count"], 1)
        self.assertEqual(stats["skipped_frames"], 0)
    
        # Test skipped frame
        monitor.start_frame()
        monitor.end_frame(skipped=True)
    
        stats = monitor.get_detailed_stats()
        self.assertEqual(stats["skipped_frames"], 1)
    
        # Test API error recording
        monitor.record_api_error()
        stats = monitor.get_detailed_stats()
        self.assertEqual(stats["api_errors"], 1)
    
        # Test reconnection recording
        monitor.record_reconnection()
        stats = monitor.get_detailed_stats()
        self.assertEqual(stats["reconnections"], 1)

    def test_performance_monitor_injection_times(self):
        """Test injection time recording."""
        monitor = PerformanceMonitor()
    
        # Record some injection times
        monitor.record_injection_time(0.01)  # 10ms
        monitor.record_injection_time(0.02)  # 20ms
        monitor.record_injection_time(0.015)  # 15ms
    
        stats = monitor.get_detailed_stats()
        self.assertGreater(stats["avg_injection_time_ms"], 0)

    def test_benchmark_runner(self):
        """Test BenchmarkRunner functionality."""
        runner = BenchmarkRunner()
    
        # Simple function to benchmark
        def test_func():
            time.sleep(0.01)  # 10ms delay
            return "test"
    
        # Run benchmark
        result = runner.run_benchmark("test_benchmark", test_func)
    
        self.assertEqual(result["name"], "test_benchmark")
        # Should be more than 5ms
        self.assertGreater(result["execution_time_ms"], 5)
        self.assertIn("timestamp", result)
    
        # Get report
        report = runner.get_benchmark_report()
        self.assertIn("Benchmark Report", report)
        self.assertIn("test_benchmark", report)

    def test_should_log(self):
        """Test should_log functionality."""
        monitor = PerformanceMonitor(log_interval=1)  # 1 second interval
    
        # Should log immediately after initialization
        self.assertTrue(monitor.should_log())
    
        # Should not log immediately after logging
        self.assertFalse(monitor.should_log())


if __name__ == "__main__":
    unittest.main()