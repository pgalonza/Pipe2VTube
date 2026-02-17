"""
Performance monitoring module for MediaPipe + VTube Studio integration.
Provides performance metrics, logging, and benchmarking.
"""

import time
import logging
import os
from typing import Dict, Any
from collections import deque
import json
from datetime import datetime

# Try to import psutil for memory monitoring, but make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.frame_times = deque(maxlen=100)  # Last 100 frame times
        self.injection_times = deque(maxlen=100)  # Last 100 injection times
        self.memory_usage = deque(maxlen=100)  # Last 100 memory usage samples
        self.skipped_frames = 0
        self.api_errors = 0
        self.reconnections = 0
        self.start_time = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        if self.frame_count > 0:
            avg_frame_time = self.total_processing_time / self.frame_count
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = 0
            fps = 0
            
        return {
            "frame_count": self.frame_count,
            "avg_frame_time_ms": avg_frame_time * 1000,
            "current_fps": fps,
            "skipped_frames": self.skipped_frames,
            "api_errors": self.api_errors,
            "reconnections": self.reconnections,
            "avg_injection_time_ms": (
                sum(self.injection_times) / len(self.injection_times) * 1000
            ) if self.injection_times else 0,
            "max_frame_time_ms": (
                max(self.frame_times) * 1000
            ) if self.frame_times else 0,
            "current_memory_mb": (
                self.memory_usage[-1] if self.memory_usage else 0
            ),
            "avg_memory_mb": (
                sum(self.memory_usage) / len(self.memory_usage)
            ) if self.memory_usage else 0
        }


class PerformanceMonitor:
    """
    Performance monitoring for the MediaPipe + VTube Studio pipeline.
    """
    
    def __init__(self, log_interval: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            log_interval: Interval in seconds between performance log entries
        """
        self.metrics = PerformanceMetrics()
        self.log_interval = log_interval
        self.last_log_time = 0.0
        self.frame_start_time = 0.0
        self.process = psutil.Process(os.getpid()) if psutil else None
        
    def start_frame(self):
        """Mark the start of frame processing."""
        self.frame_start_time = time.time()
        
    def end_frame(self, skipped: bool = False):
        """
        Mark the end of frame processing and record metrics.
        
        Args:
            skipped: Whether this frame was skipped
        """
        if skipped:
            self.metrics.skipped_frames += 1
            return
            
        if self.frame_start_time > 0:
            frame_time = time.time() - self.frame_start_time
            self.metrics.frame_times.append(frame_time)
            self.metrics.total_processing_time += frame_time
            self.metrics.frame_count += 1
            
            # Record memory usage periodically
            if (self.metrics.frame_count % 30 == 0 and
                    self.process):  # Every 30 frames
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    self.metrics.memory_usage.append(memory_mb)
                except Exception:
                    # Ignore memory monitoring errors
                    pass
        
    def record_injection_time(self, injection_time: float):
        """
        Record the time taken for parameter injection.
        
        Args:
            injection_time: Time taken for injection in seconds
        """
        self.metrics.injection_times.append(injection_time)
        
    def record_api_error(self):
        """Record an API error."""
        self.metrics.api_errors += 1
        
    def record_reconnection(self):
        """Record a reconnection event."""
        self.metrics.reconnections += 1
        
    def should_log(self) -> bool:
        """
        Check if it's time to log performance metrics.
        
        Returns:
            True if performance metrics should be logged
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False
        
    def log_performance(self):
        """Log current performance metrics."""
        metrics_dict = self.metrics.to_dict()
        
        # Format the log message
        log_msg = (
            f"Performance Stats - "
            f"Frames: {metrics_dict['frame_count']}, "
            f"FPS: {metrics_dict['current_fps']:.1f}, "
            f"Avg Frame Time: {metrics_dict['avg_frame_time_ms']:.2f}ms, "
            f"Skipped Frames: {metrics_dict['skipped_frames']}, "
            f"API Errors: {metrics_dict['api_errors']}, "
            f"Avg Injection: {metrics_dict['avg_injection_time_ms']:.2f}ms, "
            f"Memory: {metrics_dict['current_memory_mb']:.1f}MB"
        )
        
        logger.info(log_msg)
        
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics.
        
        Returns:
            Dictionary with detailed performance statistics
        """
        return self.metrics.to_dict()
        
    def reset(self):
        """Reset all performance metrics."""
        self.metrics = PerformanceMetrics()
        self.last_log_time = 0.0
        self.frame_start_time = 0.0


class BenchmarkRunner:
    """
    Benchmarking utilities for performance testing.
    """
    
    def __init__(self):
        self.benchmarks = {}
        
    def run_benchmark(
        self,
        name: str,
        func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a benchmark on a function.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Dictionary with benchmark results
        """
        start_time = time.time()
        start_memory = (
            psutil.Process(os.getpid()).memory_info().rss
            if psutil else 0
        )
        
        # Run the function
        func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = (
            psutil.Process(os.getpid()).memory_info().rss
            if psutil else 0
        )
        
        execution_time = end_time - start_time
        memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
        
        benchmark_result = {
            "name": name,
            "execution_time_ms": execution_time * 1000,
            "memory_used_mb": memory_used,
            "timestamp": datetime.now().isoformat()
        }
        
        self.benchmarks[name] = benchmark_result
        
        logger.info(
            f"Benchmark '{name}' - "
            f"Time: {execution_time*1000:.2f}ms, "
            f"Memory: {memory_used:.2f}MB"
        )
        
        return benchmark_result
        
    def save_benchmarks(self, filename: str = "benchmark_results.json"):
        """
        Save benchmark results to a JSON file.
        
        Args:
            filename: Name of the file to save results to
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.benchmarks, f, indent=2)
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
            
    def get_benchmark_report(self) -> str:
        """
        Generate a formatted benchmark report.
        
        Returns:
            Formatted string with benchmark results
        """
        if not self.benchmarks:
            return "No benchmarks run yet."
            
        report = "Benchmark Report:\n" + "="*50 + "\n"
        for name, result in self.benchmarks.items():
            report += (
                f"{name}:\n"
                f"  Execution Time: {result['execution_time_ms']:.2f}ms\n"
                f"  Memory Used: {result['memory_used_mb']:.2f}MB\n"
                f"  Timestamp: {result['timestamp']}\n\n"
            )
            
        return report


# Global instances for use throughout the application
performance_monitor = PerformanceMonitor()
benchmark_runner = BenchmarkRunner()


def setup_performance_logging():
    """Setup performance logging configuration."""
    # Add performance logging handler if not already present
    performance_logger = logging.getLogger('performance')
    if not performance_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        handler.setFormatter(formatter)
        performance_logger.addHandler(handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False


# Initialize performance logging
setup_performance_logging()