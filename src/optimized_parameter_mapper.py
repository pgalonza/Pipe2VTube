"""
Optimized parameter mapper with change detection, batching, and caching.
This module provides an enhanced parameter mapping system that reduces 
API overhead by implementing parameter change detection, batching, and 
caching mechanisms.
"""

import logging
from typing import Dict, Any, Optional
from collections import deque
import time

from src.parameter_mapper import transform_mediapipe_to_vtubestudio

logger = logging.getLogger(__name__)


class OptimizedParameterMapper:
    """
    Optimized parameter mapper that implements change detection, batching, 
    and caching to reduce API overhead while maintaining real-time performance.
    """
    
    def __init__(
        self, 
        change_threshold: float = 0.01,
        batch_window: float = 0.033, 
        cache_ttl: float = 0.1,
        max_batch_size: int = 50
    ):
        """
        Initialize the optimized parameter mapper.
        
        Args:
            change_threshold: Minimum change required to send a parameter 
                             update (0-1 range)
            batch_window: Time window for batching parameters in seconds 
                         (default 33ms for 30fps)
            cache_ttl: Time-to-live for cached computations in seconds
            max_batch_size: Maximum number of parameters per batch
        """
        # Change detection settings
        self.change_threshold = change_threshold
        
        # Batching settings
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.last_batch_time = 0.0
        
        # Caching settings
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamps = {}
        
        # Previous parameters for change detection
        self._prev_params = {}
        
        # Batch queue for parameter updates
        self._batch_queue = deque()
        
        # Statistics
        self.stats = {
            "parameters_processed": 0,
            "parameters_sent": 0,
            "parameters_cached": 0,
            "batches_sent": 0
        }
        
    def _is_significant_change(
        self, 
        param_name: str, 
        old_value: float,
        new_value: float
    ) -> bool:
        """
        Determine if a parameter change is significant enough to send.
        
        Args:
            param_name: Name of the parameter
            old_value: Previous value
            new_value: New value
            
        Returns:
            True if change is significant, False otherwise
        """
        # For some critical parameters, always send updates
        critical_params = {
            "FaceAngleX", "FaceAngleY", "FaceAngleZ", 
            "EyeOpenLeft", "EyeOpenRight"
        }
        if param_name in critical_params:
            return True
            
        # For other parameters, check if change exceeds threshold
        return abs(new_value - old_value) >= self.change_threshold
        
    def _cache_get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it's still valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key in self._cache:
            cache_time = self._cache_timestamps.get(key, 0)
            if time.time() - cache_time < self.cache_ttl:
                self.stats["parameters_cached"] += 1
                return self._cache[key]
            else:
                # Expired, remove from cache
                del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
        return None
        
    def _cache_set(self, key: str, value: Any) -> None:
        """
        Set value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
        
    def _should_send_parameter(self, param_name: str, value: float) -> bool:
        """
        Determine if a parameter should be sent based on change detection.
        
        Args:
            param_name: Name of the parameter
            value: Current value
            
        Returns:
            True if parameter should be sent, False otherwise
        """
        # Always send if this is the first time we see this parameter
        if param_name not in self._prev_params:
            self._prev_params[param_name] = value
            return True
            
        # Check if change is significant
        prev_value = self._prev_params[param_name]
        should_send = self._is_significant_change(
            param_name, prev_value, value)
        
        # Update previous value
        self._prev_params[param_name] = value
        
        return should_send
        
    def _add_to_batch(self, parameters: Dict[str, float]) -> None:
        """
        Add parameters to the batch queue.
        
        Args:
            parameters: Dictionary of parameter names and values to add 
                        to batch
        """
        for param_name, value in parameters.items():
            self._batch_queue.append((param_name, value))
            
    def _should_send_batch(self) -> bool:
        """
        Determine if current batch should be sent.
        
        Returns:
            True if batch should be sent, False otherwise
        """
        current_time = time.time()
        time_since_last_batch = current_time - self.last_batch_time
        
        # Send batch if:
        # 1. Batch window has elapsed
        # 2. Batch is getting large
        return (time_since_last_batch >= self.batch_window or 
                len(self._batch_queue) >= self.max_batch_size)
        
    def _get_batch(self) -> Dict[str, float]:
        """
        Get current batch of parameters, respecting max_batch_size.
        
        Returns:
            Dictionary of parameters for current batch
        """
        batch = {}
        batch_size = 0
        
        # Get parameters for current batch (up to max_batch_size)
        while self._batch_queue and batch_size < self.max_batch_size:
            param_name, value = self._batch_queue.popleft()
            batch[param_name] = value
            batch_size += 1
            
        return batch
        
    def transform_and_optimize(
        self, 
        mediapipe_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Transform MediaPipe data to VTube Studio parameters with optimization.
        
        Args:
            mediapipe_data: Raw MediaPipe tracking data
            
        Returns:
            Dictionary of optimized parameters ready for sending
        """
        # Transform MediaPipe data to VTube Studio parameters
        # This uses the existing transformation logic
        vtube_params = transform_mediapipe_to_vtubestudio(mediapipe_data)
        
        # Apply change detection
        filtered_params = {}
        for param_name, value in vtube_params.items():
            self.stats["parameters_processed"] += 1
            if self._should_send_parameter(param_name, value):
                filtered_params[param_name] = value
                self.stats["parameters_sent"] += 1
                
        # Add filtered parameters to batch queue
        self._add_to_batch(filtered_params)
        
        # Check if we should send a batch
        if self._should_send_batch():
            # Return the batch to send
            batch = self._get_batch()
            self.last_batch_time = time.time()
            self.stats["batches_sent"] += 1
            return batch
            
        # Return empty dict if no batch is ready to send
        return {}
        
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about parameter processing.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """
        Reset statistics counters.
        """
        self.stats = {
            "parameters_processed": 0,
            "parameters_sent": 0,
            "parameters_cached": 0,
            "batches_sent": 0
        }
        
    def flush_batch(self) -> Dict[str, float]:
        """
        Flush any remaining parameters in the batch queue.
        
        Returns:
            Dictionary of remaining parameters
        """
        batch = self._get_batch()
        if batch:
            self.stats["batches_sent"] += 1
        return batch


# Global instance for use throughout the application
optimized_mapper = OptimizedParameterMapper()