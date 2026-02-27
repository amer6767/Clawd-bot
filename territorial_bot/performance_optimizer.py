"""
Performance Optimizer for Territorial.io Bot
==============================================
Optimizes CPU and memory usage, improves processing speed.
"""

import logging
import time
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import cv2
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    avg_fps: float = 0.0
    avg_frame_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_frame_time: float = 0.0
    frames_processed: int = 0


class FrameCache:
    """Efficient frame caching to avoid redundant processing."""
    
    def __init__(self, max_size: int = 3):
        self.cache = deque(maxlen=max_size)
        self.last_frame_hash = None
        
    def get(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get cached processed frame if available."""
        if len(self.cache) == 0:
            return None
            
        # Simple hash check
        frame_hash = hash(frame.tobytes()[:100])  # Quick hash
        
        if frame_hash == self.last_frame_hash:
            return self.cache[-1][1]  # Return most recent cached
            
        return None
    
    def put(self, frame: np.ndarray, processed: np.ndarray):
        """Cache a processed frame."""
        frame_hash = hash(frame.tobytes()[:100])
        self.cache.append((frame_hash, processed))
        self.last_frame_hash = frame_hash


class PerformanceOptimizer:
    """
    Performance optimization system for the bot.
    
    Features:
    - Frame skipping for efficiency
    - Intelligent caching
    - Memory management
    - Processing time optimization
    - Adaptive quality settings
    """
    
    def __init__(self, target_fps: int = 10):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = 0.0
        
        # Optimization settings
        self.frame_skip = 1  # Process every N frames
        self.last_processed_frame = 0
        self.enable_cache = True
        self.frame_cache = FrameCache(max_size=3)
        
        # Adaptive quality
        self.current_quality = "high"  # "high", "medium", "low"
        self.quality_settings = {
            "high": {
                "grid_size": (10, 10),
                "resize_factor": 1.0,
                "color_detection": True,
                "cnn_enabled": True,
            },
            "medium": {
                "grid_size": (8, 8),
                "resize_factor": 0.8,
                "color_detection": True,
                "cnn_enabled": False,
            },
            "low": {
                "grid_size": (5, 5),
                "resize_factor": 0.6,
                "color_detection": True,
                "cnn_enabled": False,
            }
        }
        
        # Threading support
        self.processing_lock = threading.Lock()
        
        logger.info(f"PerformanceOptimizer initialized (target: {target_fps} FPS)")
    
    def should_process_frame(self, frame_count: int) -> bool:
        """
        Determine if this frame should be processed based on target FPS.
        Implements frame skipping for efficiency.
        """
        if self.frame_skip <= 1:
            return True
        
        return frame_count - self.last_processed_frame >= self.frame_skip
    
    def mark_frame_processed(self, frame_count: int):
        """Mark that a frame was processed."""
        self.last_processed_frame = frame_count
    
    def update_performance_metrics(self, processing_time: float):
        """Update performance metrics after processing a frame."""
        current_time = time.time()
        
        # Calculate frame time
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        
        self.last_frame_time = current_time
        
        # Update metrics
        self.metrics.frames_processed += 1
        if len(self.frame_times) > 0:
            self.metrics.avg_frame_time = np.mean(self.frame_times)
            self.metrics.avg_fps = 1.0 / max(0.001, self.metrics.avg_frame_time)
        self.metrics.last_frame_time = processing_time
        
        # Auto-adjust quality based on performance
        self._adapt_quality()
    
    def _adapt_quality(self):
        """Adapt processing quality based on current performance."""
        if self.metrics.avg_fps < self.target_fps * 0.7:
            # Performance is poor, reduce quality
            if self.current_quality == "high":
                self.current_quality = "medium"
                self.frame_skip = 2
                logger.warning("Performance low, reducing to medium quality")
            elif self.current_quality == "medium":
                self.current_quality = "low"
                self.frame_skip = 3
                logger.warning("Performance very low, reducing to low quality")
                
        elif self.metrics.avg_fps > self.target_fps * 1.2 and self.current_quality != "high":
            # Performance is good, can increase quality
            if self.current_quality == "low":
                self.current_quality = "medium"
                self.frame_skip = 2
                logger.info("Performance improved, increasing to medium quality")
            elif self.current_quality == "medium":
                self.current_quality = "high" 
                self.frame_skip = 1
                logger.info("Performance excellent, using high quality")
    
    def get_quality_settings(self) -> dict:
        """Get current quality settings."""
        return self.quality_settings[self.current_quality].copy()
    
    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply optimizations to frame before processing.
        Returns optimized frame.
        """
        settings = self.get_quality_settings()
        
        # Resize if needed
        if settings["resize_factor"] < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * settings["resize_factor"])
            new_h = int(h * settings["resize_factor"])
            frame = cv2.resize(frame, (new_w, new_h))
        
        return frame
    
    def optimize_grid_processing(self, grid_settings: tuple) -> tuple:
        """Get optimized grid processing settings."""
        settings = self.get_quality_settings()
        return settings["grid_size"]
    
    # ── Memory Management ────────────────────────────────────────────────────
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self.frame_cache = FrameCache(max_size=3)
        logger.debug("Caches cleared")
    
    def get_memory_estimate(self) -> float:
        """Estimate current memory usage in MB."""
        import sys
        # Rough estimate based on known objects
        cache_size = 3 * 640 * 480 * 3 * 8 / (1024 * 1024)  # RGB float64
        return cache_size
    
    # ── Processing Optimization ─────────────────────────────────────────────
    
    def optimize_color_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimize color detection by using faster algorithms.
        """
        # Use UMat for GPU acceleration if available
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return frame_gray
    
    def fast_resize(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Fast image resizing using INTER_AREA."""
        if scale >= 1.0:
            return frame
        
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Use INTER_AREA for downscaling (faster and better quality)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ── Threading Support ───────────────────────────────────────────────────
    
    def try_acquire_lock(self, timeout: float = 0.01) -> bool:
        """Try to acquire processing lock with timeout."""
        return self.processing_lock.acquire(timeout=timeout)
    
    def release_lock(self):
        """Release processing lock."""
        try:
            self.processing_lock.release()
        except:
            pass
    
    # ── Performance Reporting ───────────────────────────────────────────────
    
    def get_performance_report(self) -> str:
        """Get formatted performance report."""
        return (
            f"Performance Report:\n"
            f"  FPS: {self.metrics.avg_fps:.1f} (target: {self.target_fps})\n"
            f"  Frame time: {self.metrics.avg_frame_time*1000:.1f}ms\n"
            f"  Quality: {self.current_quality}\n"
            f"  Frame skip: {self.frame_skip}\n"
            f"  Frames processed: {self.metrics.frames_processed}\n"
            f"  Memory est: {self.get_memory_estimate():.1f}MB"
        )
    
    def log_performance(self):
        """Log current performance metrics."""
        logger.info(self.get_performance_report())


class SmartFrameSelector:
    """
    Intelligently selects which frames to process for efficiency.
    Uses motion detection and change detection.
    """
    
    def __init__(self, motion_threshold: float = 5.0):
        self.motion_threshold = motion_threshold
        self.previous_frame = None
        self.frame_difference = 0.0
        
    def should_process(self, frame: np.ndarray) -> bool:
        """
        Determine if frame should be processed based on changes.
        """
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return True
        
        # Calculate difference
        try:
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize for faster comparison
            gray_current = cv2.resize(gray_current, (160, 120))
            gray_prev = cv2.resize(gray_prev, (160, 120))
            
            diff = cv2.absdiff(gray_current, gray_prev)
            self.frame_difference = np.mean(diff)
            
            # Update previous frame periodically
            if self.frame_difference > self.motion_threshold:
                self.previous_frame = frame.copy()
                return True
            
            return self.frame_difference > self.motion_threshold
            
        except Exception as e:
            logger.debug(f"Frame selection error: {e}")
            return True
    
    def get_change_magnitude(self) -> float:
        """Get magnitude of change in current frame."""
        return self.frame_difference
