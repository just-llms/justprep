"""Metrics collection and monitoring utilities.

This module provides centralized metrics tracking for performance, quality,
and system metrics. It supports latency metrics (histograms), quality metrics
(counters/rates), and system metrics (gauges) with percentile calculations.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, system metrics will be limited")

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Centralized metrics collection and aggregation.
    
    This class provides thread-safe metrics collection with support for:
    - Latency metrics (histograms) with percentile calculations
    - Quality metrics (counters/rates)
    - System metrics (gauges)
    
    Metrics are stored in memory with sliding windows for percentile calculations.
    All operations are thread-safe.
    """
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()
    
    # Configuration
    DEFAULT_WINDOW_SIZE = 1000  # Number of recent measurements to keep
    PERCENTILES = [50, 95, 99]  # Percentiles to calculate
    
    def __init__(self) -> None:
        """Initialize MetricsCollector."""
        # Latency metrics (histograms) - store recent values for percentiles
        self._latency_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.DEFAULT_WINDOW_SIZE)
        )
        self._latency_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Quality metrics (counters)
        self._counters: Dict[str, int] = defaultdict(int)
        self._counter_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # System metrics (gauges) - current values
        self._gauges: Dict[str, float] = {}
        self._gauge_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # System metrics collection task
        self._system_metrics_task: Optional[asyncio.Task] = None
        self._system_metrics_running = False
        
        logger.info("MetricsCollector initialized")
    
    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get singleton instance of MetricsCollector.
        
        Returns:
            MetricsCollector singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def record_latency(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record latency metric with optional tags.
        
        Args:
            metric_name: Name of the metric (e.g., "vad.detection_latency")
            value: Latency value in seconds
            tags: Optional dictionary of tags (e.g., {"session_id": "abc123"})
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._latency_locks[metric_key]:
            self._latency_metrics[metric_key].append(value)
    
    def increment_counter(
        self,
        metric_name: str,
        amount: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter metric.
        
        Args:
            metric_name: Name of the metric (e.g., "vad.false_positives")
            amount: Amount to increment (default: 1)
            tags: Optional dictionary of tags
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._counter_locks[metric_key]:
            self._counters[metric_key] += amount
    
    def set_gauge(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge metric value.
        
        Args:
            metric_name: Name of the metric (e.g., "system.active_sessions")
            value: Current value
            tags: Optional dictionary of tags
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._gauge_locks[metric_key]:
            self._gauges[metric_key] = value
    
    def get_percentiles(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get percentiles (p50, p95, p99) for a latency metric.
        
        Args:
            metric_name: Name of the metric
            tags: Optional dictionary of tags
            
        Returns:
            Dictionary with p50, p95, p99 values, or empty dict if no data
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._latency_locks[metric_key]:
            values = list(self._latency_metrics[metric_key])
        
        if not values:
            return {}
        
        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        percentiles = {}
        for p in self.PERCENTILES:
            index = int(n * (p / 100.0))
            # Clamp index to valid range
            index = min(index, n - 1)
            percentiles[f"p{p}"] = sorted_values[index]
        
        return percentiles
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value.
        
        Args:
            metric_name: Name of the metric
            tags: Optional dictionary of tags
            
        Returns:
            Current counter value
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._counter_locks[metric_key]:
            return self._counters.get(metric_key, 0)
    
    def get_gauge(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value.
        
        Args:
            metric_name: Name of the metric
            tags: Optional dictionary of tags
            
        Returns:
            Current gauge value, or None if not set
        """
        # Create metric key with tags if provided
        if tags:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            metric_key = f"{metric_name}[{tag_str}]"
        else:
            metric_key = metric_name
        
        with self._gauge_locks[metric_key]:
            return self._gauges.get(metric_key)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics (memory, CPU)."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            self.set_gauge("system.memory_usage_mb", memory_mb)
            self.set_gauge("system.cpu_percent", cpu_percent)
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
    
    def start_system_metrics_collection(self, interval_seconds: int = 10) -> None:
        """Start background task to collect system metrics.
        
        Args:
            interval_seconds: Interval between metric collections (default: 10)
        """
        if self._system_metrics_running:
            return
        
        self._system_metrics_running = True
        
        async def collect_loop() -> None:
            """Background loop to collect system metrics."""
            while self._system_metrics_running:
                self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
        
        self._system_metrics_task = asyncio.create_task(collect_loop())
        logger.info(f"Started system metrics collection (interval={interval_seconds}s)")
    
    def stop_system_metrics_collection(self) -> None:
        """Stop system metrics collection."""
        self._system_metrics_running = False
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
            self._system_metrics_task = None
        logger.info("Stopped system metrics collection")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for logging/monitoring.
        
        Returns:
            Dictionary containing all metrics organized by type
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "latency_metrics": {},
            "counters": {},
            "gauges": {},
        }
        
        # Export latency metrics with percentiles
        for metric_key in list(self._latency_metrics.keys()):
            with self._latency_locks[metric_key]:
                values = list(self._latency_metrics[metric_key])
            
            if values:
                percentiles = self.get_percentiles(metric_key.split("[")[0])
                metrics["latency_metrics"][metric_key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    **percentiles,
                }
        
        # Export counters
        for metric_key in list(self._counters.keys()):
            with self._counter_locks[metric_key]:
                metrics["counters"][metric_key] = self._counters[metric_key]
        
        # Export gauges
        for metric_key in list(self._gauges.keys()):
            with self._gauge_locks[metric_key]:
                metrics["gauges"][metric_key] = self._gauges[metric_key]
        
        return metrics
    
    def reset_metrics(self, metric_name: Optional[str] = None) -> None:
        """Reset metrics (for testing or periodic cleanup).
        
        Args:
            metric_name: Optional specific metric to reset, or None to reset all
        """
        if metric_name:
            # Reset specific metric
            if metric_name in self._latency_metrics:
                with self._latency_locks[metric_name]:
                    self._latency_metrics[metric_name].clear()
            if metric_name in self._counters:
                with self._counter_locks[metric_name]:
                    self._counters[metric_name] = 0
            if metric_name in self._gauges:
                with self._gauge_locks[metric_name]:
                    del self._gauges[metric_name]
        else:
            # Reset all metrics
            for metric_key in list(self._latency_metrics.keys()):
                with self._latency_locks[metric_key]:
                    self._latency_metrics[metric_key].clear()
            for metric_key in list(self._counters.keys()):
                with self._counter_locks[metric_key]:
                    self._counters[metric_key] = 0
            self._gauges.clear()
        
        logger.info(f"Reset metrics: {metric_name or 'all'}")

