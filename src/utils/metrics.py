"""
Metrics recording utilities for tracking model performance and resource usage.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import psutil
import torch


class MetricsRecorder:
    """
    Records and tracks metrics for model inference including memory usage,
    inference time, and generation parameters.
    """

    def __init__(self, save_dir: str = "results/metrics"):
        """
        Initialize the MetricsRecorder.

        Args:
            save_dir: Directory to save metrics files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_session: Dict[str, Any] = {}

    def start_recording(self, operation_name: str, **kwargs) -> None:
        """
        Start recording metrics for an operation.

        Args:
            operation_name: Name of the operation being recorded
            **kwargs: Additional metadata to record
        """
        self.current_session = {
            "operation": operation_name,
            "start_time": time.time(),
            "timestamp": datetime.now().isoformat(),
            "metadata": kwargs
        }

        # Record initial memory state
        self.current_session["memory_start"] = self._get_memory_usage()

    def stop_recording(self, **kwargs) -> Dict[str, Any]:
        """
        Stop recording and calculate final metrics.

        Args:
            **kwargs: Additional data to record (e.g., output_tokens, prompt_length)

        Returns:
            Dictionary containing all recorded metrics
        """
        if not self.current_session:
            raise RuntimeError("No recording session active. Call start_recording() first.")

        end_time = time.time()
        duration = end_time - self.current_session["start_time"]

        # Record final memory state
        memory_end = self._get_memory_usage()

        # Calculate metrics
        metrics = {
            **self.current_session["metadata"],
            **kwargs,
            "operation": self.current_session["operation"],
            "timestamp": self.current_session["timestamp"],
            "duration_seconds": round(duration, 3),
            "memory_start_mb": self.current_session["memory_start"],
            "memory_end_mb": memory_end,
            "memory_delta_mb": {
                k: round(memory_end[k] - self.current_session["memory_start"][k], 2)
                for k in memory_end.keys()
            }
        }

        # Calculate throughput if token counts provided
        if "output_tokens" in kwargs and duration > 0:
            metrics["tokens_per_second"] = round(kwargs["output_tokens"] / duration, 2)

        self.metrics_history.append(metrics)
        self.current_session = {}

        return metrics

    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage for CPU and GPU.

        Returns:
            Dictionary with memory usage in MB
        """
        memory_info = {
            "cpu_ram_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "cpu_ram_percent": round(psutil.virtual_memory().percent, 2)
        }

        # Add GPU memory if CUDA is available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info[f"gpu_{i}_allocated_mb"] = round(
                    torch.cuda.memory_allocated(i) / 1024 / 1024, 2
                )
                memory_info[f"gpu_{i}_reserved_mb"] = round(
                    torch.cuda.memory_reserved(i) / 1024 / 1024, 2
                )

        return memory_info

    def get_current_memory(self) -> Dict[str, float]:
        """
        Get current memory usage without recording.

        Returns:
            Dictionary with current memory usage
        """
        return self._get_memory_usage()

    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """
        Save metrics history to a JSON file.

        Args:
            filename: Optional filename. If not provided, generates timestamp-based name

        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                "metrics": self.metrics_history,
                "summary": self._generate_summary()
            }, f, indent=2)

        return filepath

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from metrics history.

        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}

        summary = {
            "total_operations": len(self.metrics_history),
            "total_duration_seconds": sum(m["duration_seconds"] for m in self.metrics_history)
        }

        # Calculate average tokens per second if available
        tps_values = [m["tokens_per_second"] for m in self.metrics_history if "tokens_per_second" in m]
        if tps_values:
            summary["avg_tokens_per_second"] = round(sum(tps_values) / len(tps_values), 2)
            summary["max_tokens_per_second"] = round(max(tps_values), 2)
            summary["min_tokens_per_second"] = round(min(tps_values), 2)

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of recorded metrics."""
        if not self.metrics_history:
            print("No metrics recorded yet.")
            return

        summary = self._generate_summary()
        print("\n" + "="*50)
        print("METRICS SUMMARY")
        print("="*50)
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")

        if "avg_tokens_per_second" in summary:
            print(f"\nThroughput:")
            print(f"  Average: {summary['avg_tokens_per_second']:.2f} tokens/s")
            print(f"  Max: {summary['max_tokens_per_second']:.2f} tokens/s")
            print(f"  Min: {summary['min_tokens_per_second']:.2f} tokens/s")

        print("\nLatest Memory Usage:")
        latest = self.metrics_history[-1]
        mem_end = latest["memory_end_mb"]
        print(f"  CPU RAM: {mem_end['cpu_ram_mb']:.2f} MB ({mem_end['cpu_ram_percent']:.1f}%)")

        if any(k.startswith("gpu_") for k in mem_end.keys()):
            for key, value in mem_end.items():
                if key.startswith("gpu_") and "allocated" in key:
                    print(f"  {key}: {value:.2f} MB")

        print("="*50 + "\n")

    def clear_history(self) -> None:
        """Clear all recorded metrics."""
        self.metrics_history = []
        self.current_session = {}
