import os
import torch
from typing import Dict, Any, Optional
from .logger import logger

class TelemetryTracker:
    """Tracks GPU VRAM usage, system memory, and reports hardware telemetry to W&B."""

    def __init__(self, wandb_run: Optional[Any] = None):
        self.wandb_run = wandb_run
        self.cuda_available = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU / MPS"

    def get_vram_stats(self) -> Dict[str, float]:
        """Returns current VRAM allocated and reserved in Gigabytes (GB)."""
        if not self.cuda_available:
            return {"vram_allocated_gb": 0.0, "vram_reserved_gb": 0.0, "vram_max_allocated_gb": 0.0}

        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

        return {
            "vram_allocated_gb": round(allocated, 3),
            "vram_reserved_gb": round(reserved, 3),
            "vram_max_allocated_gb": round(max_allocated, 3)
        }

    def log_telemetry(self, step: Optional[int] = None, extra_metrics: Optional[Dict[str, Any]] = None):
        """Logs current VRAM metrics and extra telemetry data to W&B if active."""
        stats = self.get_vram_stats()
        if extra_metrics:
            stats.update(extra_metrics)

        logger.debug(f"Hardware Telemetry: Device={self.device_name} | Stats={stats}")

        if self.wandb_run is not None:
            log_payload = {f"telemetry/{k}": v for k, v in stats.items()}
            if step is not None:
                log_payload["step"] = step
            self.wandb_run.log(log_payload)

    def print_summary(self):
        """Prints a summary log of peak memory usage."""
        stats = self.get_vram_stats()
        logger.info(
            f"[bold green]Hardware Summary:[/bold green] Device={self.device_name} | "
            f"Peak VRAM: {stats['vram_max_allocated_gb']} GB | "
            f"Currently Reserved: {stats['vram_reserved_gb']} GB"
        )
