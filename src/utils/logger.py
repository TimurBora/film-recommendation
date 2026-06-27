import time
import os
import psutil
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    log_wall_time: bool = True
    log_cpu_time: bool = True
    log_memory_rss: bool = True
    log_memory_vms: bool = True
    log_memory_delta: bool = True
    log_cpu_percent: bool = True
    log_data_shapes: bool = True
    show_progress_bars: bool = True
    log_per_prediction: bool = False


class StepLogger:
    def __init__(self, config: LoggingConfig):
        self.config = config
        self._prev_memory = None

    def log_step(self, step_name: str, start_time: float, start_cpu: float, extra_metrics: Optional[dict] = None):
        now = time.perf_counter()
        cpu_now = time.process_time()
        mem = psutil.Process(os.getpid()).memory_info()
        cpu_percent = psutil.Process(os.getpid()).cpu_percent(interval=None)

        lines = [f"--- Step: {step_name} ---"]

        if self.config.log_wall_time:
            lines.append(f"  Wall clock time : {now - start_time:.2f} s")
        if self.config.log_cpu_time:
            lines.append(f"  CPU time (user) : {cpu_now - start_cpu:.2f} s")
        if self.config.log_memory_rss:
            lines.append(f"  Memory RSS      : {mem.rss / (1024*1024):.1f} MB")
        if self.config.log_memory_vms:
            lines.append(f"  Memory VMS      : {mem.vms / (1024*1024):.1f} MB")
        if self.config.log_memory_delta and self._prev_memory is not None:
            delta = (mem.rss - self._prev_memory) / (1024*1024)
            lines.append(f"  RSS change      : {delta:+.1f} MB")
        if self.config.log_cpu_percent:
            lines.append(f"  CPU usage       : {cpu_percent:.1f} %")

        if extra_metrics:
            for k, v in extra_metrics.items():
                lines.append(f"  {k.ljust(15)} : {v}")

        self._prev_memory = mem.rss

        logger.opt(raw=True).info("\n".join(lines) + "\n")