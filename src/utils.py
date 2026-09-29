"""Small helpers shared by every script: seeding, hardware info, run log, file I/O."""
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np

import config

# Windows consoles default to cp1252; make printing of any query text safe
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def set_seed(seed=config.SEED):
    """Seed 42 everywhere: python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def hardware():
    """One-line hardware description for the run log."""
    cpu = f"CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} threads)"
    if cuda_available():
        import torch
        return f"{cpu}; GPU: {torch.cuda.get_device_name(0)}"
    return f"{cpu}; GPU: none"


def log_run(command, duration_s, outcome, hw=None):
    """Append one run to results/run_log.md (timestamp, command, hardware, duration, outcome)."""
    new = not config.RUN_LOG.exists()
    with open(config.RUN_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("# Run log\n\n| Timestamp | Command | Hardware | Duration (s) | Outcome |\n"
                    "|---|---|---|---|---|\n")
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M:%S} | `{command}` | {hw or hardware()} "
                f"| {duration_s:.1f} | {outcome} |\n")


class Timer:
    """with Timer() as t: ...  then t.seconds"""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
