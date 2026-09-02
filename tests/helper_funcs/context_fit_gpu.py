"""nvidia-smi based VRAM peak polling for the context-fit benchmark."""

from __future__ import annotations

import subprocess
import threading
from typing import Optional


class GpuPeakPoller:
    '''Polls nvidia-smi for peak VRAM usage on the specified devices at a given interval.'''

    def __init__(self, devices: list[int], interval_s: float):
        '''Initializes peak tracking for the given devices and poll interval.'''
        self.devices = devices
        self.interval_s = interval_s
        self.peak_used: dict[int, int] = {d: 0 for d in devices}
        self.total_mem: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_snapshot(self) -> None:
        '''Captures a snapshot of current VRAM usage and updates peak values.'''

        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            return

        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 3:
                continue

            try:
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])

            except ValueError:
                continue

            if idx not in self.devices:
                continue

            if used > self.peak_used.get(idx, 0):
                self.peak_used[idx] = used

            self.total_mem[idx] = total

    def _loop(self) -> None:
        '''Thread loop that polls VRAM usage until stopped.'''

        while not self._stop.is_set():
            self._read_snapshot()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        '''Takes a baseline snapshot and starts the background polling thread.'''

        self._read_snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        '''Stops the polling thread, waits for it, and takes a final snapshot.'''

        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._read_snapshot()
