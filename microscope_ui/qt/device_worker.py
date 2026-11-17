"""One background thread that owns every *setup* call to the backend.

The Qt view never calls the backend on the GUI thread: jogs, objective
changes, go-to moves, snaps and live frames are queued here, and results come
back as Qt signals (delivered on the GUI thread). A long exposure or a turret
rotation therefore never freezes the window.

Only one thread may drive the backend at a time. Before a cycle starts the
view calls :meth:`DeviceWorker.stop_and_drain`, so the acquisition engine never
shares the backend with this worker.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from qtpy.QtCore import QObject, Qt, Signal

from ..backend import HardwareError, StagePosition
from ..models import Channel


@dataclass
class _Job:
    function: Callable[..., Any] | None  # None = just wake the loop
    args: tuple = ()
    on_done: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None


class DeviceWorker(QObject):
    frame_ready = Signal(object)  # np.ndarray
    position_changed = Signal(object)  # StagePosition
    live_failed = Signal(str)
    _deliver = Signal(object, object)  # callback, value -> run on GUI thread

    def __init__(self, backend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Condition(self._lock)
        self._live = False
        self._live_settings: tuple[Channel, float, float] | None = None
        self._applied: tuple[str, float, float] | None = None
        # Live/snap need per-frame shutter; a cycle may leave it off with the shutter closed.
        self._illumination_ready = False
        self._stopped = False
        self.last_position: StagePosition = backend.position
        # Bound to this QObject (not a lambda): queued results are delivered on the GUI
        # thread and are dropped together with the worker when its window is deleted.
        # A free-function connection outlives the window under PySide6 and would run
        # callbacks into already-deleted widgets.
        self._deliver.connect(self._run_callback, Qt.ConnectionType.QueuedConnection)
        self._thread = threading.Thread(target=self._run, name="device-worker", daemon=True)
        self._thread.start()

    # -- API (GUI thread) ----------------------------------------------------
    def submit(
        self,
        function: Callable[..., Any],
        *args,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._put(_Job(function, args, on_done, on_error))

    def snap(self, on_error: Callable[[Exception], None] | None = None) -> None:
        self._put(_Job(self.acquire_frame, (), self.frame_ready.emit, on_error))

    def set_live_settings(self, channel: Channel, exposure_ms: float, power_percent: float) -> None:
        with self._lock:
            self._live_settings = (channel, exposure_ms, power_percent)

    def set_live(self, on: bool) -> None:
        with self._lock:
            if on == self._live or (on and self._stopped):
                return
            self._live = on
        if on:
            self._put(_Job(None))

    def _run_callback(self, callback, value) -> None:
        with self._lock:
            stopped = self._stopped
        if not stopped:  # the window is shutting down: nobody to report to
            callback(value)

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._pending == 0 and not self._live

    def stop_and_drain(self, timeout: float = 30.0) -> bool:
        """Stop live and wait until queued jobs are done (before a cycle starts)."""

        with self._lock:
            self._live = False
            drained = self._idle.wait_for(lambda: self._pending == 0, timeout)
            # The engine changes channel/exposure/power and the shutter mode during
            # a cycle, so the next live frame must re-apply its own settings.
            self._applied = None
            self._illumination_ready = False
            return drained

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._live = False
            self._stopped = True
        self._jobs.put(None)
        self._thread.join(timeout)

    def _put(self, job: _Job) -> None:
        with self._lock:
            if self._stopped:
                raise RuntimeError("Device worker is stopped")
            self._pending += 1
        self._jobs.put(job)

    # -- worker thread ---------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                live = self._live
            try:
                job = self._jobs.get(timeout=None if not live else 0.001)
            except queue.Empty:
                job = _Job(None)
                with self._lock:
                    self._pending += 1
            if job is None:
                return
            self._run_job(job)
            with self._lock:
                live = self._live
            if live and self._jobs.empty():
                self._live_frame()
            with self._lock:
                self._pending -= 1
                if self._pending == 0:
                    self._idle.notify_all()

    def _run_job(self, job: _Job) -> None:
        if job.function is None:
            return
        try:
            result = job.function(*job.args)
        except Exception as exc:  # delivered to the GUI, never raised here
            self._applied = None
            if job.on_error is not None:
                self._deliver.emit(job.on_error, exc)
        else:
            if job.on_done is not None:
                self._deliver.emit(job.on_done, result)
        self._emit_position()

    def acquire_frame(self):
        """Snap one frame with the live settings. Worker thread only (call it from a job)."""

        with self._lock:
            settings = self._live_settings
        if settings is None:
            raise HardwareError("No live channel selected")
        channel, exposure, power = settings
        if not self._illumination_ready:
            self._backend.set_shutter_per_frame(True)  # camera opens the shutter per exposure
            self._illumination_ready = True
        state = (channel.id, exposure, power)
        if state != self._applied:
            self._backend.set_channel(channel, exposure, power)
            self._applied = state
        return self._backend.snap()

    def acquire_with(self, channel: Channel, exposure_ms: float, power_percent: float):
        """Snap one frame with explicit settings (worker thread only, from a job).

        Used by calibration; the next live frame re-applies the live settings.
        """

        if not self._illumination_ready:
            self._backend.set_shutter_per_frame(True)
            self._illumination_ready = True
        self._applied = None
        self._backend.set_channel(channel, exposure_ms, power_percent)
        self._applied = (channel.id, exposure_ms, power_percent)
        return self._backend.snap()

    def _live_frame(self) -> None:
        try:
            frame = self.acquire_frame()
        except HardwareError as exc:
            with self._lock:
                self._live = False
            self._applied = None
            self.live_failed.emit(str(exc))
            return
        self.frame_ready.emit(frame)
        self._emit_position()

    def _emit_position(self) -> None:
        try:
            position = self._backend.position
        except HardwareError:
            return
        if position != self.last_position:
            self.last_position = position
            self.position_changed.emit(position)
