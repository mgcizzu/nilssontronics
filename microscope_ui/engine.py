"""Acquisition engine: runs one cycle on a worker thread.

The engine is independent of any GUI. The interface (Tk today, Qt next) only
starts it, polls :meth:`AcquisitionEngine.status`, and calls :meth:`pause`,
:meth:`resume` or :meth:`abort`.

Per tile the order is: XY move (optionally with a consistent approach
direction) → for each channel: set channel/exposure/power → for each Z:
move Z, snap, hand the frame to the bounded SSD writer.

Pausing takes effect between tiles. Aborting takes effect between frames; the
unfinished tile is discarded and re-acquired on resume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Condition, Event, Lock, Thread
import time
from typing import Callable, Literal

from .acquisition import BoundedAsyncWriter, PlaneWriteRequest
from .backend import HardwareError, MicroscopeBackend
from .models import ChannelSetting, MicroscopeConfig, Tile

EngineState = Literal[
    "idle", "running", "pausing", "paused", "aborting", "draining",
    "post-processing", "complete", "aborted", "failed",
]
FINAL_STATES = ("complete", "aborted", "failed")


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState
    tiles_total: int
    tiles_done: int
    tiles_skipped: int
    frames_acquired: int
    current_tile: Tile | None
    current_channel_id: str | None
    current_z_index: int
    current_z_planes: int
    backpressure: bool
    message: str = ""
    error: str | None = None
    started_monotonic: float = 0.0
    finished_monotonic: float | None = None

    @property
    def finished(self) -> bool:
        return self.state in FINAL_STATES

    @property
    def fraction(self) -> float:
        return self.tiles_done / self.tiles_total if self.tiles_total else 1.0


@dataclass(frozen=True)
class CycleRun:
    """Everything the engine needs to acquire one cycle."""

    objective_id: str
    tiles: tuple[Tile, ...]
    channels: tuple[ChannelSetting, ...]
    z_positions: dict[int, tuple[float, ...]]  # tile.index → absolute Z list
    skip_tiles: frozenset[int] = frozenset()
    consistent_approach: bool = False
    shutter_per_frame: bool = True


class AcquisitionEngine:
    def __init__(
        self,
        config: MicroscopeConfig,
        backend: MicroscopeBackend,
        writer: BoundedAsyncWriter,
        run: CycleRun,
        on_tile_written: Callable[[int], None] | None = None,
        post_process: Callable[[Callable[[], bool]], str | None] | None = None,
        on_finished: Callable[[EngineStatus], None] | None = None,
        writer_stop_timeout_s: float = 5.0,
    ) -> None:
        missing = [tile.index for tile in run.tiles if tile.index not in run.z_positions]
        if missing:
            raise ValueError(f"No Z positions for tiles {missing[:5]}")
        self._config = config
        self._backend = backend
        self._writer = writer
        self._run = run
        self._post_process = post_process
        self._on_finished = on_finished
        self._writer_stop_timeout_s = writer_stop_timeout_s
        self._lock = Lock()
        self._resume_signal = Condition(self._lock)
        self._abort = Event()
        self._pause_requested = False
        skipped = sum(1 for tile in run.tiles if tile.index in run.skip_tiles)
        self._status = EngineStatus(
            state="idle",
            tiles_total=len(run.tiles),
            tiles_done=skipped,
            tiles_skipped=skipped,
            frames_acquired=0,
            current_tile=None,
            current_channel_id=None,
            current_z_index=0,
            current_z_planes=0,
            backpressure=False,
        )
        self._writer_tiles_at_start = writer.snapshot().completed_tiles
        self._thread = Thread(target=self._thread_main, name="acquisition-engine", daemon=True)

    # -- control (any thread) ----------------------------------------------------
    def start(self) -> None:
        self._set(state="running", started_monotonic=time.monotonic(), message="Starting")
        self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if self._status.state == "running":
                self._pause_requested = True
                self._status = replace(self._status, state="pausing", message="Pausing after this tile")

    def resume(self) -> None:
        with self._lock:
            self._pause_requested = False
            if self._status.state in ("pausing", "paused"):
                self._status = replace(self._status, state="running", message="Resumed")
            self._resume_signal.notify_all()

    def abort(self) -> None:
        self._abort.set()
        with self._lock:
            if self._status.state not in FINAL_STATES:
                self._status = replace(self._status, state="aborting", message="Stopping safely")
            self._resume_signal.notify_all()

    def status(self) -> EngineStatus:
        writer = self._writer.snapshot()
        with self._lock:
            done = self._status.tiles_skipped + max(
                0, writer.completed_tiles - self._writer_tiles_at_start
            )
            if self._status.state in FINAL_STATES:
                done = self._status.tiles_done
            return replace(
                self._status,
                tiles_done=min(done, self._status.tiles_total),
                backpressure=writer.queue_depth >= writer.queue_capacity,
            )

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- worker ------------------------------------------------------------------
    def _set(self, **changes) -> None:
        with self._lock:
            self._status = replace(self._status, **changes)

    def _thread_main(self) -> None:
        try:
            outcome = self._acquire()
        except HardwareError as exc:
            outcome = ("failed", f"Hardware: {exc}")
        except Exception as exc:  # report anything unexpected to the operator
            outcome = ("failed", f"{type(exc).__name__}: {exc}")
        state, error = outcome
        self._safe_close_shutter()

        if state == "complete":
            self._set(state="draining", message="Camera complete — draining SSD buffer")
            if not self._writer.wait_until_idle():
                state, error = "failed", f"SSD writer: {self._writer.snapshot().failure}"
            stopped = self._writer.shutdown(drain=True, timeout=self._writer_stop_timeout_s)
        else:
            stopped = self._writer.shutdown(drain=False, timeout=self._writer_stop_timeout_s)
        if not stopped:
            # Never report a final state while the writer can still touch the cycle's
            # storage: resume/close would close it underneath an in-flight write.
            self._wait_for_writer()

        snapshot = self._writer.snapshot()
        if snapshot.failure and state != "failed":
            state, error = "failed", f"SSD writer: {snapshot.failure}"
        done = self._status.tiles_skipped + max(0, snapshot.completed_tiles - self._writer_tiles_at_start)

        message = ""
        if state == "complete" and self._post_process is not None:
            self._set(state="post-processing", message="Estimating flat-field")
            try:
                message = self._post_process(self._abort.is_set) or ""
            except Exception as exc:
                message = f"Flat-field estimation failed: {exc}"

        with self._lock:
            final = replace(
                self._status,
                state=state,
                error=error,
                tiles_done=min(done, self._status.tiles_total),
                current_tile=None,
                backpressure=False,
                message=message
                or {"complete": "Cycle complete", "aborted": "Stopped by operator"}.get(state, error or ""),
                finished_monotonic=time.monotonic(),
            )
        # Persist first, publish second: until the final state is visible the
        # controller stays busy, so nobody can switch experiments before this
        # cycle's record is saved.
        if self._on_finished is not None:
            self._set(message="Saving the cycle record")
            try:
                self._on_finished(final)
            except Exception as exc:  # the data is on disk; report the bookkeeping failure
                final = replace(
                    final,
                    error=(final.error + "; " if final.error else "")
                    + f"Could not save the cycle record: {exc}",
                )
        with self._lock:
            self._status = final

    def _wait_for_writer(self) -> None:
        started = time.monotonic()
        while not self._writer.wait_stopped(timeout=1.0):
            waited = int(time.monotonic() - started)
            self._set(message=f"Waiting for the SSD writer to finish its current frame ({waited} s)")

    def _wait_if_paused(self) -> bool:
        """Return False if aborted while paused."""

        with self._lock:
            if not self._pause_requested:
                return not self._abort.is_set()
        # Let the writer save the tile that was just queued before reporting
        # "paused", so every acquired tile is on disk while the operator waits.
        self._writer.wait_until_idle()
        self._safe_close_shutter()
        with self._lock:
            if self._pause_requested and not self._abort.is_set():
                self._status = replace(
                    self._status, state="paused", current_tile=None, message="Paused between tiles"
                )
            while self._pause_requested and not self._abort.is_set():
                self._resume_signal.wait(0.2)
            if not self._abort.is_set():
                self._status = replace(self._status, state="running", message="Resumed")
        return not self._abort.is_set()

    def _acquire(self) -> tuple[EngineState, str | None]:
        run = self._run
        backend = self._backend
        backend.set_shutter_per_frame(run.shutter_per_frame)
        fov_width_um, fov_height_um = self._config.fov_um(run.objective_id)
        approach = self._config.approach_offset_um if run.consistent_approach else 0.0
        channel_lookup = {c.id: c for c in self._config.channels}
        frames = 0

        for tile in run.tiles:
            if tile.index in run.skip_tiles:
                continue
            if not self._wait_if_paused():
                return ("aborted", None)
            z_list = run.z_positions[tile.index]
            self._set(current_tile=tile, current_z_planes=len(z_list), message=f"Tile {tile.index + 1}")

            if approach > 0:
                backend.move_xy(tile.x_um - approach, tile.y_um - approach)
            backend.move_xy(tile.x_um, tile.y_um)

            for channel_number, setting in enumerate(run.channels):
                channel = channel_lookup[setting.channel_id]
                backend.set_channel(channel, setting.exposure_ms, setting.power_percent)
                if not run.shutter_per_frame:
                    backend.set_shutter_open(True)
                for z_index, z_um in enumerate(z_list):
                    if self._abort.is_set():
                        return ("aborted", None)
                    if abs(backend.position.z_um - z_um) > 1e-6:
                        if not run.shutter_per_frame:
                            backend.set_shutter_open(False)
                        backend.move_z(z_um)
                        if not run.shutter_per_frame:
                            backend.set_shutter_open(True)
                    self._set(current_channel_id=channel.id, current_z_index=z_index)
                    frame = backend.snap()
                    request = PlaneWriteRequest(
                        tile_index=tile.index,
                        x_um=tile.x_um,
                        y_um=tile.y_um,
                        fov_width_um=fov_width_um,
                        fov_height_um=fov_height_um,
                        z_planes=len(z_list),
                        channel_id=channel.id,
                        z_index=z_index,
                        frame=frame,
                        z_um=z_um,
                        row=tile.row,
                        col=tile.col,
                        z_positions_um=tuple(z_list),
                        is_last_plane=(
                            channel_number == len(run.channels) - 1
                            and z_index == len(z_list) - 1
                        ),
                    )
                    if not self._writer.submit(request, self._abort.is_set):
                        if self._abort.is_set():
                            return ("aborted", None)
                        return ("failed", f"SSD writer: {self._writer.snapshot().failure}")
                    frames += 1
                    self._set(frames_acquired=frames)
                if not run.shutter_per_frame:
                    backend.set_shutter_open(False)
        return ("complete", None)

    def _safe_close_shutter(self) -> None:
        try:
            self._backend.set_shutter_open(False)
        except Exception:
            pass
