from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Condition, Lock, Thread
import time
from typing import Callable, Protocol

import numpy as np


class PlaneSink(Protocol):
    """Storage boundary used by the asynchronous acquisition writer."""

    def begin_tile(
        self,
        tile_index: int,
        x_um: float,
        y_um: float,
        fov_width_um: float,
        fov_height_um: float,
        z_planes: int,
        row: int = 0,
        col: int = 0,
        z_positions_um: tuple[float, ...] = (),
    ) -> None: ...

    def write_plane(
        self,
        channel_id: str,
        z_index: int,
        frame: np.ndarray,
        z_um: float | None = None,
    ) -> None: ...

    def finish_tile(self): ...

    def abandon_tile(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PlaneWriteRequest:
    """One camera frame plus the tile context needed by the disk writer."""

    tile_index: int
    x_um: float
    y_um: float
    fov_width_um: float
    fov_height_um: float
    z_planes: int
    channel_id: str
    z_index: int
    frame: np.ndarray
    z_um: float | None = None
    row: int = 0
    col: int = 0
    z_positions_um: tuple[float, ...] = ()
    is_last_plane: bool = False


@dataclass(frozen=True, slots=True)
class WriterSnapshot:
    queue_depth: int
    queue_capacity: int
    accepted_frames: int
    written_frames: int
    completed_tiles: int
    busy: bool
    accepting: bool
    failure: str | None

    @property
    def idle(self) -> bool:
        return self.queue_depth == 0 and not self.busy


_STOP = object()


class BoundedAsyncWriter:
    """Single disk writer fed by a small, strictly bounded frame queue.

    The acquisition thread never waits for an individual disk write. It only
    pauses when ``queue_capacity`` frames are already waiting for the writer.
    Consequently RAM use is bounded while short SSD stalls can be absorbed.
    """

    def __init__(self, sink: PlaneSink, queue_capacity: int = 4) -> None:
        if queue_capacity < 1:
            raise ValueError("Writer queue capacity must be at least one frame")
        self._sink = sink
        self._queue: Queue[PlaneWriteRequest | object] = Queue(
            maxsize=queue_capacity
        )
        self._capacity = queue_capacity
        self._state_lock = Lock()
        self._state_changed = Condition(self._state_lock)
        self._accepted_frames = 0
        self._written_frames = 0
        self._completed_tiles = 0
        self._pending_frames = 0
        self._busy = False
        self._accepting = True
        self._failure: str | None = None
        self._thread = Thread(
            target=self._run,
            name="microscope-ssd-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def queue_capacity(self) -> int:
        return self._capacity

    def can_accept(self) -> bool:
        with self._state_lock:
            return (
                self._accepting
                and self._failure is None
                and self._pending_frames < self._capacity
            )

    def try_submit(self, request: PlaneWriteRequest) -> bool:
        """Queue a captured frame without blocking the acquisition thread."""

        with self._state_lock:
            if (
                not self._accepting
                or self._failure is not None
                or self._pending_frames >= self._capacity
            ):
                return False
            try:
                self._queue.put_nowait(request)
            except Full:
                return False
            self._accepted_frames += 1
            self._pending_frames += 1
            self._state_changed.notify_all()
            return True

    def submit(
        self,
        request: PlaneWriteRequest,
        should_stop: Callable[[], bool] = lambda: False,
        poll_s: float = 0.05,
    ) -> bool:
        """Block the acquisition thread only while the queue is full.

        Returns False without queuing if ``should_stop`` becomes true or the
        writer fails/stops accepting. Never drops a frame silently.
        """

        while True:
            if self.try_submit(request):
                return True
            with self._state_changed:
                if not self._accepting or self._failure is not None:
                    return False
                if should_stop():
                    return False
                self._state_changed.wait(poll_s)

    @property
    def backpressure(self) -> bool:
        with self._state_lock:
            return self._pending_frames >= self._capacity

    def snapshot(self) -> WriterSnapshot:
        with self._state_lock:
            return WriterSnapshot(
                # Includes the frame currently being written, so this is a
                # strict upper bound on frame payloads owned by the pipeline.
                queue_depth=self._pending_frames,
                queue_capacity=self._capacity,
                accepted_frames=self._accepted_frames,
                written_frames=self._written_frames,
                completed_tiles=self._completed_tiles,
                busy=self._busy,
                accepting=self._accepting,
                failure=self._failure,
            )

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Wait for tests or non-UI callers; the Tk UI polls snapshots instead."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_changed:
            while self._pending_frames or self._busy:
                if self._failure is not None:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._state_changed.wait(remaining)
            return self._failure is None

    @property
    def stopped(self) -> bool:
        """True once the writer thread has exited (no further access to the sink)."""

        return not self._thread.is_alive()

    def wait_stopped(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return self.stopped

    def shutdown(self, *, drain: bool, timeout: float = 5.0) -> bool:
        """Stop the worker after draining, or discard frames that have not begun."""

        with self._state_lock:
            if not self._accepting and not self._thread.is_alive():
                return True
            self._accepting = False
            self._state_changed.notify_all()

        if not drain:
            while True:
                try:
                    item = self._queue.get_nowait()
                except Empty:
                    break
                else:
                    self._queue.task_done()
                    if item is _STOP:
                        break
                    with self._state_changed:
                        self._pending_frames -= 1
                        self._state_changed.notify_all()

        try:
            self._queue.put(_STOP, timeout=max(0.0, timeout))
        except Full:
            return False
        self._thread.join(timeout=max(0.0, timeout))
        return not self._thread.is_alive()

    def _run(self) -> None:
        current_tile: int | None = None
        while True:
            item = self._queue.get()
            if item is _STOP:
                if current_tile is not None:
                    # Unfinished tile (abort/stop): discard it so the resume
                    # journal never lists a partial tile.
                    self._safe_abandon()
                    current_tile = None
                self._queue.task_done()
                with self._state_changed:
                    self._busy = False
                    self._state_changed.notify_all()
                return

            assert isinstance(item, PlaneWriteRequest)
            with self._state_changed:
                self._busy = True
                self._state_changed.notify_all()
            plane_counted = False
            try:
                if current_tile is None:
                    self._sink.begin_tile(
                        item.tile_index,
                        item.x_um,
                        item.y_um,
                        item.fov_width_um,
                        item.fov_height_um,
                        item.z_planes,
                        item.row,
                        item.col,
                        item.z_positions_um,
                    )
                    current_tile = item.tile_index
                elif current_tile != item.tile_index:
                    raise RuntimeError(
                        f"Tile {current_tile} was not finalized before tile "
                        f"{item.tile_index}"
                    )

                self._sink.write_plane(
                    item.channel_id,
                    item.z_index,
                    item.frame,
                    item.z_um,
                )
                with self._state_changed:
                    self._written_frames += 1
                    self._pending_frames -= 1
                    self._state_changed.notify_all()
                plane_counted = True

                if item.is_last_plane:
                    self._sink.finish_tile()
                    current_tile = None
                    with self._state_changed:
                        self._completed_tiles += 1
            except Exception as exc:  # The UI reports the exact storage failure.
                self._safe_abandon()
                with self._state_changed:
                    self._failure = f"{type(exc).__name__}: {exc}"
                    self._accepting = False
                    if not plane_counted:
                        self._pending_frames -= 1
                    self._busy = False
                    self._state_changed.notify_all()
                self._queue.task_done()
                self._discard_pending()
                return
            else:
                self._queue.task_done()
                with self._state_changed:
                    self._busy = False
                    self._state_changed.notify_all()

    def _safe_abandon(self) -> None:
        abandon = getattr(self._sink, "abandon_tile", None)
        if abandon is None:
            return
        try:
            abandon()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def _discard_pending(self) -> None:
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            else:
                self._queue.task_done()
                if item is not _STOP:
                    discarded += 1
        if discarded:
            with self._state_changed:
                self._pending_frames -= discarded
                self._state_changed.notify_all()
