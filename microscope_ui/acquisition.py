from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Condition, Lock, Thread
import time
from typing import Protocol


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
    ) -> None: ...

    def write_plane(
        self,
        channel_id: str,
        z_index: int,
        payload: bytes | None = None,
        width: int = 64,
        height: int = 64,
    ) -> None: ...

    def finish_tile(self): ...


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
    payload: bytes
    width: int
    height: int
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
                self._queue.task_done()
                with self._state_changed:
                    self._busy = False
                    self._state_changed.notify_all()
                return

            assert isinstance(item, PlaneWriteRequest)
            with self._state_changed:
                self._busy = True
                self._state_changed.notify_all()
            try:
                if current_tile is None:
                    self._sink.begin_tile(
                        item.tile_index,
                        item.x_um,
                        item.y_um,
                        item.fov_width_um,
                        item.fov_height_um,
                        item.z_planes,
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
                    item.payload,
                    item.width,
                    item.height,
                )
                with self._state_changed:
                    self._written_frames += 1
                    self._pending_frames -= 1

                if item.is_last_plane:
                    self._sink.finish_tile()
                    current_tile = None
                    with self._state_changed:
                        self._completed_tiles += 1
            except Exception as exc:  # The UI reports the exact storage failure.
                with self._state_changed:
                    self._failure = f"{type(exc).__name__}: {exc}"
                    self._accepting = False
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
