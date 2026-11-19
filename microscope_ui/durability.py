"""Cross-platform helpers for making acquisition metadata durable."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def durable_fsync(file_descriptor: int) -> None:
    """Flush a file through the drive cache where the OS exposes that operation."""

    if sys.platform == "darwin":
        try:
            import fcntl

            fcntl.fcntl(file_descriptor, getattr(fcntl, "F_FULLFSYNC", 51))
            return
        except (AttributeError, OSError):
            # Some filesystems do not implement F_FULLFSYNC. A normal fsync is
            # still the strongest available fallback there.
            pass
    os.fsync(file_descriptor)


def durable_fsync_directory(path: Path | str) -> None:
    """Persist directory-entry changes when supported by the filesystem."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        durable_fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
