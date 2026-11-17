"""Tk view helpers that need no display."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tkinter")

from microscope_ui.app import _frame_to_pgm  # noqa: E402


def test_camera_frame_becomes_a_scaled_pgm():
    frame = np.zeros((64, 128), dtype=np.uint16)
    frame[:, 64:] = 4000  # right half bright
    data = _frame_to_pgm(frame, 40, 20)
    header, pixels = data.split(b"\n", 3)[:3], data.split(b"\n", 3)[3]
    assert header == [b"P5", b"40 20", b"255"]
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(20, 40)
    assert image[:, :20].max() == 0 and image[:, 20:].min() == 255  # orientation kept
