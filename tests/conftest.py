"""Shared test setup.

Qt tests create and close many windows. Under PySide6 a closed window's C++
object is destroyed when Python garbage-collects its wrapper, which can happen
at an arbitrary moment, including while Qt is dispatching a timer to one of its
children (cursor blink, scrollbar animation, ...): that crashes the process.
After each test we therefore collect garbage and flush deferred deletions at a
controlled point, outside any event dispatch.
"""

from __future__ import annotations

import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _destroy_closed_qt_objects_safely():
    yield
    try:
        from qtpy.QtCore import QCoreApplication, QEvent
    except Exception:  # Qt not installed: nothing to clean up
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    for _ in range(2):  # wrappers freed by the first pass may post more deletions
        gc.collect()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
