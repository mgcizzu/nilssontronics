"""Qt view (work in progress) built on qtpy, ndv and pymmcore-widgets.

It drives the same :class:`~microscope_ui.controller.ScanController` as the Tk
view; nothing in the controller, engine or planner is Qt-specific.

Install with ``pip install -e .[qt]`` and start with ``python -m microscope_ui.qt``.
"""

from .window import ScanWindow, main

__all__ = ["ScanWindow", "main"]
