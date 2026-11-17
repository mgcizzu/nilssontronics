# CLAUDE.md — nilssontronics

Simplified, workflow-driven desktop front end for Micro-Manager, built for **cyclic
imaging of large tissue sections** (combinatorial decoding happens downstream).
See README.md for the architecture, data layout and configuration.

## Decisions — don't revisit without asking Marco
- **Experiment → Cycle:** objective, ROI set, overlap and tile order are fixed by
  cycle 1. Every cycle replays the identical tile list (index, row/col, order).
- **No marker-to-channel mapping.** Channels, exposures and powers are free per
  cycle and only recorded in `experiment.json`.
- **Re-registration = XY translation only** (drag "Move ROI set" or
  "Align selected F → stage"). No rotation/affine, no automatic registration.
- **Refocus = manual**, guided "Refocus all" through stored focus points (Z only).
  No software/hardware autofocus for now.
- **Output format is NOT ASHLAR/MCMICRO.** It must match a specific downstream
  pipeline whose file names/format are still to be agreed. `.tstack` in
  `mosaic.py` is an interim container; keep format knowledge inside `mosaic.py`.
- **Resume** works per cycle from `cycle_NN/tiles.jsonl`; a new cycle can't start
  while an earlier one is unfinished.
- **Optional features stay optional:** serpentine order, consistent stage approach,
  shutter closed during moves (default on), flat-field profiles (non-destructive).

## Architecture rules
- `app.py` (Tk view) talks only to `ScanController` (`controller.py`); it never
  calls devices for acquisition. The engine (`engine.py`) runs on a worker thread.
- Keep `controller.py`, `engine.py`, `experiment.py`, `planner.py`, `focus.py`,
  `mosaic.py` GUI-independent and unit-tested; the planned Qt view must reuse them.
- Backends implement the `MicroscopeBackend` protocol (`backend.py`). Out-of-range
  XY/Z targets raise `StageLimitError` **before** any command; never clamp.
- Frames from `snap()` are already oriented (+col = +stage X, +row = +stage Y).

## Commands
```bash
pip install -e ".[hardware,qt,test]"
python3 -m pytest                              # 80 tests, no display needed
python3 -m microscope_ui                       # simulator
python3 -m microscope_ui --backend mmcore      # pymmcore-plus (hardware section of config)
python3 -m microscope_ui.qt                    # Qt view (same flags; --no-ndv for plain image view)
```

## Open items (in order)
1. Qt port (`microscope_ui/qt/`), keeping controller/engine as is. Done: cycle workflow,
   overview canvas (draw rectangle/polygon, drag "Move ROI set", double-click stage move,
   focus-point click, mosaic underlay), focus points, live/snap via ndv, MMCore device
   docks. All setup device calls run on `DeviceWorker` (never the GUI thread), drained
   before the engine starts. Guided "Refocus all" and "Align selected F → stage" read the
   position through the worker queue. Missing vs Tk: spiral preview, recipes, flat-field
   UI, admin dialog.
   Never use pymmcore-widgets StageWidget: it bypasses backend stage-limit checks.
2. Agree output file naming/format with the downstream pipeline, then replace `.tstack`.
3. Before any real-hardware run: calibrate turret positions, pixel sizes, camera
   orientation, stage limits and safe Z ranges at the microscope.

Done 2026-11-17: PR mgcizzu/nilssontronics#1 merged; tests + app run on Marco's Python (miniforge 3.13, `.venv`);
`MMCoreBackend` validated against `MMConfig_demo.cfg` (`tests/test_mmcore_demo.py`).
