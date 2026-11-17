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
- **Output = ISS_preprocessing input** (github.com/Moldia/ISS2025, `mode='tif_autosaved'`):
  each `cycle_NN/` holds `R01--Stage0000--Z00--C00.tif` planes plus
  `Metadata/R01.xml` (Leica-style TileScanInfo with TileIndex = Stage, µm positions,
  pixel size). Regions = 8-connected tile groups; C = cycle channel order. Agreed
  2026-11-17; keep all format knowledge inside `mosaic.py`.
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
python3 -m pytest                              # 106 tests, no display needed
python3 -m microscope_ui                       # simulator
python3 -m microscope_ui --backend mmcore      # pymmcore-plus (hardware section of config)
python3 -m microscope_ui.qt                    # Qt view (same flags; --no-ndv for plain image view)
```

## Open items (in order)
1. Qt port (`microscope_ui/qt/`), keeping controller/engine as is. Feature parity with
   the Tk view: cycle workflow, overview canvas (rectangle/polygon, drag "Move ROI set",
   mosaic underlay), guided "Refocus all", "Align selected F → stage", recipes, ROI set
   save/load, spiral preview (snaps at each position; Tk only moved the stage),
   flat-field option, configuration dialog, MMCore device docks. All setup device
   calls run on `DeviceWorker` (never the GUI thread), drained before the engine starts.
   Never use pymmcore-widgets StageWidget: it bypasses backend stage-limit checks.
   Next: Marco tries it at the microscope, then decide whether to retire the Tk view.
2. Run ISS_preprocessing's `align_and_stitch` (ashlar + Java) on our cycles. Verified
   2026-11-17 with the pipeline's own code: `deconvolve_and_mip(mode='tif_autosaved')` and
   `mipped_to_OME_tiffs` on 2 simulated cycles × 2 regions — units/pixel size/magnification
   read from our XML without warnings, tile positions exact, MIPs identical.
   `tests/test_pipeline_format.py` mirrors its parsing rules for CI.
3. Before any real-hardware run: calibrate turret positions, pixel sizes, camera
   orientation, stage limits and safe Z ranges at the microscope.

Done 2026-11-17: PR mgcizzu/nilssontronics#1 merged; tests + app run on Marco's Python (miniforge 3.13, `.venv`);
`MMCoreBackend` validated against `MMConfig_demo.cfg` (`tests/test_mmcore_demo.py`).
