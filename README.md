# Microscope Scan Console

A workflow-focused desktop front end for **cyclic imaging of large tissue
sections** on a Micro-Manager microscope — a deliberately simplified,
workflow-driven experience that hides Micro-Manager's device tree and property
browser. It runs on a built-in simulator by default; a pymmcore-plus backend
drives real hardware.

## Try it

Python 3.11+ with Tk and NumPy.

```bash
pip install -e .                 # simulator only
pip install -e ".[hardware]"     # + pymmcore-plus for a real microscope
pip install -e ".[flatfield]"    # + BaSiC (basicpy) for flat-field estimation
pip install -e ".[qt]"           # + Qt view (PyQt6, ndv, pymmcore-widgets)

python3 -m microscope_ui                       # simulator
python3 -m microscope_ui --backend mmcore      # Micro-Manager (see below)
python3 -m microscope_ui --config my_scope.json
python3 -m microscope_ui.qt                    # Qt view (work in progress), same flags
```

## Cyclic imaging workflow

An **experiment** fixes the objective, ROI set, overlap and tile order once.
That produces one tile list (index, grid row/column, stage XY, acquisition
order) which **every cycle replays exactly**. Channels, exposures and
illumination power are chosen freely per cycle (decoding happens downstream,
so no marker mapping is stored).

1. **Cycle 1.** Choose objective and overlap, draw rectangular or polygon ROIs,
   capture focus points, pick channels, then **Review and start cycle 1**. This
   creates the experiment folder. From now on the objective, overlap, tile
   order and ROI shapes are locked.
2. **Strip / restain and remount the slide.**
3. **Re-align by translation.** Either drag the whole set in **Move ROI set**
   mode, or – more precisely – select a focus point, drive the stage until the
   same landmark is centred in Live, and press **Align selected F → stage**.
   The offset relative to cycle 1 is shown under the viewer and stored with
   the cycle.
4. **Refocus.** Press **Refocus all**: the stage visits F1, F2, … in turn; focus
   in Live and press **Accept Z ▶** for each (Z only is updated). Single focus
   points can still be clicked and updated with **Update Z**.
5. **Review and start cycle N.** Repeat from step 2.

**Pause** stops between tiles (after the current tile is saved). **Abort**
stops after the current frame; the unfinished tile is discarded.
After an abort, error, power cut or crash, **Open…** the experiment (or keep
it open) and **Resume cycle**: only tiles missing from that cycle's journal
are acquired, with the cycle's original channels, powers and focus positions.
A new cycle cannot start while an earlier one is unfinished.

## Z modes

- **Single** – the Z at the moment the cycle starts (checked against the
  objective's safe range).
- **Z-stack** – absolute Begin/End captured from live Z; plane count from the
  widefield axial Nyquist interval `n·λ/NA²` (shortest enabled emission).
- **Surface map** – one plane per tile at the Z of a focus surface:
  - *Triangulated* (default): piecewise-linear over a Delaunay triangulation,
    so local warping of large sections is followed; outside the points' hull
    Z is held at the nearest hull edge (never extrapolated).
  - *Plane fit*: least-squares plane, robust for noisy points on flat sections.

  Every tile's surface Z is checked against the safe range before scanning.

## Options (all optional)

- **Tile order:** *Raster* (default; every row left→right, the same X approach
  direction) or *Serpentine* (shorter travel). Fixed per experiment.
- **Close shutter during moves** (default on): light only during exposures.
  Off keeps the shutter open per tile/channel for speed, still closing it for
  stage and Z moves.
- **Consistent stage approach:** every XY target is approached from the same
  side after an overshoot of `stage.approach_offset_um` (set it in the config).
- **Flat-field profile after each cycle:** writes `cycle_NN/flatfield/<ch>.npy`
  (mean 1, `corrected = raw / profile`); raw data are never modified. Uses
  BaSiC when `basicpy` is installed, otherwise a median-based estimate.

## Tile grid

All ROIs share **one grid** anchored on their combined bounding box, so
overlapping ROIs never produce duplicate tiles and the requested overlap is
exact everywhere. A grid cell is kept when its field of view overlaps any ROI.
Tile centres (and approach points) are checked against the stage limits.

## Data layout

```
<output>/<project>/<sample>_<timestamp>/
  experiment.json          objective, ROIs, tile list, every cycle's settings,
                           offset, focus points, status (atomic + fsync)
  cycle_01/
    tiles/tile_00000.tstack   one finished tile (all channels and Z planes)
    tiles.jsonl               append-only journal of finished tiles (resume)
    disk_preview/*.u16        fixed-size memory-mapped preview per channel
    flatfield/                optional
  cycle_02/ …
```

`.tstack` is an **interim** container (JSON header + raw planes, readable
with `microscope_ui.mosaic.read_tile_stack`) until the file names and format
required by the downstream pipeline are agreed; only `mosaic.py` knows it.

Each tile is written to `.tmp`, fsynced, atomically renamed, then appended to
`tiles.jsonl` (also fsynced). Every tile in the journal is therefore complete
on disk after a crash. Camera frames pass through a bounded queue
(`storage.writer_queue_frames`) to one writer thread: if the SSD falls behind,
the camera waits; frames are never dropped and RAM never grows. The preview is
a fixed-size map regardless of scan area; the current tile shows a running MIP.

## Architecture

```
app.py            Tk view
qt/               Qt view (in progress): overview canvas, ndv live view on a device-worker
                  thread, pymmcore-widgets device docks
controller.py     ScanController: experiments, preflight, start/pause/abort/resume
engine.py         AcquisitionEngine: one cycle on a worker thread
acquisition.py    BoundedAsyncWriter: camera → SSD queue
mosaic.py         tile storage, journal, preview maps
experiment.py     Experiment → Cycle model (experiment.json)
planner.py        tile grid, estimates, validation
focus.py          focus surfaces
backend.py        backend protocol, limit checks, simulator
backend_mmcore.py pymmcore-plus backend
flatfield.py      optional flat-field estimation
```

The interface never talks to devices; it builds an `AcquisitionPlan` and calls
the controller, so the whole acquisition path is tested without a display.

## Hardware (pymmcore-plus)

Set `"backend": "mmcore"` (or pass `--backend mmcore`) and fill the
`hardware` section of the configuration:

| key | meaning |
| --- | --- |
| `mm_config_path` | Micro-Manager `.cfg` file |
| `mm_adapter_path` | Micro-Manager install folder (optional; default discovery otherwise) |
| `channel_group`, `channel_presets` | config group and one preset per channel id |
| `objective_device`, `turret_position_offset` | turret state device; MM state = `turret_position + offset` |
| `objective_change_z_um` | clearance Z used before rotating the turret |
| `power_properties` | optional per channel: `{"device", "property", "scale", "offset"}` |
| `pixel_size_configs` | optional per objective: MM pixel-size preset to select |
| `settle_timeout_ms` | device wait timeout |

`camera.orientation` (`flip_x`, `flip_y`, `transpose`) maps camera pixels to
stage axes; set it so that a stage move to +X moves image content to −column.

Safety rules in the backend: XY/Z targets outside the configured limits (and
the objective's safe Z range) are **refused before any command is sent**,
never clamped; objective changes move to the clearance Z first and leave Z
there for a deliberate refocus; device faults are reported as plain messages.

The shipped `hardware` values match Micro-Manager's demo configuration
(`MMConfig_demo.cfg`). **No real-hardware run should start until turret
positions, pixel sizes, orientation, stage limits and safe Z ranges have been
calibrated at the microscope by an operator.**

## Configuration

`config/microscope.json` defines objectives, channels, stage limits, camera
size and orientation, overlap, timing used for estimates, recipes and the
hardware mapping. The file is validated on start-up and every problem is
listed (unique IDs, NA ≤ n, safe Z inside stage limits, default ≤ max
exposure/power, recipe references, …).

## Tests

```bash
python3 -m pytest           # or: python3 -m unittest discover -s tests
```

`tests/test_mmcore_demo.py` runs `MMCoreBackend` against a real `CMMCorePlus`
loaded with `MMConfig_demo.cfg`; it is skipped unless Micro-Manager is found
(`mmcore install`, or `MICROMANAGER_PATH`). On Apple silicon, `mmcore install`
fetches only the arm64 test adapters (DemoCamera etc.), which is all this needs;
an x86_64 `/Applications/Micro-Manager*` build cannot be loaded by arm64 Python.
`tests/test_qt_view.py` needs `.[qt]` and runs headless (`QT_QPA_PLATFORM=offscreen`)
with a plain image view; ndv/vispy needs a real OpenGL display.
