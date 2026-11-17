# Microscope Scan Console

A workflow-focused desktop prototype for operating a microscope without exposing
Micro-Manager's device tree and property browser. Version 0.1 runs entirely on a
simulated microscope and sends **no hardware commands**.

## Try the prototype

Python 3.11 or newer with Tk support is the only requirement.

```bash
python3 -m microscope_ui
```

The prototype supports:

- any number of configuration-defined objectives, including immersion-change warnings;
- named channels with exposure sliders and illumination safety limits;
- independent X and Y tile overlap controls, both defaulting to 10%;
- live **Set Begin** / **Set End** capture for absolute Z-stack limits;
- automatic endpoint-inclusive Z plane counts derived from objective NA;
- rectangular and irregular-polygon ROIs drawn directly on the overview;
- reusable ROI-set files containing outlines and focus-point XYZ coordinates;
- reloadable, editable focus points captured from the current live XYZ position;
- drag-to-translate for an entire ROI/focus-point set while preserving focus Z values;
- single-channel, single-plane center-out spiral previews;
- guarded single-plane, Z-stack, and surface-following acquisition modes;
- configuration-defined acquisition recipes;
- estimates for tiles, frames, time, and storage;
- XY, Z, exposure, illumination, and disk-space preflight checks;
- incremental disk-backed scan mosaics with composite and per-channel display;
- a bounded asynchronous camera-to-SSD queue with safe backpressure;
- immediate per-plane flushing by a dedicated writer into one tile stack at a time;
- fixed-size memory-mapped preview maps and an append-only recovery index.

Use **Rectangle** to drag a rectangular ROI. In **Irregular polygon** mode, click
each corner and double-click or right-click to close the shape; Escape cancels an
unfinished outline. Use **Move ROI set** to drag every ROI and focus point together
in XY as one registration object.

To record focus, choose **Focus points** and click the sample to move there. Focus
the live image with the physical microscope control (or the simulator's Z jog
buttons), then press **Capture XYZ**. Click a reloaded numbered marker to navigate
back to its stored XYZ; after refocusing, **Update Z** replaces only that point's Z
coordinate. Surface following requires at least three points inside the selected
ROIs that do not all lie on one line.

The **Spiral preview** channel menu lists the currently enabled acquisition
channels. Choose one and press the button to acquire a center-out XY preview at
the current Z. The preview never runs a Z-stack. If ROIs exist it covers their tile
positions; otherwise it previews a 5×5 field around the current stage position.

The image viewer uses pointer-centred mouse-wheel zoom. Move the pointer over the
green **Live Z (wheel)** toolbar instead to jog focus by 1 µm per wheel step;
holding Shift changes that to 10 µm. The Overview button resets the image to 100%.

For a Z-stack, select **Z-stack**, focus the first limit in Live mode, and press
**Set Begin**. Move Z to the other limit and press **Set End**. The app calculates
the plane count and actual endpoint-inclusive spacing automatically. It estimates
the widefield axial Nyquist interval as `n × wavelength / NA²`, using the selected
objective's immersion refractive index and the shortest enabled channel emission
wavelength. Changing the objective clears captured limits; changing enabled
channels recalculates the plane count.

**Save ROI set** and **Load ROI set** preserve regions and focus points in a small,
human-readable JSON file using stage coordinates in micrometres. Select **Review
and start scan** to exercise the full preflight and confirmation flow.

## Large-scan storage and display

During a simulated full scan, the camera side places each channel/Z plane into a
strictly bounded queue and immediately proceeds when space remains. A dedicated
writer appends and flushes those frames to the current tile's `.simstack` file.
Only that one tile container remains open; completing it atomically renames the
file and appends a record to `tiles.jsonl`. A crash therefore leaves previously
completed tiles and their index intact. Real camera frames will use the same
lifecycle boundary with an OME-Zarr writer instead of the compact simulator
payload.

The default queue holds at most four frame payloads, configured by
`storage.writer_queue_frames` in `config/microscope.json`. At 2048×2048 and 16-bit,
that bounds writer-owned image payloads to 32 MiB. If the SSD falls behind long
enough to fill the queue, acquisition pauses safely until one slot is released;
it does not grow RAM use or drop a frame. The status line reports acquired and
saved frame counts, buffer occupancy, and any backpressure independently.

The on-screen mosaic is assembled from fixed-size memory-mapped RGB files under
`disk_preview/`. Its disk and RAM requirements do not grow with the physical scan
area. The viewer keeps only one screen-sized image while the writer owns only its
bounded frame queue, throttling redraws independently of camera timing. The current tile is
outlined; Z planes appear while acquired and are replaced by a MIP when that tile
stack completes. **Scan mosaic display** switches between the composite and any
enabled channel without interrupting acquisition. Pan/zoom continues to read from
the disk preview rather than retaining acquired tiles in RAM.

## Configuration

Edit [`config/microscope.json`](config/microscope.json) to define objectives,
channels, stage limits, camera size, overlap, and recipes. The supplied values are
examples, not measured values for the real microscope. In particular, turret
positions, pixel sizes, stage limits, and safe Z ranges must be calibrated before
hardware use.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Architecture and hardware integration

The interface uses the narrow `MicroscopeBackend` protocol in
`microscope_ui/backend.py`. `SimulatedMicroscope` implements it now. The next
hardware milestone is a `pymmcore-plus` implementation that:

1. loads a known-good Micro-Manager configuration;
2. maps objective and channel IDs to verified configuration presets;
3. performs safe-Z turret changes and waits for devices to settle;
4. exposes live and snapped images to the viewer;
5. runs a `useq-schema` acquisition and writes OME-Zarr or OME-TIFF;
6. reports device faults without exposing raw device properties to normal users.

No real-hardware backend should be enabled until the device mappings and travel
limits have been checked at the microscope by an operator.
