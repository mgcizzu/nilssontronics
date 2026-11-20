# Code review — 2026-11-19

Reviewed: `main` at `4eceb94` ("Harden real hardware safety boundaries").
Focus: acquisition path, hardware safety, Qt threading. All 162 tests pass
(6 skipped: Tk / Micro-Manager demo config not available in the review environment).

Work through the items in order. For each fix, add a regression test next to the
existing ones (`tests/test_cycles.py`, `tests/test_qt_*.py`) and tick the box.

## A. Fix before running on the microscope

- [x] **A1. Redundant Z move before every frame.**
  `microscope_ui/engine.py:298` compares the *read-back* Z with the target to
  1e-6 µm. Real focus drives report Z with noise, so it never matches and the
  engine re-issues `move_z` for every plane, even in single-plane and focus-map
  modes (and, with `shutter_per_frame=False`, also closes/reopens the shutter).
  Measured with ±0.03 µm read-back noise in the simulator: 60 frames → 60 Z moves.
  *Fix:* track the last commanded Z inside the engine and move only when the
  target differs from it (reset it after pause/resume and at cycle start).
  *Test:* simulator subclass with noisy `position.z_um`; assert one Z move per
  tile in single/focus-map mode.

- [x] **A2. Too many controller round trips per frame.**
  Per plane: `backend.position` in the engine (XY + Z read), `move_z` →
  `refresh_objective_id()` (turret read) + `setPosition` + wait + `self.position`
  (XY + Z read again). `move_xy` also returns a fresh position. On serial stage
  controllers (~5–20 ms per query) that is ~50–100 ms per frame — 1–2 h extra on
  a 2000 tiles × 4 channels × 10 planes cycle.
  *Fix:* check the turret once per tile (already done at engine.py:277) and pass
  a flag/use a separate method so `move_z` inside the engine doesn't re-read the
  turret; don't read the position after moves inside the engine (callers that need
  it can ask). Keep the turret re-read in `move_z` for interactive (device-worker) moves.
  *Test:* `FakeCore` call counter in `tests/test_cycles.py`.

- [x] **A3. Qt view reads hardware on the GUI thread.**
  `qt/window.py:857` (`_refresh`, runs on every input change) calls
  `controller.preflight()`, which in single-Z mode reads `self.backend.position`
  (`controller.py:187`) on the GUI thread — concurrently with the device worker
  (live, objective change). Breaks the DeviceWorker rule and can freeze the window.
  *Fix:* give `preflight()` an optional `current_z_um` argument; `_refresh` passes
  `self.device.last_position.z_um`. Only `start_cycle` (after `stop_and_drain`)
  reads the real position.
  *Test:* backend whose `position` raises if called from the GUI thread.

- [x] **A4. Distortion correction throttles acquisition.**
  `FrameCorrector.correct` runs on the single SSD-writer thread: ~0.28 s per
  2048×2048 frame in the review environment. The bounded queue fills, the camera
  waits (safe, but 3–10× slower) and the time estimate ignores it.
  *Fix (pick one):* cache the Keys weights per channel and/or use a faster remap
  (`scipy.ndimage.map_coordinates` or OpenCV `remap`, check licences — MIT app);
  run correction in a small thread pool between engine and writer; or correct after
  acquisition. Add the correction cost to `estimate_acquisition`.
  *Test:* benchmark-style test with a generous bound, plus estimate includes it.

- [x] **A5. Disk estimate ignores uncorrected copies.**
  With distortion correction on, `keep_uncorrected` defaults to `True` and doubles
  the data, but `estimate_acquisition` (`planner.py:121`) and the free-space check
  in `validate_plan` don't count it. A cycle can pass preflight and run out of disk.
  *Fix:* multiply `data_bytes` by 2 when `distortion_correction and keep_uncorrected`;
  show it in the summary.

## B. Cycle-workflow traps

- [x] **B1. Z-stack Begin/End silently carried into later cycles.**
  Only reset on objective change (`qt/window.py:1131` `_reset_z_stack_capture`).
  After strip/restain/remount, cycle 2 reuses cycle 1's absolute Z range; "Refocus
  all" doesn't touch it.
  *Fix:* reset Begin/End when a cycle finishes (and when an experiment is opened),
  or make preflight refuse a stack captured before the current cycle.

- [x] **B2. Opening an experiment can rotate the turret without confirmation.**
  `_load_experiment_into_view` (`qt/window.py:1690`) calls `_objective_selected`,
  which only asks when the immersion medium differs. Otherwise "Open…" moves Z to
  the objective-change clearance and rotates the turret unprompted, then offers to
  resume. (Same pattern for recipes at `qt/window.py:1086`.)
  *Fix:* always confirm hardware moves triggered by Open/recipe, or just select
  the objective in the box and let preflight say "put objective X in the light path".

- [x] **B3. "Set Z here" / "Accept Z ▶" can get stuck.**
  `_accept_pending` (`qt/window.py:1329`) is reset only in the success callback.
  If the position read fails, the flag stays `True` and the button does nothing
  for the rest of the session; Refocus all cannot continue.
  *Fix:* reset it in an `on_error` handler and in `stop_refocus()`
  (`with_fresh_position` needs an `on_error` parameter).

## C. Minor

- [x] **C1. Pause can't be aborted while the writer stalls.** `engine.py:244`
  waits for the writer without a timeout; Abort can't break it if the disk hangs.
  Wait in a loop with a timeout and check `self._abort`.
- [x] **C2. Resume skips most checks.** `controller.resume_cycle` doesn't re-check
  disk space or preset-fixed channel powers. Reuse the relevant parts of preflight.
- [x] **C3. `resume_info()` on every refresh.** `qt/window.py:927` → reads
  `tiles.jsonl` and stats every plane file whenever settings change while a cycle
  is unfinished. Cache per experiment/cycle and invalidate when a cycle runs.
- [x] **C4. fsync on macOS isn't a full flush.** Use `fcntl.F_FULLFSYNC` on Darwin
  in `mosaic.py` / `experiment.py` (Windows/Linux are fine as is).
- [ ] **C5. Tk view calls the backend on its GUI thread.** Fine for the simulator,
  blocks with real hardware; plus two views to keep in sync. Decide on retiring
  `app.py` after the Qt view has been tried at the microscope.
  **Assessment:** valid architectural debt, but intentionally deferred. The team
  has chosen Qt; remove the Tk implementation only after the Qt hardware trial,
  rather than combining a large deletion with acquisition-safety changes.

## D. Feature idea (not a bug)

- [x] **D1. Z-stack relative to the focus surface.** Stacks are absolute for every
  tile, so on a tilted section they must span the whole tilt. A "stack around the
  surface" mode (±N µm around the focus-map Z of each tile) would shorten stacks a
  lot on large sections. Needs Marco's OK (changes the Z-mode set).
  **Implemented:** approved by Marco. The `surface_stack` mode captures a stack
  below/above each tile's interpolated surface Z, always includes the surface
  plane, and preflights the complete per-tile range against safe Z limits.

## E. Second review — 2026-11-20 (`7d3c52d`..`75a193b`)

Reviewed the fixes for A–D (all confirmed, each with a regression test) and the
new commits: surface-relative Z-stacks, compact UI, Micro-Manager profile wizard.
185 of 186 tests pass in the review environment (see E7). Acquisition runs on
Windows, so the macOS `F_FULLFSYNC`-per-plane cost is deliberately not listed.

- [x] **E1. The profile wizard is not read-only and can fight the running app.**
  `inspect_mm_config` (`microscope_ui/mm_profile.py:33`) calls
  `core.loadSystemConfiguration` (line 60), which initializes every device and
  applies the `.cfg` System/Startup presets — depending on the adapters this can
  home the XY stage, move the turret/filter wheels or open shutters. If the app is
  already running on the MMCore backend, the wizard creates a *second* core on the
  same serial/USB ports: loading fails or conflicts, and `unloadAllDevices()` in
  `finally` can send shutdown commands to devices the app still uses. It also runs
  on the GUI thread (`SourcePage.validatePage`, `qt/profile_wizard.py:78`), so the
  window freezes while drivers load.
  *Fix:* refuse to start the wizard while `backend.simulated` is False (tell the
  user to switch to the simulator profile and relaunch); run inspection on a
  worker thread with a progress indicator; replace "The wizard only reads the
  configuration" with an explicit warning that devices are initialized as in
  Micro-Manager (stage may home, turret/filters may move) and ask for confirmation.
  *Test:* wizard refuses with a hardware backend; inspection runs off the GUI thread.

- [x] **E2. Wizard pre-fills plausible safety values.**
  Every objective row starts as 10×, NA 0.3, 1.0 µm/px, safe Z −4000/3500
  (`qt/profile_wizard.py:234`). These pass validation, so ticking the single
  confirmation box saves wrong pixel sizes and safe-Z ranges.
  *Fix:* leave magnification, NA, pixel size and safe-Z cells empty and refuse to
  continue until each row is filled (turret index and name can stay pre-filled).
  Also `immersion_refractive_index` is 1.515 for every non-air medium (line 281):
  use oil 1.515, water 1.333, silicone 1.406, and require a value for "other"
  (or add an editable column).
  *Test:* `selected_objectives()` rejects empty cells; water → 1.333.

- [x] **E3. Z spacing in the XML can be wrong; surface stacks are unevenly spaced.**
  Both stack modes shrink the spacing to fit the range, but the region XML records
  the step typed by the user (`controller.py:567`, `z_step_um=record.z.step_um`).
  Example: absolute stack 0→10 µm, step 3 → planes at 0, 2.5, 5, 7.5, 10, but the
  XML says step 3 and length 12 µm. Surface stacks compute the halves separately
  (`ZSettings.stack_positions`, `models.py:305`): −5/+3 µm with step 2 gives
  1.67 µm spacing below the surface and 1.5 µm above.
  *Fix:* for `surface_stack`, use offsets `k·step` for
  `k = floor(bottom/step) … ceil(top/step)` (uniform, always includes 0; may extend
  slightly past the requested span — show that in the summary). For both modes,
  pass the *actual* spacing (from the computed positions) to `CycleMetadata`.
  Optionally restore the NA-based automatic step in the Qt view (see E5).
  *Test:* uniform spacing for surface stacks; XML Z length = (n−1)·actual spacing.

- [x] **E4. One failed launch forgets the microscope profile.**
  When the saved profile fails at startup (e.g. microscope switched off),
  `qt/window.py:2134` removes `startup_profile`, so every later launch silently
  opens the simulator.
  *Fix:* fall back to the simulator for this launch only, keep the setting, and
  show which profile failed plus a "Retry" / "Use simulator from now on" choice.
  *Test:* failed startup keeps `startup_profile`.

- [x] **E5. README describes the NA-based Z step, but the Qt view doesn't use it.**
  `README.md:60` says the plane count comes from the axial Nyquist interval
  `n·λ/NA²`; only the Tk view computes it — Qt uses the typed step.
  *Fix:* either add a "Nyquist" button/default for the Qt step box
  (`optics.recommended_z_step_um`) or update the README.

- [x] **E6. Surface stack capture always centres the span on the surface.**
  `_apply_surface_stack_capture` (`qt/window.py:1309`) keeps only the captured
  span and centres it (±span/2). If focus points were taken at the top or bottom
  of the tissue, half the stack is wasted.
  *Fix:* keep the captured asymmetry: when Begin/End are captured at a position,
  compute below/above relative to the surface Z *at that XY*. Keep the symmetric
  option as a choice if wanted (decision for Marco).

- [x] **E7. Clipped button label at small window sizes.**
  `test_compact_button_labels_fit_at_small_window_size` (`tests/test_qt_view.py:278`)
  fails on Linux fonts: "Align selected F → stage" is 186 px wide but needs 199 px
  at 1024×720. Passes alone/on macOS, but Windows with 125–150 % display scaling
  is likely to clip too.
  *Fix:* shorter label ("Align F → stage", full text as tooltip) or let the row
  wrap; make the test independent of the platform font (check `sizeHint()`
  against the laid-out width rather than a fixed font).
