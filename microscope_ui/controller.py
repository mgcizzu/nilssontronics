"""GUI-independent scan controller.

Owns the backend, the current experiment and the running engine. The Tk view
(and the planned Qt view) only translate user input into an
:class:`~microscope_ui.models.AcquisitionPlan` and call these methods, so the
complete acquisition path is testable without a display.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock

from .acquisition import BoundedAsyncWriter
from .backend import MicroscopeBackend
from .config import DEFAULT_CONFIG_PATH
from .distortion import (
    EXPERIMENT_COPY,
    CalibrationError,
    CalibrationResult,
    CorrectingPlaneSink,
    DistortionCorrection,
    FrameCorrector,
    correction_path,
    fit_correction,
)
from .engine import AcquisitionEngine, CycleRun, EngineStatus
from .experiment import CycleRecord, Experiment
from .flatfield import estimate_flat_fields
from .focus import FocusSurface
from .models import AcquisitionEstimate, AcquisitionPlan, MicroscopeConfig, Tile, ZSettings
from .mosaic import CycleMetadata, DiskBackedMosaic, completed_tile_indices, mosaic_bounds_for_tiles
from .planner import disk_space_problem, estimate_acquisition, tiles_for_plan, validate_plan


class ControllerError(RuntimeError):
    pass


def calibration_snapshot(config: MicroscopeConfig, objective_id: str) -> dict:
    """What fixes the geometry of saved images; stored with the experiment."""

    objective = config.objective(objective_id)
    orientation = config.camera_orientation
    return {
        "objective_id": objective.id,
        "pixel_size_um": objective.pixel_size_um,
        "magnification": objective.magnification,
        "camera_width_px": config.camera_width_px,
        "camera_height_px": config.camera_height_px,
        "camera_orientation": {
            "flip_x": orientation.flip_x,
            "flip_y": orientation.flip_y,
            "transpose": orientation.transpose,
        },
    }


def calibration_differences(recorded: dict, current: dict) -> list[str]:
    """Human-readable differences between two calibration snapshots."""

    labels = {
        "objective_id": "objective",
        "pixel_size_um": "pixel size (µm/px)",
        "magnification": "magnification",
        "camera_width_px": "camera width (px)",
        "camera_height_px": "camera height (px)",
        "camera_orientation": "camera orientation",
    }
    differences = []
    for key, label in labels.items():
        old, new = recorded.get(key), current.get(key)
        if isinstance(old, float) or isinstance(new, float):
            same = old is not None and new is not None and abs(float(old) - float(new)) <= 1e-9 * max(
                1.0, abs(float(old))
            )
        else:
            same = old == new
        if not same:
            differences.append(f"{label}: {old} when cycle 1 started, now {new}")
    return differences


class ScanController:
    def __init__(self, config: MicroscopeConfig, backend: MicroscopeBackend) -> None:
        self.config = config
        self.backend = backend
        self.experiment: Experiment | None = None
        self.engine: AcquisitionEngine | None = None
        self.mosaic: DiskBackedMosaic | None = None
        self.active_cycle: int | None = None
        self.active_tiles: tuple[Tile, ...] = ()
        self._experiment_lock = Lock()
        self._resume_cache: tuple[CycleRecord, int, int] | None = None
        self._resume_cache_valid = False

    def _invalidate_resume_cache(self) -> None:
        self._resume_cache = None
        self._resume_cache_valid = False

    # -- state ---------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.engine is not None and not self.engine.status().finished

    def status(self) -> EngineStatus | None:
        return None if self.engine is None else self.engine.status()

    # -- planning ------------------------------------------------------------
    def tiles_for(self, plan: AcquisitionPlan) -> list[Tile]:
        """Tile list for the next cycle: fixed by the experiment once it exists."""

        if self.experiment is None:
            return tiles_for_plan(self.config, plan)
        dx, dy = self.experiment.offset_of(plan.regions)
        return [tile.translated(dx, dy) for tile in self.experiment.tiles]

    def compatibility_problems(self, plan: AcquisitionPlan) -> list[str]:
        experiment = self.experiment
        if experiment is None:
            return []
        problems = self.calibration_problems()
        if plan.objective_id != experiment.objective_id:
            problems.append("The objective is fixed for this experiment.")
        if (plan.overlap_x_percent, plan.overlap_y_percent) != (
            experiment.overlap_x_percent,
            experiment.overlap_y_percent,
        ):
            problems.append("Tile overlap is fixed for this experiment.")
        if plan.options.tile_order != experiment.options.tile_order:
            problems.append("Tile order is fixed for this experiment.")
        try:
            experiment.offset_of(plan.regions)
        except ValueError as exc:
            problems.append(str(exc))
        unfinished = experiment.resumable_cycle()
        if unfinished is not None:
            problems.append(
                f"Cycle {unfinished.number} is unfinished — resume it before starting a new cycle."
            )
        return problems

    def calibration_problems(self) -> list[str]:
        """Refuse to add images to an experiment whose calibration has changed."""

        experiment = self.experiment
        if experiment is None:
            return []
        if not experiment.calibration:
            return [
                "This experiment has no calibration record (created by an older version), so "
                "its pixel size and camera geometry cannot be verified. Start a new experiment."
            ]
        differences = calibration_differences(
            experiment.calibration, calibration_snapshot(self.config, experiment.objective_id)
        )
        if not differences:
            return []
        return [
            "The configuration no longer matches this experiment's images ("
            + "; ".join(differences)
            + "). Restore the original calibration or start a new experiment."
        ]

    def estimate(self, plan: AcquisitionPlan) -> tuple[list[Tile], AcquisitionEstimate]:
        tiles = self.tiles_for(plan)
        return tiles, estimate_acquisition(self.config, plan, tiles)

    def preflight(
        self,
        plan: AcquisitionPlan,
        check_disk: bool = True,
        current_z_um: float | None = None,
    ) -> list[str]:
        problems = self.compatibility_problems(plan)
        if any("ROI set changed" in item for item in problems):
            return problems
        tiles = self.tiles_for(plan)
        problems += validate_plan(self.config, plan, tiles, check_disk=check_disk)
        if self.backend.current_objective_id != plan.objective_id:
            problems.append("The selected objective is not in the light path.")
        for setting in plan.channels:
            try:
                channel = self.config.channel(setting.channel_id)
            except KeyError:
                continue  # validate_plan already reports the unknown channel
            if (
                not self.backend.power_is_adjustable(setting.channel_id)
                and abs(setting.power_percent - channel.default_power_percent) > 1e-6
            ):
                problems.append(
                    f"{channel.name} power is fixed by its Micro-Manager preset at "
                    f"{channel.default_power_percent:g}%."
                )
        if plan.z.mode == "single":
            low, high = self.config.safe_z_range(plan.objective_id)
            z = self.backend.position.z_um if current_z_um is None else current_z_um
            if not low <= z <= high:
                problems.append(
                    f"Current Z {z:.1f} µm is outside the safe range {low:g}–{high:g} µm."
                )
        if self.busy:
            problems.append("An acquisition is already running.")
        return problems + self.distortion_problems(plan)

    def z_positions(
        self,
        z: ZSettings,
        focus_points,
        focus_method: str,
        tiles: tuple[Tile, ...],
        single_z_um: float | None,
    ) -> dict[int, tuple[float, ...]]:
        if z.mode == "stack":
            stack = z.stack_positions()
            return {tile.index: stack for tile in tiles}
        if z.mode == "focus_map":
            surface = FocusSurface(focus_points, focus_method)
            return {tile.index: (surface.z_at(tile.x_um, tile.y_um),) for tile in tiles}
        if z.mode == "surface_stack":
            surface = FocusSurface(focus_points, focus_method)
            offsets = z.stack_positions()
            return {
                tile.index: tuple(
                    surface.z_at(tile.x_um, tile.y_um) + offset for offset in offsets
                )
                for tile in tiles
            }
        if single_z_um is None:
            raise ControllerError("Single-plane acquisition needs a Z position")
        return {tile.index: (single_z_um,) for tile in tiles}

    # -- distortion correction (optional) ---------------------------------------
    @property
    def calibration_dir(self) -> Path:
        return self.config.calibration_dir or DEFAULT_CONFIG_PATH.parent / "calibration"

    def correction_file(self, objective_id: str) -> Path:
        return correction_path(self.calibration_dir, objective_id)

    def load_correction(self, objective_id: str) -> DistortionCorrection | None:
        """The current calibration for an objective (None if it was never calibrated)."""

        path = self.correction_file(objective_id)
        if not path.exists():
            return None
        return DistortionCorrection.load(path)

    def save_correction(self, correction: DistortionCorrection) -> Path:
        """Store a new calibration. Running experiments keep their own copy."""

        self.config.objective(correction.objective_id)
        return correction.save(self.correction_file(correction.objective_id))

    def fit_distortion(
        self,
        images,
        objective_id: str,
        grid_pitch_um: float,
        reference_channel: str | None = None,
    ) -> CalibrationResult:
        """Fit a correction from grid images (one per channel, same stage XY).

        The nominal pixel size is the configured field of view over the frame
        width, so it also holds for binned or simulated frames.
        """

        if not images:
            raise CalibrationError("Snap the grid in at least one channel first")
        fov_w, _fov_h = self.config.fov_um(objective_id)
        nominal = fov_w / images[0].image.shape[1]
        return fit_correction(images, objective_id, grid_pitch_um, nominal, reference_channel)

    def read_correction_for_import(self, path: Path | str) -> tuple[DistortionCorrection, list[str]]:
        """Load a correction file made elsewhere and list what may not fit this microscope.

        Raises ValueError for files that cannot be used at all (not a correction file,
        edited by hand, or for an objective this configuration does not have).
        Nothing is saved; call :meth:`save_correction` to adopt it.
        """

        correction = DistortionCorrection.load(path)
        try:
            objective = self.config.objective(correction.objective_id)
        except KeyError:
            raise ValueError(
                f"The file is for objective '{correction.objective_id}', which this configuration "
                "does not have"
            ) from None
        warnings = []
        unknown = sorted(item for item in correction.channels if item not in {c.id for c in self.config.channels})
        if unknown:
            warnings.append(f"Channels not in this configuration (ignored): {', '.join(unknown)}")
        size = (correction.frame_width_px, correction.frame_height_px)
        camera = self.config.oriented_camera_size_px
        if size != camera:
            warnings.append(
                f"Made for {size[0]}×{size[1]} px frames; the configured camera is {camera[0]}×{camera[1]} px. "
                "Correction only works with frames of the calibrated size (same binning)."
            )
        if correction.nominal_pixel_size_um and size[0]:
            calibrated_fov = correction.nominal_pixel_size_um * size[0]
            fov = self.config.fov_um(objective.id)[0]
            if abs(calibrated_fov / fov - 1) > 0.01:
                warnings.append(
                    f"Calibrated with a {calibrated_fov:.1f} µm field of view; {objective.name} is "
                    f"configured for {fov:.1f} µm. Was it made on another microscope?"
                )
        return correction, warnings

    def experiment_correction(self, experiment: Experiment) -> DistortionCorrection:
        """The correction an experiment uses: its own copy, made when cycle 1 started."""

        recorded = experiment.calibration.get("distortion")
        if not recorded:
            raise ValueError("This experiment has no distortion correction record")
        correction = DistortionCorrection.load(experiment.root / recorded["file"])
        if correction.fingerprint != recorded.get("fingerprint"):
            raise ValueError(
                f"{recorded['file']} in the experiment folder no longer matches the correction "
                "used for cycle 1; restore it before adding images"
            )
        return correction

    def distortion_problems(self, plan: AcquisitionPlan) -> list[str]:
        experiment = self.experiment
        wanted = plan.options.distortion_correction
        if experiment is not None and wanted != experiment.options.distortion_correction:
            state = "on" if experiment.options.distortion_correction else "off"
            return [f"Distortion correction is fixed {state} for this experiment (set by cycle 1)."]
        if not wanted:
            return []
        objective = self.config.objective(plan.objective_id)
        try:
            if experiment is not None:
                correction = self.experiment_correction(experiment)
            else:
                correction = self.load_correction(plan.objective_id)
        except (OSError, ValueError, KeyError) as exc:
            return [f"Distortion correction: {exc}"]
        if correction is None:
            return [
                f"No distortion calibration for {objective.name}: calibrate it with a grid slide "
                "first, or turn distortion correction off."
            ]
        if correction.objective_id != plan.objective_id:
            return [f"The distortion calibration belongs to objective {correction.objective_id}."]
        missing = correction.missing_channels(item.channel_id for item in plan.channels)
        if missing:
            names = ", ".join(self.config.channel(item).name for item in missing)
            if experiment is None:
                return [f"The distortion calibration does not cover {names}: calibrate every channel you will use."]
            return [
                f"This experiment's distortion calibration (fixed by cycle 1) does not cover {names}, "
                "so those channels cannot be corrected."
            ]
        return []

    # -- experiments ---------------------------------------------------------
    def new_experiment(self) -> None:
        if self.busy:
            raise ControllerError("Stop the running acquisition first")
        self._close_cycle_storage()
        self.experiment = None
        self._invalidate_resume_cache()

    def open_experiment(self, path: Path | str) -> Experiment:
        if self.busy:
            raise ControllerError("Stop the running acquisition first")
        experiment = Experiment.load(path)
        self._close_cycle_storage()
        self.experiment = experiment
        self._invalidate_resume_cache()
        return experiment

    def start_cycle(self, plan: AcquisitionPlan) -> CycleRecord:
        problems = self.preflight(plan)
        if problems:
            raise ControllerError("\n".join(problems))
        single_z = self.backend.position.z_um if plan.z.mode == "single" else None
        if self.experiment is None:
            tiles = tuple(tiles_for_plan(self.config, plan))
            calibration = calibration_snapshot(self.config, plan.objective_id)
            correction = None
            if plan.options.distortion_correction:
                correction = self.load_correction(plan.objective_id)
                if correction is None:
                    raise ControllerError("The distortion calibration disappeared; calibrate again")
                calibration["distortion"] = {
                    "file": EXPERIMENT_COPY,
                    "fingerprint": correction.fingerprint,
                    "calibrated_at": correction.created_at,
                }
            self.experiment = Experiment.create(
                plan.output_root,
                plan.project,
                plan.sample,
                plan.objective_id,
                plan.overlap_x_percent,
                plan.overlap_y_percent,
                plan.regions,
                tiles,
                plan.options,
                plan.focus_points,
                calibration=calibration,
            )
            if correction is not None:
                correction.save(self.experiment.root / EXPERIMENT_COPY)
            dx = dy = 0.0
        else:
            dx, dy = self.experiment.offset_of(plan.regions)
        record = CycleRecord(
            number=self.experiment.next_cycle_number,
            status="running",
            channels=plan.channels,
            z=plan.z,
            focus_points=plan.focus_points,
            offset_x_um=dx,
            offset_y_um=dy,
            single_z_um=single_z,
            started_at=_now(),
            options=plan.options,
        )
        with self._experiment_lock:
            self.experiment.add_cycle(record)
        self._invalidate_resume_cache()
        self._launch(record, skip=frozenset())
        return record

    def resume_cycle(self) -> CycleRecord:
        if self.experiment is None:
            raise ControllerError("Open an experiment first")
        if self.busy:
            raise ControllerError("An acquisition is already running")
        record = self.experiment.resumable_cycle()
        if record is None:
            raise ControllerError("Every cycle in this experiment is complete")
        if self.backend.current_objective_id != self.experiment.objective_id:
            raise ControllerError("Put the experiment's objective in the light path first")
        problems = self.calibration_problems()
        problems += self._resume_preflight(record)
        if problems:
            raise ControllerError("\n".join(problems))
        skip = frozenset(completed_tile_indices(self.experiment.cycle_folder(record.number)))
        with self._experiment_lock:
            record = self.experiment.update_cycle(record.number, status="running", error=None)
        self._invalidate_resume_cache()
        self._launch(record, skip=skip)
        return record

    def _resume_preflight(self, record: CycleRecord) -> list[str]:
        """Checks that can become invalid between an interrupted run and resume."""

        experiment = self.experiment
        assert experiment is not None
        problems: list[str] = []
        for setting in record.channels:
            try:
                channel = self.config.channel(setting.channel_id)
            except KeyError:
                problems.append(f"Unknown channel: {setting.channel_id}.")
                continue
            if not 0 < setting.exposure_ms <= channel.max_exposure_ms:
                problems.append(
                    f"{channel.name} exposure must be between 0 and "
                    f"{channel.max_exposure_ms:g} ms."
                )
            if not 0 <= setting.power_percent <= channel.max_power_percent:
                problems.append(
                    f"{channel.name} power must be between 0 and "
                    f"{channel.max_power_percent:g}%."
                )
            if (
                not self.backend.power_is_adjustable(setting.channel_id)
                and abs(setting.power_percent - channel.default_power_percent) > 1e-6
            ):
                problems.append(
                    f"{channel.name} power is fixed by its Micro-Manager preset at "
                    f"{channel.default_power_percent:g}%."
                )
        completed = completed_tile_indices(experiment.cycle_folder(record.number))
        remaining = tuple(
            tile for tile in experiment.tiles_for_cycle(record.number) if tile.index not in completed
        )
        options = record.options or experiment.options
        plan = AcquisitionPlan(
            project=experiment.project,
            sample=experiment.sample,
            output_root=experiment.root,
            objective_id=experiment.objective_id,
            channels=record.channels,
            regions=experiment.regions_with_offset(record.offset_x_um, record.offset_y_um),
            focus_points=record.focus_points,
            z=record.z,
            overlap_x_percent=experiment.overlap_x_percent,
            overlap_y_percent=experiment.overlap_y_percent,
            options=options,
        )
        if remaining:
            disk_problem = disk_space_problem(self.config, plan, remaining)
            if disk_problem:
                problems.append(disk_problem)
        return problems

    def resume_info(self) -> tuple[CycleRecord, int, int] | None:
        """(cycle, tiles already complete, total tiles) for an unfinished cycle."""

        if self.experiment is None or self.busy:
            return None
        if self._resume_cache_valid:
            return self._resume_cache
        record = self.experiment.resumable_cycle()
        if record is None:
            self._resume_cache_valid = True
            return None
        done = completed_tile_indices(self.experiment.cycle_folder(record.number))
        self._resume_cache = record, len(done), len(self.experiment.tiles)
        self._resume_cache_valid = True
        return self._resume_cache

    def _launch(self, record: CycleRecord, skip: frozenset[int]) -> None:
        experiment = self.experiment
        assert experiment is not None
        options = record.options or experiment.options
        tiles = experiment.tiles_for_cycle(record.number)
        z_positions = self.z_positions(
            record.z, record.focus_points, options.focus_method, tiles, record.single_z_um
        )
        safe_low, safe_high = self.config.safe_z_range(experiment.objective_id)
        unsafe = [
            index for index, values in z_positions.items()
            if any(not safe_low <= value <= safe_high for value in values)
        ]
        if unsafe:
            with self._experiment_lock:
                experiment.update_cycle(record.number, status="failed", error="Unsafe Z positions")
            raise ControllerError(f"{len(unsafe)} tiles would need Z outside the safe range")

        channels = tuple(self.config.channel(item.channel_id) for item in record.channels)
        fov_w, fov_h = self.config.fov_um(experiment.objective_id)
        corrector = None
        if options.distortion_correction:
            try:
                correction = self.experiment_correction(experiment)
                corrector = FrameCorrector(correction, [c.id for c in channels], fov_w, fov_h)
            except (OSError, ValueError) as exc:
                with self._experiment_lock:
                    experiment.update_cycle(record.number, status="failed", error=f"Distortion correction: {exc}")
                raise ControllerError(f"Distortion correction: {exc}") from exc
        self._close_cycle_storage()
        folder = experiment.cycle_folder(record.number)
        z_planes = max((len(values) for values in z_positions.values()), default=1)
        self.mosaic = DiskBackedMosaic(
            folder,
            mosaic_bounds_for_tiles(list(tiles), fov_w, fov_h),
            channels,
            fsync=self.config.fsync_tiles,
            tiles=tiles,
            metadata=CycleMetadata(
                cycle=record.number,
                objective=self.config.objective(experiment.objective_id),
                channel_settings=tuple(record.channels),
                camera_width_px=self.config.oriented_camera_size_px[0],
                camera_height_px=self.config.oriented_camera_size_px[1],
                z_planes=z_planes,
                z_step_um=(
                    record.z.step_um
                    if record.z.mode in ("stack", "surface_stack")
                    else None
                ),
                project=experiment.project,
                sample=experiment.sample,
                distortion_correction=corrector.correction.fingerprint if corrector else None,
            ),
        )
        sink = self.mosaic
        if corrector is not None:
            sink = CorrectingPlaneSink(self.mosaic, corrector, keep_uncorrected=options.keep_uncorrected)
        writer = BoundedAsyncWriter(sink, queue_capacity=self.config.writer_queue_frames)
        run = CycleRun(
            objective_id=experiment.objective_id,
            tiles=tiles,
            channels=record.channels,
            z_positions=z_positions,
            skip_tiles=skip,
            consistent_approach=options.consistent_approach,
            shutter_per_frame=options.shutter_per_frame,
        )
        post_process = None
        if options.flat_field:
            channel_ids = tuple(item.channel_id for item in record.channels)

            def post(should_stop, folder=folder, channel_ids=channel_ids):
                result = estimate_flat_fields(folder, channel_ids, should_stop=should_stop)
                if result is None:
                    return "Cycle complete • flat-field skipped (too few tiles)"
                with self._experiment_lock:
                    experiment.update_cycle(record.number, flat_field=str(result.relative_to(experiment.root)))
                return "Cycle complete • flat-field profiles saved"

            post_process = post

        self.active_cycle = record.number
        self.active_tiles = tiles
        self.engine = AcquisitionEngine(
            self.config,
            self.backend,
            writer,
            run,
            post_process=post_process,
            # Bound to this experiment: the record must land in the experiment the
            # cycle belongs to, whatever is open by the time it finishes.
            on_finished=lambda status, number=record.number, experiment=experiment: (
                self._cycle_finished(experiment, number, status)
            ),
        )
        self.engine.start()

    def _cycle_finished(self, experiment: Experiment, number: int, status: EngineStatus) -> None:
        state = {"complete": "complete", "aborted": "aborted"}.get(status.state, "failed")
        with self._experiment_lock:
            experiment.update_cycle(
                number,
                status=state,
                finished_at=_now(),
                error=status.error,
                tiles_completed=status.tiles_done,
            )
        self._invalidate_resume_cache()

    # -- control ---------------------------------------------------------------
    def pause(self) -> None:
        if self.engine is not None:
            self.engine.pause()

    def resume(self) -> None:
        if self.engine is not None:
            self.engine.resume()

    def abort(self) -> None:
        if self.engine is not None:
            self.engine.abort()

    def wait(self, timeout: float | None = None) -> bool:
        return True if self.engine is None else self.engine.join(timeout)

    def _close_cycle_storage(self) -> None:
        if self.mosaic is not None:
            self.mosaic.close()
            self.mosaic = None

    def close(self, timeout: float = 30.0) -> bool:
        """Stop any acquisition, then release storage and hardware.

        Returns False, and touches nothing, while the engine thread (and so its
        SSD writer) is still running: closing storage under an in-flight write
        would corrupt it. The caller should keep the program open and retry.
        """

        if self.engine is not None and not self.engine.status().finished:
            self.engine.abort()
        if self.engine is not None and not self.engine.join(timeout):
            return False
        self._close_cycle_storage()
        self.backend.close()
        return True


def _now() -> str:
    return datetime.now().astimezone().isoformat()
