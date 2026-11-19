"""Optional distortion correction: grid calibration, resampling and the cycle path."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import tifffile

from microscope_ui import distortion as D
from microscope_ui.backend import SimulatedMicroscope
from microscope_ui.config import load_config
from microscope_ui.controller import ScanController
from microscope_ui.models import AcquisitionOptions, AcquisitionPlan, ChannelSetting, ScanRegion
from microscope_ui.mosaic import tile_records

SLIDE = {"x_um": 30000, "y_um": 0, "size_um": 8000, "pitch_um": 50, "line_width_um": 5}
OPTICS = {"k1": 0.02, "chromatic_scale_per_nm": 2e-5, "channel_shift_px": {"cy5": [1.3, -0.8]}}
CHANNELS = ("dapi", "af488", "cy5")


@pytest.fixture(scope="module")
def config():
    return load_config()


def grid_images(config, optics, frame_px=512, channels=CHANNELS, at=(30007.0, 13.0)):
    sim = SimulatedMicroscope(config, frame_px=frame_px, distortion=optics, calibration_slide=SLIDE)
    sim.change_objective(config.objective("20x_air"))
    sim.move_xy(*at)
    images = []
    for channel_id in channels:
        sim.set_channel(config.channel(channel_id), 50, 20)
        images.append(D.GridImage(channel_id, sim.snap(), at))
    return images


def nominal_pixel(config, frame_px):
    return config.fov_um("20x_air")[0] / frame_px


def matched_offsets(a: np.ndarray, b: np.ndarray, pitch_px: float) -> np.ndarray:
    """Distances from each point in ``a`` to its nearest point in ``b`` (matched pairs only)."""

    distance = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]).min(axis=1)
    return distance[distance < pitch_px / 3]


# -- calibration maths ---------------------------------------------------------------
def test_fit_recovers_distortion_and_channel_shift(config):
    result = D.fit_correction(grid_images(config, OPTICS), "20x_air", 50.0, nominal_pixel(config, 512))
    correction = result.correction
    assert correction.reference_channel == "dapi"
    for channel_id in CHANNELS:
        model = correction.channels[channel_id]
        assert model.rms_px < 0.1
        assert model.points > 100
        assert 6.0 < correction.max_distortion_px(channel_id) < 8.0  # simulated ~7 px at the corners
    cy5 = correction.channels["cy5"]
    assert cy5.shift_x_px == pytest.approx(1.3, abs=0.1)
    assert cy5.shift_y_px == pytest.approx(-0.8, abs=0.1)
    # Longer wavelength → slightly higher simulated magnification → smaller true pixel.
    assert cy5.pixel_size_um < correction.channels["dapi"].pixel_size_um


def test_corrected_frames_match_an_ideal_microscope(config):
    pixel = nominal_pixel(config, 512)
    distorted = grid_images(config, OPTICS)
    ideal = {item.channel_id: item.image for item in grid_images(config, {})}
    correction = D.fit_correction(distorted, "20x_air", 50.0, pixel).correction
    corrector = D.FrameCorrector(correction, CHANNELS, 512 * pixel, 512 * pixel)
    pitch_px = 50.0 / pixel
    reference_points, _ = D.detect_grid_points(ideal["dapi"], pitch_px, -1)
    for item in distorted:
        corrected = corrector.correct(item.channel_id, item.image)
        assert corrected.dtype == item.image.dtype and corrected.shape == item.image.shape
        before = matched_offsets(D.detect_grid_points(item.image, pitch_px, -1)[0], reference_points, pitch_px)
        after = matched_offsets(D.detect_grid_points(corrected, pitch_px, -1)[0], reference_points, pitch_px)
        assert np.max(before) > 3.0
        # Every channel lands on the same, undistorted grid (reference = the ideal DAPI image).
        assert np.median(after) < 0.1 and np.max(after) < 0.35


def test_undistorted_optics_give_identity(config):
    result = D.fit_correction(grid_images(config, {}, channels=("dapi",)), "20x_air", 50.0, nominal_pixel(config, 512))
    assert result.correction.max_distortion_px("dapi") < 0.15
    assert not result.warnings


def test_file_round_trip_and_edit_detection(config, tmp_path):
    correction = D.fit_correction(
        grid_images(config, OPTICS, frame_px=256, channels=("dapi",)), "20x_air", 50.0, nominal_pixel(config, 256)
    ).correction
    path = correction.save(tmp_path / "c.json")
    loaded = D.DistortionCorrection.load(path)
    assert loaded == correction and loaded.fingerprint == correction.fingerprint
    data = json.loads(path.read_text())
    data["channels"]["dapi"]["k1"] += 0.01
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="edited"):
        D.DistortionCorrection.load(path)


def test_refuses_wrong_frame_size_and_missing_channel(config):
    correction = D.fit_correction(
        grid_images(config, OPTICS, frame_px=256, channels=("dapi",)), "20x_air", 50.0, nominal_pixel(config, 256)
    ).correction
    with pytest.raises(D.CalibrationError, match="cy5"):
        D.FrameCorrector(correction, ("dapi", "cy5"), 665.6, 665.6)
    corrector = D.FrameCorrector(correction, ("dapi",), 665.6, 665.6)
    with pytest.raises(D.CalibrationError, match="binning"):
        corrector.correct("dapi", np.zeros((128, 128), np.uint16))


def test_full_frame_correction_has_a_generous_performance_bound():
    size = 1024
    correction = D.DistortionCorrection(
        objective_id="20x_air",
        frame_width_px=size,
        frame_height_px=size,
        grid_pitch_um=50.0,
        reference_channel="dapi",
        channels={"dapi": D.ChannelModel(pixel_size_um=0.325, k1=0.02)},
        nominal_pixel_size_um=0.325,
    )
    corrector = D.FrameCorrector(
        correction, ("dapi",), size * 0.325, size * 0.325
    )
    frame = np.arange(size * size, dtype=np.uint16).reshape(size, size)
    corrector.correct("dapi", frame)  # build the sampling map before timing
    started = time.perf_counter()
    corrected = corrector.correct("dapi", frame)
    elapsed = time.perf_counter() - started
    assert corrected.shape == frame.shape and corrected.dtype == frame.dtype
    assert elapsed < 2.0


def test_calibration_input_checks(config):
    images = grid_images(config, OPTICS, frame_px=256, channels=("dapi", "cy5"))
    moved = [images[0], replace(images[1], stage_xy_um=(30107.0, 13.0))]
    with pytest.raises(D.CalibrationError, match="stage moved"):
        D.fit_correction(moved, "20x_air", 50.0, nominal_pixel(config, 256))
    with pytest.raises(D.CalibrationError, match="pitch"):
        D.fit_correction(images, "20x_air", 10.0, nominal_pixel(config, 256))  # 3.8 px: too fine
    with pytest.raises(D.CalibrationError, match="does not match"):
        D.fit_correction(images, "20x_air", 80.0, nominal_pixel(config, 256))
    count, spacing = D.inspect_grid_image(images[0].image, 50.0 / nominal_pixel(config, 256))
    assert count > 30 and spacing == pytest.approx(19.2, abs=0.5)


def test_command_line_fits_saved_images(config, tmp_path, capsys):
    arguments = ["--objective", "20x_air", "--pitch", "50", "--output", str(tmp_path / "out.json")]
    for item in grid_images(config, OPTICS, frame_px=256, channels=("dapi", "cy5")):
        tifffile.imwrite(tmp_path / f"{item.channel_id}.tif", item.image)
        arguments += ["--image", f"{item.channel_id}={tmp_path / f'{item.channel_id}.tif'}"]
    # The CLI takes the configured pixel size, i.e. full-size camera frames; give it a
    # configuration whose camera matches the 256 px test images.
    cfg_path = tmp_path / "microscope.json"
    data = json.loads(Path("config/microscope.json").read_text())
    data["camera"]["width_px"] = data["camera"]["height_px"] = 256
    for objective in data["objectives"]:
        objective["pixel_size_um"] *= 2048 / 256
    cfg_path.write_text(json.dumps(data))
    assert D.main(["--config", str(cfg_path), *arguments]) == 0
    saved = D.DistortionCorrection.load(tmp_path / "out.json")
    assert set(saved.channels) == {"dapi", "cy5"}
    assert "Saved" in capsys.readouterr().out


# -- acquisition path -------------------------------------------------------------------
@pytest.fixture
def rig(config, tmp_path):
    cfg = replace(config, calibration_dir=tmp_path / "calibration")
    backend = SimulatedMicroscope(cfg, frame_px=128, distortion=OPTICS, calibration_slide=SLIDE)
    backend.change_objective(cfg.objective("20x_air"))
    controller = ScanController(cfg, backend)
    with mock.patch("microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)):
        yield cfg, backend, controller
    controller.close()


def calibrate(cfg, backend, controller, channels=("dapi", "cy5")):
    backend.move_xy(30000, 0)
    images = []
    for channel_id in channels:
        backend.set_channel(cfg.channel(channel_id), 50, 20)
        images.append(D.GridImage(channel_id, backend.snap(), (30000.0, 0.0)))
    result = controller.fit_distortion(images, "20x_air", 50.0)
    controller.save_correction(result.correction)
    backend.move_xy(0, 0)
    return result.correction


def plan_for(tmp_path, channels=("dapi", "cy5"), **options) -> AcquisitionPlan:
    return AcquisitionPlan(
        project="Proj",
        sample="Slide",
        output_root=tmp_path / "out",
        objective_id="20x_air",
        channels=tuple(ChannelSetting(item, 40, 20) for item in channels),
        regions=(ScanRegion(-600, -300, 1200, 600),),
        options=AcquisitionOptions(**options),
    )


def run_cycle(controller, plan):
    controller.start_cycle(plan)
    assert controller.wait(30)
    status = controller.status()
    assert status.state == "complete", status.error
    return controller.experiment.cycle_folder(controller.experiment.cycles[-1].number)


def test_cycle_writes_corrected_and_uncorrected_planes(rig, tmp_path):
    cfg, backend, controller = rig
    plan = plan_for(tmp_path, distortion_correction=True)
    assert any("No distortion calibration" in item for item in controller.preflight(plan))
    correction = calibrate(cfg, backend, controller)
    assert controller.preflight(plan) == []
    folder = run_cycle(controller, plan)

    experiment = controller.experiment
    recorded = experiment.calibration["distortion"]
    assert recorded["fingerprint"] == correction.fingerprint
    assert D.DistortionCorrection.load(experiment.root / recorded["file"]) == correction
    record = next(iter(tile_records(folder).values()))
    plane = record["planes"][0]
    corrected = tifffile.imread(folder / plane["file"])
    original = tifffile.imread(folder / plane["uncorrected_file"])
    assert plane["uncorrected_file"].startswith("uncorrected/")
    assert corrected.shape == original.shape and not np.array_equal(corrected, original)
    corrector = D.FrameCorrector(correction, ("dapi", "cy5"), *cfg.fov_um("20x_air"))
    assert np.array_equal(corrected, corrector.correct(plane["channel"], original))
    xml = (folder / "Metadata" / "R01.xml").read_text()
    assert f'DistortionCorrection="{correction.fingerprint}"' in xml

    # Later cycles use the experiment's copy, even after the lab recalibrates.
    calibrate(cfg, backend, controller, channels=("dapi",))
    assert controller.preflight(plan_for(tmp_path, distortion_correction=True)) == []
    off = plan_for(tmp_path, distortion_correction=False)
    assert any("fixed on" in item for item in controller.preflight(off))
    uncalibrated = plan_for(tmp_path, channels=("dapi", "af488"), distortion_correction=True)
    assert any("does not cover AF488" in item for item in controller.preflight(uncalibrated))


def test_cycle_without_uncorrected_copy(rig, tmp_path):
    cfg, backend, controller = rig
    calibrate(cfg, backend, controller)
    folder = run_cycle(controller, plan_for(tmp_path, distortion_correction=True, keep_uncorrected=False))
    assert not (folder / "uncorrected").exists()
    assert all("uncorrected_file" not in plane for record in tile_records(folder).values() for plane in record["planes"])


def test_correction_off_leaves_frames_untouched(rig, tmp_path):
    cfg, backend, controller = rig
    calibrate(cfg, backend, controller)
    folder = run_cycle(controller, plan_for(tmp_path))
    assert not (folder / "uncorrected").exists()
    assert "distortion" not in controller.experiment.calibration
    assert "DistortionCorrection" not in (folder / "Metadata" / "R01.xml").read_text()
    assert any("fixed off" in item for item in controller.preflight(plan_for(tmp_path, distortion_correction=True)))


def test_tampered_experiment_copy_blocks_new_cycles(rig, tmp_path):
    cfg, backend, controller = rig
    calibrate(cfg, backend, controller)
    run_cycle(controller, plan_for(tmp_path, distortion_correction=True))
    copy = controller.experiment.root / D.EXPERIMENT_COPY
    other = calibrate(cfg, backend, controller, channels=("dapi",))
    other.save(copy)
    problems = controller.preflight(plan_for(tmp_path, distortion_correction=True))
    assert any("no longer matches" in item for item in problems)


def test_import_checks_a_correction_made_elsewhere(rig, tmp_path):
    cfg, backend, controller = rig
    correction = calibrate(cfg, backend, controller)
    exported = correction.save(tmp_path / "from_another_pc.json")
    loaded, warnings = controller.read_correction_for_import(exported)
    assert loaded == correction
    # Simulated 128 px frames vs the configured 2048 px camera: flagged, not refused.
    assert any("128×128" in item and "2048×2048" in item for item in warnings)

    data = json.loads(exported.read_text())
    data["channels"]["mystery"] = data["channels"]["dapi"]
    del data["fingerprint"]
    (tmp_path / "extra.json").write_text(json.dumps(data))
    _, warnings = controller.read_correction_for_import(tmp_path / "extra.json")
    assert any("mystery" in item for item in warnings)

    data = json.loads(exported.read_text())
    data["objective_id"] = "100x_unknown"
    del data["fingerprint"]
    (tmp_path / "other.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="100x_unknown"):
        controller.read_correction_for_import(tmp_path / "other.json")
    (tmp_path / "junk.json").write_text("{}")
    with pytest.raises(ValueError, match="not a distortion correction"):
        controller.read_correction_for_import(tmp_path / "junk.json")


def test_calibration_dir_follows_the_config_file(tmp_path):
    data = json.loads(Path("config/microscope.json").read_text())
    data["calibration"] = {"directory": "cal"}
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(data))
    assert load_config(path).calibration_dir == tmp_path / "cal"
    assert load_config().calibration_dir == Path("config/microscope.json").resolve().parent / "calibration"
