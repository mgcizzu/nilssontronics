from __future__ import annotations

import json
from pathlib import Path

from microscope_ui.config import load_config
from microscope_ui.mm_profile import (
    MMConfigDiscovery,
    build_profile,
    inspect_mm_config,
    save_profile,
)


class FakeCore:
    def __init__(self) -> None:
        self.loaded = None
        self.unloaded = False

    def loadSystemConfiguration(self, path):
        self.loaded = path

    def getAvailableConfigGroups(self):
        return ("Channel", "System")

    def getAvailableConfigs(self, group):
        return {"Channel": ("DAPI", "FITC"), "System": ("Startup",)}[group]

    def getLoadedDevices(self):
        return ("Camera", "Objective", "XY", "Z")

    def getDeviceType(self, device):
        return "StateDevice" if device == "Objective" else "CameraDevice"

    def getStateLabels(self, device):
        return ("10x", "20x")

    def getCameraDevice(self):
        return "Camera"

    def getImageWidth(self):
        return 2048

    def getImageHeight(self):
        return 1024

    def getXYStageDevice(self):
        return "XY"

    def getFocusDevice(self):
        return "Z"

    def unloadAllDevices(self):
        self.unloaded = True


def test_inspection_reads_inventory_and_always_unloads(tmp_path):
    config_path = tmp_path / "scope.cfg"
    config_path.write_text("# fake", encoding="utf-8")
    core = FakeCore()
    result = inspect_mm_config(config_path, core_factory=lambda _path: core)

    assert core.loaded == str(config_path)
    assert core.unloaded
    assert result.camera_device == "Camera"
    assert (result.camera_width_px, result.camera_height_px) == (2048, 1024)
    assert result.config_groups["Channel"] == ("DAPI", "FITC")
    assert result.state_labels == {"Objective": ("10x", "20x")}


def test_generated_profile_is_normal_valid_app_config(tmp_path):
    mm_config = tmp_path / "scope.cfg"
    mm_config.write_text("# fake", encoding="utf-8")
    discovery = MMConfigDiscovery(
        config_path=mm_config,
        adapter_path=None,
        camera_device="Camera",
        camera_width_px=512,
        camera_height_px=256,
        xy_stage_device="XY",
        focus_device="Z",
        config_groups={"Channel": ("DAPI",)},
        state_labels={"Objective": ("20x",)},
    )
    output = tmp_path / "nilssontronics_profile.json"
    profile = build_profile(
        discovery,
        microscope_name="Lab microscope",
        channel_group="Channel",
        channels=[
            {
                "id": "dapi",
                "name": "DAPI",
                "preset": "DAPI",
                "color": "#6c8cff",
                "emission_wavelength_nm": 461,
                "default_exposure_ms": 40,
                "max_exposure_ms": 1000,
                "default_power_percent": 0,
                "max_power_percent": 0,
                "manual_prompt": None,
            }
        ],
        objective_device="Objective",
        objectives=[
            {
                "id": "20x",
                "name": "20x air",
                "magnification": 20,
                "numerical_aperture": 0.5,
                "immersion": "air",
                "immersion_refractive_index": 1.0,
                "turret_position": 0,
                "pixel_size_um": 0.325,
                "safe_z_min_um": -1000,
                "safe_z_max_um": 1000,
            }
        ],
        stage_limits={
            "x_min_um": -50000,
            "x_max_um": 50000,
            "y_min_um": -30000,
            "y_max_um": 30000,
            "z_min_um": -2000,
            "z_max_um": 2000,
        },
        objective_change_z_um=-1500,
        output_path=output,
    )

    saved = save_profile(profile, output)
    config = load_config(saved)
    raw = json.loads(saved.read_text(encoding="utf-8"))
    assert config.hardware["backend"] == "mmcore"
    assert config.hardware["mm_config_path"] == str(mm_config)
    assert raw["hardware"]["mm_config_path"] == "scope.cfg"
    assert config.hardware["channel_presets"] == {"dapi": "DAPI"}
    assert config.camera_width_px == 512
