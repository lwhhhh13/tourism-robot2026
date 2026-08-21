#!/usr/bin/env python3
"""Windows/local DISCOVERSE camera inspector (development only, not formal client input)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CAMERA_EXPECTED = {0: "head_cam", 1: "lft_handeye", 2: "rgt_handeye"}


def describe(array: np.ndarray) -> dict:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": None if finite.size == 0 else float(finite.min()),
        "max": None if finite.size == 0 else float(finite.max()),
    }


def save_depth_preview(depth: np.ndarray, path: Path) -> None:
    values = np.asarray(depth).squeeze().astype(np.float32)
    valid = np.isfinite(values) & (values > 0)
    preview = np.zeros(values.shape, np.uint8)
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        preview[valid] = np.clip((values[valid] - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(preview, cv2.COLORMAP_TURBO))


def build_sim(repository_root: Path):
    sys.path.insert(0, str(repository_root))
    from discoverse.robots_env.mmk2_base import MMK2Cfg
    from discoverse.task_base import MMK2TaskBase

    class ObservationSim(MMK2TaskBase):
        def domain_randomization(self):
            pass

        def check_success(self):
            return False

    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = False
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_box.xml"
    cfg.sync = False
    cfg.headless = True
    cfg.render_set = {"fps": 20, "width": 640, "height": 480}
    cfg.obs_rgb_cam_id = [0, 1, 2]
    cfg.obs_depth_cam_id = [0, 1, 2]
    return ObservationSim(cfg)


def main() -> int:
    root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root_default)
    parser.add_argument("--output", type=Path, default=Path("recordings/sim_observation"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sim = build_sim(args.repository_root.resolve())
    try:
        observation = sim.reset()
        observation, _, _, _, _ = sim.step(np.asarray(sim.target_control).copy())
        print("obs.keys():", list(observation.keys()))
        print("camera_names:", list(sim.camera_names))
        report = {"obs_keys": list(observation.keys()), "cameras": {}}
        for camera_id in (0, 1, 2):
            name = sim.camera_names[camera_id]
            rgb = observation.get("img", {}).get(camera_id)
            depth = observation.get("depth", {}).get(camera_id)
            camera_dir = output / f"cam_{camera_id}_{name}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            print(f"cam_id={camera_id} name={name} expected={CAMERA_EXPECTED[camera_id]}")
            print("  RGB:", "MISSING" if rgb is None else describe(rgb))
            print("  Depth:", "MISSING" if depth is None else describe(depth))
            if rgb is not None:
                cv2.imwrite(str(camera_dir / "frame_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if depth is not None:
                np.save(camera_dir / "frame_depth.npy", np.asarray(depth, np.float32))
                save_depth_preview(depth, camera_dir / "frame_depth_preview.png")
            report["cameras"][str(camera_id)] = {
                "name": name, "expected": CAMERA_EXPECTED[camera_id],
                "mapping_ok": name == CAMERA_EXPECTED[camera_id],
                "rgb": None if rgb is None else describe(rgb),
                "depth": None if depth is None else describe(depth),
            }
        (output / "observation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("saved:", output)
        return 0
    finally:
        sim.running = False
        cleanup = getattr(sim, "_cleanup_before_exit", None)
        if callable(cleanup):
            cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
