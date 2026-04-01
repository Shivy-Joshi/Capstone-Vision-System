

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.camera.realsense_camera import RealSenseCamera
from src.detection.apriltag_detector import AprilTagDetector


class TagCalibration:
    """
    Capture and average AprilTag poses so the result can be copied into
    tag_targets.json for a tool.

    Intended workflow:
    1. Mount the tool in its desired robot pose.
    2. Make sure the tag is visible to the camera.
    3. Run this script and collect multiple samples.
    4. Copy the printed JSON snippet into tag_targets.json.
    """

    def __init__(
        self,
        tag_size_m: float = 0.03,
        warmup_frames: int = 20,
    ) -> None:
        self.camera = RealSenseCamera()
        self.detector = AprilTagDetector(default_tag_size_m=tag_size_m)
        self.tag_size_m = tag_size_m
        self.warmup_frames = warmup_frames
        self.camera_intrinsics: dict[str, Any] | None = None
        self.is_started = False

    def start(self) -> None:
        if self.is_started:
            return

        self.camera.start()
        self.camera_intrinsics = self.camera.get_intrinsics()

        for _ in range(self.warmup_frames):
            self.camera.get_aligned_frames()

        self.is_started = True

    def stop(self) -> None:
        if self.is_started:
            self.camera.stop()
            self.is_started = False

    @staticmethod
    def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> list[float]:
        """
        Convert a 3x3 rotation matrix into quaternion [x, y, z, w].
        """
        r = np.asarray(rotation_matrix, dtype=float)
        trace = float(np.trace(r))

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s

        quaternion = np.array([x, y, z, w], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        return quaternion.tolist()

    @staticmethod
    def build_transform_matrix(
        translation_m: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> list[list[float]]:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation_m
        return transform.tolist()

    def get_tag_pose(self, tag_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.is_started:
            raise RuntimeError("Camera has not been started.")
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable.")

        color_frame, _ = self.camera.get_aligned_frames()

        detections = self.detector.detect(
            color_image=color_frame,
            camera_intrinsics=self.camera_intrinsics,
            tag_size_m=self.tag_size_m,
        )

        tag_data = detections.get(tag_id)
        if not tag_data or not tag_data.get("in_frame", False):
            return None

        translation = np.array(tag_data["translation_m"], dtype=float)
        rotation = np.array(tag_data["rotation_matrix"], dtype=float)
        return translation, rotation

    def collect_samples(
        self,
        tag_id: int,
        sample_count: int,
        delay_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []

        print(f"Collecting {sample_count} samples for tag {tag_id}...")
        print("Keep the tool steady in its desired target pose.")

        while len(translations) < sample_count:
            pose = self.get_tag_pose(tag_id)

            if pose is None:
                print("Tag not visible. Waiting...")
                time.sleep(delay_s)
                continue

            translation, rotation = pose
            translations.append(translation)
            rotations.append(rotation)

            print(
                f"Sample {len(translations)}/{sample_count}: "
                f"translation = {np.round(translation, 6).tolist()}"
            )
            time.sleep(delay_s)

        avg_translation = np.mean(np.stack(translations, axis=0), axis=0)

        avg_rotation = np.mean(np.stack(rotations, axis=0), axis=0)
        u, _, vt = np.linalg.svd(avg_rotation)
        avg_rotation = u @ vt

        if np.linalg.det(avg_rotation) < 0:
            u[:, -1] *= -1.0
            avg_rotation = u @ vt

        return avg_translation, avg_rotation

    def build_tag_target_entry(
        self,
        tool_name: str,
        tag_id: int,
        translation_m: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> dict[str, Any]:
        quaternion_xyzw = self.rotation_matrix_to_quaternion(rotation_matrix)
        transform_matrix = self.build_transform_matrix(translation_m, rotation_matrix)

        return {
            tool_name: {
                "tag_id": tag_id,
                "target_translation_m": np.round(translation_m, 6).tolist(),
                "target_rotation_matrix": np.round(rotation_matrix, 6).tolist(),
                "target_quaternion_xyzw": np.round(quaternion_xyzw, 6).tolist(),
                "target_transform": np.round(np.array(transform_matrix), 6).tolist(),
            }
        }


def merge_into_existing_config(
    config_path: Path,
    new_entry: dict[str, Any],
) -> dict[str, Any]:
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    else:
        existing = {}

    existing.update(new_entry)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate AprilTag target poses for tag_targets.json."
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Tool name to use as the top-level key in tag_targets.json.",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        required=True,
        help="AprilTag ID attached to the tool.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of pose samples to average.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between samples.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.03,
        help="AprilTag size in meters.",
    )
    parser.add_argument(
        "--config",
        default="src/config/tag_targets.json",
        help="Path to tag_targets.json.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the generated entry into the config file.",
    )
    args = parser.parse_args()

    calibrator = TagCalibration(tag_size_m=args.tag_size)

    try:
        calibrator.start()
        avg_translation, avg_rotation = calibrator.collect_samples(
            tag_id=args.tag_id,
            sample_count=args.samples,
            delay_s=args.delay,
        )
    finally:
        calibrator.stop()

    entry = calibrator.build_tag_target_entry(
        tool_name=args.tool,
        tag_id=args.tag_id,
        translation_m=avg_translation,
        rotation_matrix=avg_rotation,
    )

    print("\nCalibration result:")
    print(json.dumps(entry, indent=4))

    if args.write:
        config_path = Path(args.config)
        merged = merge_into_existing_config(config_path, entry)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(merged, file, indent=4)
        print(f"\nUpdated config written to: {config_path}")
    else:
        print("\nDry run only. Use --write to save into tag_targets.json.")


if __name__ == "__main__":
    main()