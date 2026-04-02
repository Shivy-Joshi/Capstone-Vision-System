from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

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
    4. Save or copy the printed JSON snippet into tag_targets.json.
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
        self.camera_intrinsics = self.camera.get_color_intrinsics()

        for _ in range(self.warmup_frames):
            self.get_color_frame()

        self.is_started = True

    def stop(self) -> None:
        if self.is_started:
            self.camera.stop()
            self.is_started = False

    def get_color_frame(self) -> np.ndarray:
        """
        Retrieve a color frame from the camera wrapper while tolerating
        different RealSenseCamera method names.
        """
        if hasattr(self.camera, "get_aligned_frames"):
            color_frame, _ = self.camera.get_aligned_frames()
            return color_frame

        if hasattr(self.camera, "get_frames"):
            frames = self.camera.get_frames()
            if isinstance(frames, tuple):
                return frames[0]
            return frames

        if hasattr(self.camera, "get_color_frame"):
            return self.camera.get_color_frame()

        raise AttributeError(
            "RealSenseCamera does not provide get_aligned_frames, get_frames, or get_color_frame."
        )

    @staticmethod
    def draw_tag_overlay(
        image: np.ndarray,
        tag_id: int,
        tag_data: dict[str, Any],
    ) -> np.ndarray:
        display = image.copy()

        corners = np.array(tag_data["corners_px"], dtype=int)
        center = np.array(tag_data["center_px"], dtype=int)
        translation = np.array(tag_data["translation_m"], dtype=float)

        cv2.polylines(display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(display, tuple(center), 5, (0, 0, 255), -1)

        info_lines = [
            f"Tag ID: {tag_id}",
            f"X: {translation[0]:.4f} m",
            f"Y: {translation[1]:.4f} m",
            f"Z: {translation[2]:.4f} m",
            "Press R to record calibration",
            "Press Q to quit",
        ]

        text_y = 30
        for line in info_lines:
            cv2.putText(
                display,
                line,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            text_y += 30

        return display

    @staticmethod
    def draw_waiting_overlay(
        image: np.ndarray,
        tag_id: int,
    ) -> np.ndarray:
        display = image.copy()
        info_lines = [
            f"Looking for Tag ID: {tag_id}",
            "Tag not currently visible",
            "Press Q to quit",
        ]

        text_y = 30
        for line in info_lines:
            cv2.putText(
                display,
                line,
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            text_y += 30

        return display

    def run_guided_calibration(
        self,
        tool_name: str,
        tag_id: int,
        sample_count: int,
        delay_s: float,
    ) -> dict[str, Any] | None:
        if not self.is_started:
            raise RuntimeError("Camera has not been started.")
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable.")

        window_name = "Tag Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                color_frame = self.get_color_frame()

                detections = self.detector.detect(
                    color_image=color_frame,
                    camera_intrinsics=self.camera_intrinsics,
                    tag_size_m=self.tag_size_m,
                )

                tag_data = detections.get(tag_id)
                if tag_data and tag_data.get("in_frame", False):
                    display = self.draw_tag_overlay(color_frame, tag_id, tag_data)
                else:
                    display = self.draw_waiting_overlay(color_frame, tag_id)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return None

                if key == ord("r"):
                    if not tag_data or not tag_data.get("in_frame", False):
                        print("Cannot record calibration because the tag is not visible.")
                        continue

                    print("Recording calibration samples...")
                    avg_translation, avg_rotation = self.collect_samples(
                        tag_id=tag_id,
                        sample_count=sample_count,
                        delay_s=delay_s,
                    )

                    entry = self.build_tag_target_entry(
                        tool_name=tool_name,
                        tag_id=tag_id,
                        translation_m=avg_translation,
                        rotation_matrix=avg_rotation,
                    )
                    return entry
        finally:
            cv2.destroyWindow(window_name)

    def get_tag_pose(self, tag_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.is_started:
            raise RuntimeError("Camera has not been started.")
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable.")

        color_frame = self.get_color_frame()

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

    @staticmethod
    def rotation_matrix_to_rpy_deg(rotation_matrix: np.ndarray) -> list[float]:
        """
        Convert a 3x3 rotation matrix into XYZ roll, pitch, yaw in degrees.
        """
        rpy_deg = SciRot.from_matrix(np.asarray(rotation_matrix, dtype=float)).as_euler(
            "xyz",
            degrees=True,
        )
        return np.round(rpy_deg, 6).tolist()

    def build_tag_target_entry(
        self,
        tool_name: str,
        tag_id: int,
        translation_m: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> dict[str, Any]:
        rpy_deg = self.rotation_matrix_to_rpy_deg(rotation_matrix)

        return {
            "tools": {
                tool_name: {
                    "tag_id": tag_id,
                    "desired_camera_pose_wrt_tag": {
                        "position_m": np.round(translation_m, 6).tolist(),
                        "rpy_deg": rpy_deg,
                    },
                }
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

    if "tools" not in existing or not isinstance(existing["tools"], dict):
        existing["tools"] = {}

    new_tools = new_entry.get("tools", {})
    existing["tools"].update(new_tools)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GUI calibration tool for AprilTag target poses in tag_targets.json."
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Tool name to store under tools in tag_targets.json.",
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
        help="Number of pose samples to average after pressing R.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between recorded samples after pressing R.",
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
        entry = calibrator.run_guided_calibration(
            tool_name=args.tool,
            tag_id=args.tag_id,
            sample_count=args.samples,
            delay_s=args.delay,
        )
    finally:
        calibrator.stop()

    if entry is None:
        print("\nCalibration cancelled.")
        return

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