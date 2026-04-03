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


# This class manages the full AprilTag calibration workflow.
# It starts the camera, detects a specific tag, lets the user visually confirm
# that the tag is in frame, collects multiple pose samples, averages them,
# and formats the result so it can be pasted into tag_targets.json.
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
        tag_size_m: float = 0.08,
        warmup_frames: int = 20,
    ) -> None:
        # Camera wrapper used to acquire live frames from the RealSense.
        self.camera = RealSenseCamera()
        # AprilTag detector configured with the physical tag size so pose can be estimated.
        self.detector = AprilTagDetector(default_tag_size_m=tag_size_m)
        # Store calibration settings for later reuse.
        self.tag_size_m = tag_size_m
        self.warmup_frames = warmup_frames
        # Camera intrinsics are needed for pose estimation from image coordinates.
        self.camera_intrinsics: dict[str, Any] | None = None
        # Tracks whether the camera pipeline is currently running.
        self.is_started = False

    def start(self) -> None:
        if self.is_started:
            return

        # Start the RealSense stream before trying to read any frames.
        self.camera.start()
        # Read the color camera intrinsics once so the detector can estimate 3D pose.
        self.camera_intrinsics = self.camera.get_color_intrinsics()

        # Discard a few early frames so exposure / auto settings can settle.
        for _ in range(self.warmup_frames):
            self.get_color_frame()

        # Mark the calibration session as active.
        self.is_started = True

    def stop(self) -> None:
        if self.is_started:
            # Stop the camera cleanly so the device is released properly.
            self.camera.stop()
            self.is_started = False

    def get_color_frame(self) -> np.ndarray:
        """
        Retrieve a color frame from the camera wrapper while tolerating
        different RealSenseCamera method names.
        """
        # Support multiple camera wrapper APIs so this script still works even if
        # the RealSenseCamera class uses different method names.
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

        # If none of the expected interfaces exist, fail with a clear error.
        raise AttributeError(
            "RealSenseCamera does not provide get_aligned_frames, get_frames, or get_color_frame."
        )

    @staticmethod
    def draw_tag_overlay(
        image: np.ndarray,
        tag_id: int,
        tag_data: dict[str, Any],
        camera_intrinsics: dict[str, Any],
    ) -> np.ndarray:
        # Work on a copy so the original frame is not modified in place.
        display = image.copy()

        # Extract 2D drawing information and the current 3D translation estimate.
        corners = np.array(tag_data["corners_px"], dtype=int)
        center = np.array(tag_data["center_px"], dtype=int)
        translation = np.array(tag_data["translation_m"], dtype=float)
        rotation_matrix = np.array(tag_data["rotation_matrix"], dtype=float)
        rpy_deg = SciRot.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)

        # Draw the detected tag outline and its center point.
        cv2.polylines(display, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(display, tuple(center), 5, (0, 0, 255), -1)

        # Draw projected orientation axes on top of the tag.
        axis_length_m = 0.05
        axis_points_3d = np.float32(
            [
                [0.0, 0.0, 0.0],
                [axis_length_m, 0.0, 0.0],
                [0.0, axis_length_m, 0.0],
                [0.0, 0.0, axis_length_m],
            ]
        )

        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])

        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        distortion = np.zeros((5, 1), dtype=np.float32)

        rotation_vector, _ = cv2.Rodrigues(rotation_matrix.astype(np.float32))
        translation_vector = translation.astype(np.float32).reshape(3, 1)

        image_points, _ = cv2.projectPoints(
            axis_points_3d,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
        image_points = np.round(image_points.reshape(-1, 2)).astype(int)

        origin = tuple(image_points[0])
        x_axis_end = tuple(image_points[1])
        y_axis_end = tuple(image_points[2])
        z_axis_end = tuple(image_points[3])

        cv2.arrowedLine(display, origin, x_axis_end, (0, 0, 255), 3, tipLength=0.2)
        cv2.arrowedLine(display, origin, y_axis_end, (0, 255, 0), 3, tipLength=0.2)
        cv2.arrowedLine(display, origin, z_axis_end, (255, 0, 0), 3, tipLength=0.2)

        cv2.putText(
            display,
            "X",
            (x_axis_end[0] + 8, x_axis_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Y",
            (y_axis_end[0] + 8, y_axis_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Z",
            (z_axis_end[0] + 8, z_axis_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the live translation and rotation estimate so the operator can position the tool.
        info_lines = [
            f"Tag ID: {tag_id}",
            f"X: {translation[0]:.4f} m",
            f"Y: {translation[1]:.4f} m",
            f"Z: {translation[2]:.4f} m",
            f"Roll  X: {rpy_deg[0]:.2f} deg",
            f"Pitch Y: {rpy_deg[1]:.2f} deg",
            f"Yaw   Z: {rpy_deg[2]:.2f} deg",
            "Press R to record calibration",
            "Press Q to quit",
        ]

        # Render the status text line by line near the top-left of the image.
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
        # Show a fallback overlay when the requested tag is not currently detected.
        display = image.copy()
        info_lines = [
            f"Looking for Tag ID: {tag_id}",
            "Tag not currently visible",
            "Press Q to quit",
        ]

        # Render the warning text line by line.
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

        # Create a resizable OpenCV window for the live calibration view.
        window_name = "Tag Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            # Main interactive loop: display the camera feed until the user quits
            # or records a calibration.
            while True:
                # Detect tags in the current frame and estimate their poses.
                color_frame = self.get_color_frame()

                detections = self.detector.detect(
                    color_image=color_frame,
                    camera_intrinsics=self.camera_intrinsics,
                    tag_size_m=self.tag_size_m,
                )

                # Pull out only the requested tag because that is the one tied to this tool.
                tag_data = detections.get(tag_id)
                if tag_data and tag_data.get("in_frame", False):
                    display = self.draw_tag_overlay(
                        color_frame,
                        tag_id,
                        tag_data,
                        self.camera_intrinsics,
                    )
                else:
                    display = self.draw_waiting_overlay(color_frame, tag_id)

                # Show the annotated live frame to the user.
                cv2.imshow(window_name, display)
                # Read keyboard input: q quits, r records samples.
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return None

                # Once the operator is happy with the pose, record and average samples.
                if key == ord("r"):
                    if not tag_data or not tag_data.get("in_frame", False):
                        print("Cannot record calibration because the tag is not visible.")
                        continue

                    print("Recording calibration samples...")
                    # Collect multiple measurements to reduce frame-to-frame noise.
                    avg_translation, avg_rotation = self.collect_samples(
                        tag_id=tag_id,
                        sample_count=sample_count,
                        delay_s=delay_s,
                    )

                    # Convert the averaged pose into the JSON structure expected by the config file.
                    entry = self.build_tag_target_entry(
                        tool_name=tool_name,
                        tag_id=tag_id,
                        translation_m=avg_translation,
                        rotation_matrix=avg_rotation,
                    )
                    return entry
        finally:
            # Ensure the OpenCV window is closed even if the loop exits unexpectedly.
            cv2.destroyWindow(window_name)

    def get_tag_pose(self, tag_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.is_started:
            raise RuntimeError("Camera has not been started.")
        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are unavailable.")

        # Grab one frame and run detection on it.
        color_frame = self.get_color_frame()

        # Look for the requested tag only.
        detections = self.detector.detect(
            color_image=color_frame,
            camera_intrinsics=self.camera_intrinsics,
            tag_size_m=self.tag_size_m,
        )

        tag_data = detections.get(tag_id)
        if not tag_data or not tag_data.get("in_frame", False):
            return None

        # Convert the detector output into NumPy arrays for later averaging.
        translation = np.array(tag_data["translation_m"], dtype=float)
        rotation = np.array(tag_data["rotation_matrix"], dtype=float)
        return translation, rotation

    def collect_samples(
        self,
        tag_id: int,
        sample_count: int,
        delay_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Store each successful translation and rotation sample separately.
        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []

        print(f"Collecting {sample_count} samples for tag {tag_id}...")
        print("Keep the tool steady in its desired target pose.")

        # Keep sampling until the requested number of valid tag poses has been collected.
        while len(translations) < sample_count:
            # Attempt to measure the tag pose from the current frame.
            pose = self.get_tag_pose(tag_id)

            # Skip frames where the tag is missing instead of recording bad data.
            if pose is None:
                print("Tag not visible. Waiting...")
                time.sleep(delay_s)
                continue

            # Save the current pose estimate.
            translation, rotation = pose
            translations.append(translation)
            rotations.append(rotation)

            print(
                f"Sample {len(translations)}/{sample_count}: "
                f"translation = {np.round(translation, 6).tolist()}"
            )
            time.sleep(delay_s)

        # Average all recorded translations component-wise.
        avg_translation = np.mean(np.stack(translations, axis=0), axis=0)

        # A raw element-wise average of rotation matrices is usually not a valid rotation,
        # so we project it back onto the nearest proper rotation matrix using SVD.
        avg_rotation = np.mean(np.stack(rotations, axis=0), axis=0)
        u, _, vt = np.linalg.svd(avg_rotation)
        avg_rotation = u @ vt

        # Guard against reflections by forcing the result to have determinant +1.
        if np.linalg.det(avg_rotation) < 0:
            u[:, -1] *= -1.0
            avg_rotation = u @ vt

        return avg_translation, avg_rotation

    @staticmethod
    def rotation_matrix_to_rpy_deg(rotation_matrix: np.ndarray) -> list[float]:
        """
        Convert a 3x3 rotation matrix into XYZ roll, pitch, yaw in degrees.
        """
        # Convert the rotation matrix into XYZ Euler angles because the config file
        # stores orientation as roll, pitch, yaw in degrees.
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
        # Convert the averaged rotation matrix into the config's RPY representation.
        rpy_deg = self.rotation_matrix_to_rpy_deg(rotation_matrix)

        # Build the exact nested dictionary shape expected by tag_targets.json,
        # including the physical tag size used during calibration so other scripts
        # can read it directly from the config instead of hardcoding it.
        return {
            "tools": {
                tool_name: {
                    "tag_id": tag_id,
                    "tag_size_m": self.tag_size_m,
                    "desired_camera_pose_wrt_tag": {
                        "position_m": np.round(translation_m, 6).tolist(),
                        "rpy_deg": rpy_deg,
                    },
                }
            }
        }


# Merge a newly calibrated tool entry into an existing config file without
# deleting the other tools that are already present.
def merge_into_existing_config(
    config_path: Path,
    new_entry: dict[str, Any],
    update_default_tag_size: bool = False,
) -> dict[str, Any]:
    # Load the existing config if it already exists; otherwise start from empty.
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    else:
        existing = {}

    # Optionally update the top-level default tag size.  This is useful when
    # calibrating for the first time or when all tags share a new size.
    if update_default_tag_size and "tools" in new_entry:
        # Infer the default from the first (and only) tool in the new entry.
        first_tool = next(iter(new_entry["tools"].values()), {})
        new_default = first_tool.get("tag_size_m")
        if new_default is not None:
            existing["default_tag_size_m"] = new_default

    # If no default exists yet at all, seed it from the new tool entry.
    if "default_tag_size_m" not in existing and "tools" in new_entry:
        first_tool = next(iter(new_entry["tools"].values()), {})
        new_default = first_tool.get("tag_size_m")
        if new_default is not None:
            existing["default_tag_size_m"] = new_default

    # Make sure the top-level "tools" key exists and has the correct type.
    if "tools" not in existing or not isinstance(existing["tools"], dict):
        existing["tools"] = {}

    # Insert or overwrite only the calibrated tool entry.
    new_tools = new_entry.get("tools", {})
    existing["tools"].update(new_tools)
    return existing


# CLI entry point: parse arguments, run calibration, print the result,
# and optionally write it into the JSON config file.
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
        default=None,
        help=(
            "AprilTag size in meters (e.g. 0.08 for an 80 mm tag). "
            "If omitted you will be prompted to enter it interactively."
        ),
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

    # Resolve the tag size: prefer the CLI flag, then fall back to an interactive
    # prompt so the operator always sets the value explicitly rather than relying
    # on a silent hardcoded default.
    tag_size_m = args.tag_size
    if tag_size_m is None:
        # Try to suggest the existing default from the config file as a hint.
        config_path_hint = Path(args.config)
        existing_default: float | None = None
        if config_path_hint.exists():
            try:
                with config_path_hint.open("r", encoding="utf-8") as _f:
                    existing_default = json.load(_f).get("default_tag_size_m")
            except (json.JSONDecodeError, OSError):
                pass

        hint_str = f" [{existing_default} m]" if existing_default is not None else " [e.g. 0.08]"
        while tag_size_m is None:
            raw = input(f"Enter AprilTag size in metres{hint_str}: ").strip()
            if raw == "" and existing_default is not None:
                tag_size_m = existing_default
                print(f"Using existing default: {tag_size_m} m")
            else:
                try:
                    tag_size_m = float(raw)
                    if tag_size_m <= 0:
                        print("Tag size must be greater than zero. Please try again.")
                        tag_size_m = None
                except ValueError:
                    print("Invalid input — please enter a number such as 0.08.")

    # Create the calibrator using the confirmed physical tag size.
    calibrator = TagCalibration(tag_size_m=tag_size_m)

    # Always stop the camera in a finally block so the device is not left open.
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

    # A None result means the user quit the GUI without saving a calibration.
    if entry is None:
        print("\nCalibration cancelled.")
        return

    # Print the generated JSON snippet so it can be copied manually if desired.
    print("\nCalibration result:")
    print(json.dumps(entry, indent=4))

    # Optionally merge the new calibration into the config file on disk.
    if args.write:
        config_path = Path(args.config)

        # Ask whether to update the top-level default tag size as well.
        update_default = False
        raw_update = input(
            f"\nAlso update the default_tag_size_m in the config to {tag_size_m} m? (y/N): "
        ).strip().lower()
        if raw_update == "y":
            update_default = True

        merged = merge_into_existing_config(config_path, entry, update_default_tag_size=update_default)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(merged, file, indent=4)
        print(f"\nUpdated config written to: {config_path}")
    else:
        print("\nDry run only. Use --write to save into tag_targets.json.")


if __name__ == "__main__":
    main()