from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from src.camera.realsense_camera import RealSenseCamera
from src.detection.apriltag_detector import AprilTagDetector
from src.transformations.apriltag_calculations import AprilTagCalculations


def transform_to_pose(T: list[list[float]]) -> dict[str, list[float]]:
    """
    Convert a homogeneous transform matrix into translation + quaternion.
    Quaternion returned as [x, y, z, w].
    """
    T = np.array(T)

    R = T[:3, :3]
    t = T[:3, 3]

    quat = SciRot.from_matrix(R).as_quat()
    quat = quat / np.linalg.norm(quat)

    return {
        "translation": t.tolist(),
        "quaternion": quat.tolist(),
    }


class MainVision:
    """
    High-level interface for the vision pipeline.

    Flow:
    1. Capture frame from RealSense
    2. Detect AprilTags and estimate their poses
    3. Look up the requested tool configuration
    4. Compute the pose error transform
    5. Return the result for IK
    """

    def __init__(
        self,
        config_path: str = "src/config/tag_targets.json",
        default_tag_size_m: float = 0.03,
    ) -> None:
        self.camera = RealSenseCamera()
        self.detector = AprilTagDetector(default_tag_size_m=default_tag_size_m)
        self.calculations = AprilTagCalculations(config_path=config_path)

        self.is_started = False
        self.camera_intrinsics: dict[str, Any] | None = None

    def start(self) -> None:
        """
        Start the camera pipeline and cache camera intrinsics.
        """
        if self.is_started:
            return

        self.camera.start()
        self.camera_intrinsics = self.camera.get_color_intrinsics()
        self.is_started = True

    def stop(self) -> None:
        """
        Stop the camera pipeline cleanly.
        """
        if not self.is_started:
            return

        self.camera.stop()
        self.is_started = False

    def get_detected_tags(self) -> dict[int, dict[str, Any]]:
        """
        Capture one frame and return detected tags.
        """
        if not self.is_started:
            raise RuntimeError("Vision pipeline is not started. Call start() first.")

        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are not available.")

        color_image, _ = self.camera.get_frames()
        if color_image is None:
            return {}

        detected_tags = self.detector.detect(
            color_image=color_image,
            camera_intrinsics=self.camera_intrinsics,
        )
        return detected_tags

    def get_tool_pose_error(self, tool_name: str) -> dict[str, Any]:
        """
        Capture one frame, detect tags, and compute the pose error for the requested tool.

        Output now includes quaternion representations.
        """
        detected_tags = self.get_detected_tags()

        result = self.calculations.calculate_pose_error(
            tool_name=tool_name,
            detected_tags=detected_tags,
        )

        # Convert transforms to quaternion representation
        if result.get("tag_visible"):

            if "current_T_tag_cam" in result:
                result["current_pose"] = transform_to_pose(
                    result["current_T_tag_cam"]
                )

            if "desired_T_tag_cam" in result:
                result["desired_pose"] = transform_to_pose(
                    result["desired_T_tag_cam"]
                )

            if "T_error" in result:
                result["error_pose"] = transform_to_pose(
                    result["T_error"]
                )

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the vision pipeline and output the pose error transform for a requested tool."
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="Tool name as defined in src/config/tag_targets.json",
    )
    parser.add_argument(
        "--config",
        default="src/config/tag_targets.json",
        help="Path to the tag target config JSON file.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.08,
        help="Default AprilTag size in meters.", #TODO:Change to 0.04 for smaller april tag
    )

    args = parser.parse_args()

    vision = MainVision(
        config_path=args.config,
        default_tag_size_m=args.tag_size,
    )

    try:
        vision.start()
        result = vision.get_tool_pose_error(tool_name=args.tool)
        print(json.dumps(result, indent=2))
    finally:
        vision.stop()


if __name__ == "__main__":
    main()