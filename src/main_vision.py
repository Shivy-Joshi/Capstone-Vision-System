from __future__ import annotations

import argparse
import json
from typing import Any

from src.camera.realsense_camera import RealSenseCamera
from src.detection.apriltag_detector import AprilTagDetector
from src.transformations.apriltag_calculations import AprilTagCalculations


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
    ) -> None:
        # Load the config once so tag sizes and tool poses are available everywhere.
        with open(config_path, "r", encoding="utf-8") as _f:
            self._config = json.load(_f)

        # Use the top-level default tag size from the config file as the detector's
        # fallback, keeping pose estimation consistent across the whole session.
        default_tag_size_m: float = self._config.get("default_tag_size_m", 0.08)

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
       ## print("\n--- APRILTAG DETECTOR OUTPUT ---")       #For Debugging
        ##print(json.dumps(detected_tags, indent=2, default=str))    #For Debugging
        return detected_tags

    def get_tag_size_for_tool(self, tool_name: str) -> float:
        """
        Return the tag size in metres for a specific tool, falling back to the
        config-level default if the tool does not define its own tag_size_m.
        """
        tool_config = self._config.get("tools", {}).get(tool_name, {})
        return tool_config.get(
            "tag_size_m",
            self._config.get("default_tag_size_m", 0.08),
        )

    def get_tool_pose_error(self, tool_name: str) -> dict[str, Any]:
        """
        Capture one frame, detect tags using the tool's calibrated tag size, and
        compute the pose error for the requested tool.

        Returns the homogeneous transform matrices directly.
        """
        if not self.is_started:
            raise RuntimeError("Vision pipeline is not started. Call start() first.")

        if self.camera_intrinsics is None:
            raise RuntimeError("Camera intrinsics are not available.")

        color_image, _ = self.camera.get_frames()
        if color_image is None:
            return {}

        # Use the tag size stored for this specific tool so pose estimation is
        # accurate even when different tools carry tags of different sizes.
        tag_size_m = self.get_tag_size_for_tool(tool_name)
        detected_tags = self.detector.detect(
            color_image=color_image,
            camera_intrinsics=self.camera_intrinsics,
            tag_size_m=tag_size_m,
        )

        result = self.calculations.calculate_pose_error(
            tool_name=tool_name,
            detected_tags=detected_tags,
        )
       # print("\n--- APRILTAG CALCULATIONS OUTPUT ---")    #For Debugging
        #print(json.dumps(result, indent=2, default=str))   #For Debugging
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

    args = parser.parse_args()

    # Tag sizes are read from the config file (per-tool tag_size_m or the top-level
    # default_tag_size_m), so no --tag-size flag is needed here.
    vision = MainVision(
        config_path=args.config,
    )

    try:
        vision.start()
        result = vision.get_tool_pose_error(tool_name=args.tool)
        print(json.dumps(result, indent=2))
    finally:
        vision.stop()


if __name__ == "__main__":
    main()