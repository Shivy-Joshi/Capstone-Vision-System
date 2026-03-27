import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.detection.apriltag_detector import AprilTagDetector
from src.transformations.apriltag_calculations import AprilTagCalculations


class MainVision:
    """
    Main vision pipeline.

    Responsibilities:
    - Load tag target configuration
    - Start/stop the camera stream
    - Run AprilTag detection
    - Compute target pose error for a selected tool/task
    - Provide the latest frame/results for CLI or GUI use
    """

    def __init__(
        self,
        config_path: str = "src/config/tag_targets.json",
        default_tag_size_m: float = 0.03,
    ) -> None:
        self.config_path = Path(config_path)
        self.default_tag_size_m = default_tag_size_m

        self.config = self._load_config(self.config_path)

        self.detector = AprilTagDetector(
            tag_size_m=self.default_tag_size_m,
        )
        self.calculator = AprilTagCalculations()

        self.started = False
        self.latest_frame: np.ndarray | None = None
        self.latest_detections: list[dict[str, Any]] = []
        self.latest_result: dict[str, Any] | None = None

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("Config file must contain a JSON object.")

        return config

    def start(self) -> None:
        if self.started:
            return

        self.detector.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return

        self.detector.stop()
        self.started = False

    def update(self) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
        """
        Grab the latest frame and detections from the detector.
        Returns:
            (frame, detections)
        """
        if not self.started:
            raise RuntimeError("MainVision.start() must be called before update().")

        frame, detections = self.detector.get_frame_and_detections()

        self.latest_frame = frame
        self.latest_detections = detections if detections is not None else []

        return self.latest_frame, self.latest_detections

    def get_visible_tag_ids(self) -> list[int]:
        visible_ids: list[int] = []

        for det in self.latest_detections:
            tag_id = det.get("tag_id")
            if isinstance(tag_id, int):
                visible_ids.append(tag_id)

        return visible_ids

    def _get_tool_config(self, tool_name: str) -> dict[str, Any]:
        tools = self.config.get("tools")
        if not isinstance(tools, dict):
            raise ValueError("Config must contain a top-level 'tools' object.")

        tool_cfg = tools.get(tool_name)
        if tool_cfg is None:
            raise KeyError(f"Tool '{tool_name}' not found in config.")

        if not isinstance(tool_cfg, dict):
            raise ValueError(f"Tool config for '{tool_name}' must be an object.")

        return tool_cfg

    def _find_detection_for_required_tag(
        self,
        required_tag_id: int,
    ) -> dict[str, Any] | None:
        for det in self.latest_detections:
            if det.get("tag_id") == required_tag_id:
                return det
        return None

    def get_tool_pose_error(self, tool_name: str) -> dict[str, Any]:
        """
        Compute the target pose error for the requested tool.

        Expected config structure example:

        {
          "tools": {
            "fastener_tool": {
              "required_tag_id": 1,
              "target_transform": {
                "translation_m": [0.0, 0.0, 0.05],
                "rotation_rpy_deg": [0.0, 0.0, 0.0]
              }
            }
          }
        }
        """
        if not self.started:
            raise RuntimeError("MainVision.start() must be called before get_tool_pose_error().")

        if self.latest_frame is None:
            self.update()

        tool_cfg = self._get_tool_config(tool_name)

        required_tag_id = tool_cfg.get("required_tag_id")
        if not isinstance(required_tag_id, int):
            raise ValueError(
                f"Tool '{tool_name}' must define an integer 'required_tag_id'."
            )

        detection = self._find_detection_for_required_tag(required_tag_id)

        if detection is None:
            result = {
                "success": False,
                "tool_name": tool_name,
                "required_tag_id": required_tag_id,
                "visible_tag_ids": self.get_visible_tag_ids(),
                "message": f"Required tag {required_tag_id} is not visible.",
            }
            self.latest_result = result
            return result

        target_transform_cfg = tool_cfg.get("target_transform", {})
        if not isinstance(target_transform_cfg, dict):
            raise ValueError(
                f"Tool '{tool_name}' has invalid 'target_transform' config."
            )

        translation_m = target_transform_cfg.get("translation_m", [0.0, 0.0, 0.0])
        rotation_rpy_deg = target_transform_cfg.get("rotation_rpy_deg", [0.0, 0.0, 0.0])

        if (
            not isinstance(translation_m, list)
            or len(translation_m) != 3
            or not all(isinstance(v, (int, float)) for v in translation_m)
        ):
            raise ValueError(
                f"Tool '{tool_name}' target_transform.translation_m must be a 3-element number list."
            )

        if (
            not isinstance(rotation_rpy_deg, list)
            or len(rotation_rpy_deg) != 3
            or not all(isinstance(v, (int, float)) for v in rotation_rpy_deg)
        ):
            raise ValueError(
                f"Tool '{tool_name}' target_transform.rotation_rpy_deg must be a 3-element number list."
            )

        result = self.calculator.compute_pose_error(
            detection=detection,
            target_translation_m=np.array(translation_m, dtype=float),
            target_rotation_rpy_deg=np.array(rotation_rpy_deg, dtype=float),
        )

        output = {
            "success": True,
            "tool_name": tool_name,
            "required_tag_id": required_tag_id,
            "visible_tag_ids": self.get_visible_tag_ids(),
            "tag_detection": {
                "tag_id": detection.get("tag_id"),
            },
            "pose_error": result,
        }

        self.latest_result = output
        return output

    def draw_overlay(
        self,
        frame: np.ndarray,
        tool_name: str | None = None,
    ) -> np.ndarray:
        """
        Draw AprilTag overlays and optional tool status on top of the frame.
        """
        annotated = frame.copy()

        for det in self.latest_detections:
            corners = det.get("corners")
            center = det.get("center")
            tag_id = det.get("tag_id")

            if corners is not None:
                corners_np = np.asarray(corners, dtype=int)
                if corners_np.shape == (4, 2):
                    for i in range(4):
                        p1 = tuple(corners_np[i])
                        p2 = tuple(corners_np[(i + 1) % 4])
                        cv2.line(annotated, p1, p2, (0, 255, 0), 2)

            if center is not None:
                center_np = np.asarray(center, dtype=int).reshape(2)
                center_pt = tuple(center_np)
                cv2.circle(annotated, center_pt, 5, (0, 0, 255), -1)

                cv2.putText(
                    annotated,
                    f"ID {tag_id}",
                    (center_pt[0] + 10, center_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if tool_name is not None:
            try:
                tool_cfg = self._get_tool_config(tool_name)
                required_tag_id = tool_cfg.get("required_tag_id", "N/A")
                visible = required_tag_id in self.get_visible_tag_ids()
                status_text = f"Tool: {tool_name} | Required Tag: {required_tag_id} | Visible: {visible}"
            except Exception:
                status_text = f"Tool: {tool_name}"

            cv2.putText(
                annotated,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self.latest_result is not None and self.latest_result.get("success"):
            pose_error = self.latest_result.get("pose_error", {})
            translation = pose_error.get("translation_error_m", [0.0, 0.0, 0.0])

            cv2.putText(
                annotated,
                f"dX: {translation[0]: .4f} m",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"dY: {translation[1]: .4f} m",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"dZ: {translation[2]: .4f} m",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        elif self.latest_result is not None and not self.latest_result.get("success"):
            message = self.latest_result.get("message", "No result")
            cv2.putText(
                annotated,
                message,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Main entry point for the vision system."
    )

    parser.add_argument(
        "--tool",
        required=True,
        help="Tool name defined in src/config/tag_targets.json",
    )

    parser.add_argument(
        "--config",
        default="src/config/tag_targets.json",
        help="Path to config file.",
    )

    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.03,
        help="AprilTag size in meters.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the GUI instead of CLI mode.",
    )

    args = parser.parse_args()

    if args.gui:
        from src.gui.vision_gui import VisionGUI

        gui = VisionGUI(
            tool_name=args.tool,
            config_path=args.config,
            default_tag_size_m=args.tag_size,
        )
        gui.run()
        return

    vision = MainVision(
        config_path=args.config,
        default_tag_size_m=args.tag_size,
    )

    try:
        vision.start()

        while True:
            frame, _ = vision.update()
            result = vision.get_tool_pose_error(
                tool_name=args.tool,
            )

            if frame is not None:
                annotated = vision.draw_overlay(frame, tool_name=args.tool)
                cv2.imshow("Main Vision", annotated)

            print(json.dumps(result, indent=2))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        vision.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()