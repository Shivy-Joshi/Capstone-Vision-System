from __future__ import annotations

import argparse
import json
from typing import Any

import cv2
import numpy as np

from src.main_vision import MainVision


class VisionGUI:
    """
    OpenCV-based GUI for live AprilTag visualization and pose error display.

    Features:
    - live camera feed
    - AprilTag outlines and IDs
    - selected tool display
    - required tag visibility status
    - T_error translation display
    - pretty-printed T_error in terminal on key press
    """

    def __init__(
        self,
        tool_name: str,
        config_path: str = "src/config/tag_targets.json",
        window_name: str = "Vision GUI",
    ) -> None:
        self.tool_name = tool_name
        self.window_name = window_name

        # Tag sizes are read from the config file, so no size param is needed here.
        self.vision = MainVision(
            config_path=config_path,
        )

    @staticmethod
    def _pretty_print_matrix(name: str, matrix_list: list[list[float]]) -> None:
        matrix = np.array(matrix_list, dtype=float)
        print(f"\n{name}:")
        print(np.array_str(matrix, precision=4, suppress_small=True))

    @staticmethod
    def _draw_text_block(
        image: np.ndarray,
        lines: list[str],
        origin: tuple[int, int] = (20, 30),
        line_spacing: int = 30,
        text_scale: float = 0.7,
        text_color: tuple[int, int, int] = (255, 255, 255),
        bg_color: tuple[int, int, int] = (40, 40, 40),
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Draw a semi-transparent text panel on the image.
        """
        overlay = image.copy()

        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        max_width = 0
        for line in lines:
            (w, _), _ = cv2.getTextSize(line, font, text_scale, thickness)
            max_width = max(max_width, w)

        panel_width = max_width + 20
        panel_height = line_spacing * len(lines) + 10

        cv2.rectangle(
            overlay,
            (x - 10, y - 25),
            (x - 10 + panel_width, y - 25 + panel_height),
            bg_color,
            -1,
        )

        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        for i, line in enumerate(lines):
            line_y = y + i * line_spacing
            cv2.putText(
                image,
                line,
                (x, line_y),
                font,
                text_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return image

    @staticmethod
    def _format_pose_result_lines(result: dict[str, Any]) -> list[str]:
        lines = [
            f"Tool: {result.get('tool', 'N/A')}",
            f"Tag ID: {result.get('tag_id', 'N/A')}",
        ]

        if not result.get("tag_visible", False):
            lines.append("Status: required tag not visible")
            if "error" in result:
                lines.append(f"Error: {result['error']}")
            return lines

        lines.append("Status: tag visible")

        if "T_error" in result:
            T_error = np.array(result["T_error"], dtype=float)
            tx, ty, tz = T_error[0, 3], T_error[1, 3], T_error[2, 3]

            lines.append(f"T_error tx: {tx:.4f} m")
            lines.append(f"T_error ty: {ty:.4f} m")
            lines.append(f"T_error tz: {tz:.4f} m")

        return lines

    def _print_result_to_terminal(self, result: dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print(json.dumps(result, indent=2))

        if result.get("tag_visible", False):
            if "current_T_tag_cam" in result:
                self._pretty_print_matrix("current_T_tag_cam", result["current_T_tag_cam"])
            if "desired_T_tag_cam" in result:
                self._pretty_print_matrix("desired_T_tag_cam", result["desired_T_tag_cam"])
            if "T_error" in result:
                self._pretty_print_matrix("T_error", result["T_error"])

        print("=" * 80)

    def run(self) -> None:
        self.vision.start()

        print("Vision GUI started.")
        print(f"Tracking tool: {self.tool_name}")
        print("Controls:")
        print("  q or ESC -> quit")
        print("  p        -> print full result to terminal")

        try:
            while True:
                if self.vision.camera_intrinsics is None:
                    raise RuntimeError("Camera intrinsics are not available.")

                color_image, _ = self.vision.camera.get_frames()
                if color_image is None:
                    continue

                # Detect with the tag size specific to the tracked tool so that
                # pose estimation uses the correct physical dimensions.
                tag_size_m = self.vision.get_tag_size_for_tool(self.tool_name)
                detected_tags = self.vision.detector.detect(
                    color_image=color_image,
                    camera_intrinsics=self.vision.camera_intrinsics,
                    tag_size_m=tag_size_m,
                )

                pose_result = self.vision.calculations.calculate_pose_error(
                    tool_name=self.tool_name,
                    detected_tags=detected_tags,
                )

                annotated_image = self.vision.detector.draw_detections(
                    color_image=color_image,
                    detected_tags=detected_tags,
                )

                lines = self._format_pose_result_lines(pose_result)
                annotated_image = self._draw_text_block(
                    annotated_image,
                    lines=lines,
                    origin=(20, 35),
                )

                help_lines = [
                    "q/ESC: quit",
                    "p: print current matrices",
                ]
                annotated_image = self._draw_text_block(
                    annotated_image,
                    lines=help_lines,
                    origin=(20, annotated_image.shape[0] - 50),
                    line_spacing=25,
                    text_scale=0.55,
                )

                cv2.imshow(self.window_name, annotated_image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                if key == ord("p"):
                    self._print_result_to_terminal(pose_result)

        finally:
            self.vision.stop()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live AprilTag vision GUI.")
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
        "--window-name",
        default="Vision GUI",
        help="Display window name.",
    )

    args = parser.parse_args()

    # Tag sizes come from the config file — no --tag-size flag needed.
    gui = VisionGUI(
        tool_name=args.tool,
        config_path=args.config,
        window_name=args.window_name,
    )
    gui.run()


if __name__ == "__main__":
    main()