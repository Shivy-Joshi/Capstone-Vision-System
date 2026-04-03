from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from src.main_vision import MainVision


class VisionGUI:
    """
    Live OpenCV GUI for AprilTag pose-error visualization.

    Shows per-frame:
    - Camera feed with tag outlines and IDs for every detected tag.
    - 3D XYZ axes projected directly onto each tag (red=X, green=Y, blue=Z)
      using the correct camera-frame pose so the axes track the physical tag.
    - A cyan correction arrow on the tracked tool's tag whose direction and
      magnitude represent T_error.translation — i.e. where the camera needs
      to move to eliminate the pose error.
    - A HUD panel with tx / ty / tz / |t| error values.
    - Continuous stdout printing of the error on every frame.

    Controls:
        Q / ESC  — quit
        P        — pretty-print full pose matrices to terminal

    Usage (standalone):
        vision = MainVision(config_path="src/config/tag_targets.json")
        vision.start()
        try:
            gui = VisionGUI(vision, tool_name="claw_tool")
            gui.run()
        finally:
            vision.stop()

    Usage (via main_vision CLI):
        python -m src.main_vision --tool claw_tool --gui
    """

    # Length of the projected orientation axes drawn on each tag (metres).
    _AXIS_LEN_M: float = 0.05
    # Scale applied to T_error.translation before projecting the correction
    # arrow so that small errors are still clearly visible on screen.
    _ERROR_ARROW_SCALE: float = 5.0

    # Axis-conversion matrix used by AprilTagDetector
    #   X_robot = -Z_cam,  Y_robot = -X_cam,  Z_robot = Y_cam
    _AXIS_TRANSFORM: np.ndarray = np.array(
        [
            [0.0,  0.0, -1.0],
            [-1.0, 0.0,  0.0],
            [0.0,  1.0,  0.0],
        ],
        dtype=float,
    )

    # 180-degree roll about robot-X applied after the axis conversion.
    _ROLL_180: np.ndarray = np.array(
        [
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        vision: MainVision,
        tool_name: str,
        window_name: str = "Vision GUI",
    ) -> None:
        """
        Args:
            vision:      A MainVision instance.  The caller is responsible for
                         calling vision.start() before run() and vision.stop()
                         after run() returns.
            tool_name:   Tool name as defined in tag_targets.json.
            window_name: OpenCV window title.
        """
        self.vision = vision
        self.tool_name = tool_name
        self.window_name = window_name

    # ── coordinate helpers ────────────────────────────────────────────────────

    @classmethod
    def _to_camera_frame(
        cls,
        t_robot: np.ndarray,
        R_robot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Invert the detector's axis conversion to recover the original
        camera-frame pose required by cv2.projectPoints.

        Forward (detector):
            t_robot = A  @ t_cam
            R_robot = A  @ R_cam @ A.T @ roll180

        Inverse (here):
            t_cam   = A.T @ t_robot
            R_cam   = A.T @ (R_robot @ roll180) @ A
            (roll180 is its own inverse since it is a 180-degree rotation)
        """
        At = cls._AXIS_TRANSFORM.T
        t_cam = At @ t_robot
        R_cam = At @ (R_robot @ cls._ROLL_180) @ cls._AXIS_TRANSFORM
        return t_cam, R_cam

    @staticmethod
    def _build_camera_matrix(intrinsics: dict[str, float]) -> np.ndarray:
        return np.array(
            [
                [intrinsics["fx"], 0.0,             intrinsics["cx"]],
                [0.0,             intrinsics["fy"], intrinsics["cy"]],
                [0.0,             0.0,              1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _project_point(
        pt_cam: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> tuple[int, int] | None:
        """Project a single 3-D camera-frame point to pixel coordinates.
        Returns None if the point is behind the camera."""
        if pt_cam[2] <= 0.0:
            return None
        px = int(camera_matrix[0, 0] * pt_cam[0] / pt_cam[2] + camera_matrix[0, 2])
        py = int(camera_matrix[1, 1] * pt_cam[1] / pt_cam[2] + camera_matrix[1, 2])
        return px, py

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_tag_axes(
        self,
        image: np.ndarray,
        tag_data: dict[str, Any],
        camera_matrix: np.ndarray,
    ) -> None:
        """
        Project and draw X / Y / Z orientation axes centred on the tag.
        Modifies image in-place.
        """
        t_cam, R_cam = self._to_camera_frame(
            np.array(tag_data["translation_m"], dtype=float),
            np.array(tag_data["rotation_matrix"], dtype=float),
        )

        rvec, _ = cv2.Rodrigues(R_cam.astype(np.float32))
        tvec = t_cam.astype(np.float32).reshape(3, 1)
        dist = np.zeros((5, 1), dtype=np.float32)

        axis_pts_3d = np.float32(
            [
                [0.0, 0.0, 0.0],
                [self._AXIS_LEN_M, 0.0, 0.0],
                [0.0, self._AXIS_LEN_M, 0.0],
                [0.0, 0.0, self._AXIS_LEN_M],
            ]
        )
        img_pts, _ = cv2.projectPoints(axis_pts_3d, rvec, tvec, camera_matrix, dist)
        img_pts = np.round(img_pts.reshape(-1, 2)).astype(int)

        origin = tuple(img_pts[0])
        axes = [
            (tuple(img_pts[1]), (0,   0,   255), "X"),  # Red
            (tuple(img_pts[2]), (0,   255, 0  ), "Y"),  # Green
            (tuple(img_pts[3]), (255, 0,   0  ), "Z"),  # Blue
        ]
        for tip, colour, label in axes:
            cv2.arrowedLine(image, origin, tip, colour, 2, tipLength=0.25)
            cv2.putText(
                image, label, (tip[0] + 5, tip[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )

    def _draw_error_arrow(
        self,
        image: np.ndarray,
        tag_data: dict[str, Any],
        T_error: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        """
        Draw a cyan arrow from the tag centre showing the correction direction.

        T_error.translation is in robot frame.  It is converted to camera frame
        and then projected so the arrow points the same way the physical camera
        needs to move to reach the desired pose.
        """
        t_cam, _ = self._to_camera_frame(
            np.array(tag_data["translation_m"], dtype=float),
            np.array(tag_data["rotation_matrix"], dtype=float),
        )

        # Convert T_error translation from robot frame to camera frame.
        t_err_cam = self._AXIS_TRANSFORM.T @ T_error[:3, 3]

        # Arrow tip = tag position in camera frame shifted by the scaled correction.
        tip_3d = t_cam + t_err_cam * self._ERROR_ARROW_SCALE

        tag_center_px = tuple(np.array(tag_data["center_px"], dtype=int))
        tip_px = self._project_point(tip_3d, camera_matrix)
        if tip_px is None:
            return

        cv2.arrowedLine(image, tag_center_px, tip_px, (0, 255, 255), 3, tipLength=0.2)
        cv2.putText(
            image, "err",
            (tip_px[0] + 5, tip_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_text_block(
        image: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
        line_spacing: int = 28,
        text_scale: float = 0.65,
        text_colour: tuple[int, int, int] = (255, 255, 255),
        bg_colour: tuple[int, int, int] = (30, 30, 30),
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Overlay a semi-transparent text panel.  Returns the blended image."""
        if not lines:
            return image

        overlay = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        x, y = origin

        max_w = max(
            cv2.getTextSize(line, font, text_scale, thickness)[0][0]
            for line in lines
        )
        cv2.rectangle(
            overlay,
            (x - 8,        y - 22),
            (x + max_w + 12, y + line_spacing * len(lines) - 4),
            bg_colour,
            -1,
        )
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        for i, line in enumerate(lines):
            cv2.putText(
                image, line, (x, y + i * line_spacing),
                font, text_scale, text_colour, thickness, cv2.LINE_AA,
            )
        return image

    # ── frame composition ─────────────────────────────────────────────────────

    def _annotate_frame(
        self,
        color_image: np.ndarray,
        detected_tags: dict[int, dict[str, Any]],
        pose_result: dict[str, Any],
        camera_intrinsics: dict[str, float],
    ) -> np.ndarray:
        """Return an annotated copy of color_image with all overlays applied."""
        image = color_image.copy()
        camera_matrix = self._build_camera_matrix(camera_intrinsics)

        # Tag outlines, IDs, and 3-D axes for every detected tag.
        for tag_id, tag_data in detected_tags.items():
            corners = np.array(tag_data["corners_px"], dtype=int)
            center = tuple(np.array(tag_data["center_px"], dtype=int))

            cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            cv2.putText(
                image, f"ID {tag_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA,
            )

            self._draw_tag_axes(image, tag_data, camera_matrix)

        # Correction arrow only for the tracked tool's tag.
        tool_tag_id = pose_result.get("tag_id")
        if (
            pose_result.get("tag_visible", False)
            and tool_tag_id in detected_tags
            and "T_error" in pose_result
        ):
            self._draw_error_arrow(
                image,
                detected_tags[tool_tag_id],
                np.array(pose_result["T_error"], dtype=float),
                camera_matrix,
            )

        # HUD panel — top-left.
        hud_lines = [
            f"Tool:   {pose_result.get('tool', 'N/A')}",
            f"Tag ID: {pose_result.get('tag_id', 'N/A')}",
        ]
        if pose_result.get("tag_visible", False) and "T_error" in pose_result:
            T = np.array(pose_result["T_error"], dtype=float)
            hud_lines += [
                "Tag: VISIBLE",
                f"tx: {T[0, 3]:+.4f} m",
                f"ty: {T[1, 3]:+.4f} m",
                f"tz: {T[2, 3]:+.4f} m",
                f"|t|: {np.linalg.norm(T[:3, 3]):.4f} m",
            ]
        else:
            hud_lines.append("Tag: NOT VISIBLE")

        image = self._draw_text_block(image, hud_lines, origin=(15, 30))

        # Help text — bottom-left.
        image = self._draw_text_block(
            image,
            lines=["Q / ESC: quit", "P: print full matrices"],
            origin=(15, image.shape[0] - 55),
            line_spacing=22,
            text_scale=0.5,
        )

        return image

    # ── terminal output ───────────────────────────────────────────────────────

    @staticmethod
    def _print_error(pose_result: dict[str, Any]) -> None:
        """Print the T_error translation to stdout on every frame."""
        tool = pose_result.get("tool", "?")
        if not pose_result.get("tag_visible", False):
            print(f"[{tool}] tag NOT visible")
            return
        T = np.array(pose_result["T_error"], dtype=float)
        print(
            f"[{tool}]  "
            f"tx={T[0, 3]:+.4f}  "
            f"ty={T[1, 3]:+.4f}  "
            f"tz={T[2, 3]:+.4f}  "
            f"|t|={np.linalg.norm(T[:3, 3]):.4f} m"
        )

    @staticmethod
    def _print_full_result(pose_result: dict[str, Any]) -> None:
        """Pretty-print all pose matrices to the terminal (triggered by P key)."""
        print("\n" + "=" * 80)
        print(json.dumps(pose_result, indent=2))
        if pose_result.get("tag_visible", False):
            for key in ("current_T_cam_tag", "desired_T_cam_tag", "T_error"):
                if key in pose_result:
                    mat = np.array(pose_result[key], dtype=float)
                    print(f"\n{key}:")
                    print(np.array_str(mat, precision=4, suppress_small=True))
        print("=" * 80)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the GUI loop until the user presses Q or ESC.

        The caller must have already called vision.start() before invoking this
        method.  vision.stop() should be called after this method returns.
        """
        if not self.vision.is_started:
            raise RuntimeError(
                "MainVision is not started. Call vision.start() before gui.run()."
            )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                color_image, _ = self.vision.camera.get_frames()
                if color_image is None:
                    continue

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

                # Print error to terminal on every frame.
                self._print_error(pose_result)

                frame = self._annotate_frame(
                    color_image, detected_tags, pose_result, self.vision.camera_intrinsics,
                )
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    self._print_full_result(pose_result)

        finally:
            cv2.destroyWindow(self.window_name)


# ── standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the live AprilTag vision GUI.")
    parser.add_argument(
        "--tool",
        required=True,
        help="Tool name as defined in src/config/tag_targets.json.",
    )
    parser.add_argument(
        "--config",
        default="src/config/tag_targets.json",
        help="Path to the tag target config JSON file.",
    )
    parser.add_argument(
        "--window-name",
        default="Vision GUI",
        help="OpenCV window title.",
    )
    args = parser.parse_args()

    vision = MainVision(config_path=args.config)
    vision.start()
    try:
        gui = VisionGUI(
            vision=vision,
            tool_name=args.tool,
            window_name=args.window_name,
        )
        gui.run()
    finally:
        vision.stop()


if __name__ == "__main__":
    main()
