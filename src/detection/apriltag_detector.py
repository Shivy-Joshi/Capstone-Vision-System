from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector


class AprilTagDetector:
    """
    Detect AprilTags and estimate their pose relative to the camera.

    Output format:
    {
        tag_id: {
            "in_frame": True,
            "tag_family": "tag36h11",
            "translation_m": [x_robot, y_robot, z_robot],
            "rotation_matrix": [[...], [...], [...]],  # expressed in robot axes
            "center_px": [u, v],
            "corners_px": [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
        },
        ...
    }
    """

    def __init__(
        self,
        tag_family: str = "tag36h11",
        default_tag_size_m: float | None = None,
        nthreads: int = 4,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        debug: int = 0,
    ) -> None:
        self.tag_family = tag_family
        self.default_tag_size_m = default_tag_size_m

        self.detector = Detector(
            families=tag_family,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=debug,
        )

    @staticmethod
    def convert_pose_to_robot_axes(
        translation_m: list[float],
        rotation_matrix: list[list[float]],
    ) -> tuple[list[float], list[list[float]]]:
        """
        Convert pose data from the camera axis convention to the robot axis convention.

        Camera axes:
            -Z = depth
            +X = left
            +Y = up

        Robot axes:
            Y = depth
            +X = right
            +Z = up

        Mapping used:
            X_robot = -X_camera
            Y_robot = Z_camera
            Z_robot = Y_camera
        """
        axis_transform = np.array(
            [
                [0.0, 0.0, 1.0],   #robot x
                [-1.0, 0.0, 0.0],   #robot y
                [0.0, 1.0, 0.0],    # robot z
            ],
            dtype=float,
        )

        translation_vec = np.array(translation_m, dtype=float).reshape(3)
        rotation_mat = np.array(rotation_matrix, dtype=float).reshape(3, 3)

        converted_translation = axis_transform @ translation_vec
        converted_rotation = axis_transform @ rotation_mat @ axis_transform.T

        return converted_translation.tolist(), converted_rotation.tolist()

    def detect(
        self,
        color_image: np.ndarray,
        camera_intrinsics: dict,
        tag_size_m: float | None = None,
    ) -> dict[int, dict[str, Any]]:
        """
        Detect AprilTags from a BGR image and return a dictionary keyed by tag ID.

        Args:
            color_image:
                OpenCV BGR image.
            camera_intrinsics:
                Dictionary containing fx, fy, cx, cy.
            tag_size_m:
                Physical tag size in meters. If omitted, default_tag_size_m is used.

        Returns:
            Dictionary keyed by tag_id with pose and image information.
        """
        if color_image is None:
            raise ValueError("color_image cannot be None.")

        if color_image.ndim != 3 or color_image.shape[2] != 3:
            raise ValueError("color_image must be a BGR image with shape (H, W, 3).")

        required_keys = ("fx", "fy", "cx", "cy")
        missing_keys = [key for key in required_keys if key not in camera_intrinsics]
        if missing_keys:
            raise KeyError(f"camera_intrinsics is missing keys: {missing_keys}")

        if tag_size_m is None:
            tag_size_m = self.default_tag_size_m

        if tag_size_m is None:
            raise ValueError(
                "tag_size_m was not provided and no default_tag_size_m is set."
            )

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])

        detections = self.detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size_m,
        )

        detected_tags: dict[int, dict[str, Any]] = {}

        for detection in detections:
            tag_id = int(detection.tag_id)

            translation_m = detection.pose_t.reshape(3).astype(float).tolist()
            rotation_matrix = detection.pose_R.astype(float).tolist()
            translation_m, rotation_matrix = self.convert_pose_to_robot_axes(
                translation_m,
                rotation_matrix,
            )
            center_px = detection.center.astype(float).tolist()
            corners_px = detection.corners.astype(float).tolist()

            tag_family = detection.tag_family
            if isinstance(tag_family, bytes):
                tag_family = tag_family.decode("utf-8")

            detected_tags[tag_id] = {
                "in_frame": True,
                "tag_family": str(tag_family),
                "translation_m": translation_m,
                "rotation_matrix": rotation_matrix,
                "center_px": center_px,
                "corners_px": corners_px,
            }

        return detected_tags

    @staticmethod
    def draw_detections(
        color_image: np.ndarray,
        detected_tags: dict[int, dict[str, Any]],
    ) -> np.ndarray:
        """
        Return a copy of the image with tag outlines, centers, and IDs drawn on it.
        """
        annotated_image = color_image.copy()

        for tag_id, tag_data in detected_tags.items():
            corners = np.array(tag_data["corners_px"], dtype=int)
            center = tuple(np.array(tag_data["center_px"], dtype=int))

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(annotated_image, p1, p2, (0, 255, 0), 2)

            cv2.circle(annotated_image, center, 5, (0, 0, 255), -1)

            translation_m = tag_data["translation_m"]
            cv2.putText(
                annotated_image,
                f"ID {tag_id}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                annotated_image,
                f"x={translation_m[0]:.3f} y={translation_m[1]:.3f} z={translation_m[2]:.3f}",
                (center[0] + 10, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        return annotated_image