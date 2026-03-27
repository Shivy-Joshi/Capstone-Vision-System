from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class AprilTagCalculations:
    """
    Compute current-vs-desired camera pose error for a requested tool.

    Expected detector input format:
    {
        tag_id: {
            "in_frame": True,
            "tag_family": "tag36h11",
            "translation_m": [x, y, z],
            "rotation_matrix": [[...], [...], [...]],
            "center_px": [u, v],
            "corners_px": [[...], [...], [...], [...]]
        },
        ...
    }

    Expected config JSON format:
    {
      "tools": {
        "connector_tool": {
          "tag_id": 1,
          "desired_camera_pose_wrt_tag": {
            "position_m": [0.0, 0.0, 0.12],
            "rpy_deg": [180.0, 0.0, 0.0]
          }
        }
      }
    }
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _pose_to_transform(rotation_matrix: list[list[float]], translation_m: list[float]) -> np.ndarray:
        """
        Convert rotation + translation into a 4x4 homogeneous transform.
        """
        R = np.array(rotation_matrix, dtype=float)
        t = np.array(translation_m, dtype=float).reshape(3)

        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {R.shape}")

        if t.shape != (3,):
            raise ValueError(f"Translation vector must have shape (3,), got {t.shape}")

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def _invert_transform(T: np.ndarray) -> np.ndarray:
        """
        Invert a rigid body transform.
        """
        if T.shape != (4, 4):
            raise ValueError(f"Transform must have shape (4, 4), got {T.shape}")

        R = T[:3, :3]
        t = T[:3, 3]

        T_inv = np.eye(4, dtype=float)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    @staticmethod
    def _rpy_deg_to_rotation_matrix(rpy_deg: list[float]) -> np.ndarray:
        """
        Convert roll, pitch, yaw in degrees to a rotation matrix.

        Convention used:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        roll, pitch, yaw = np.radians(np.array(rpy_deg, dtype=float))

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ],
            dtype=float,
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ],
            dtype=float,
        )

        Rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        return Rz @ Ry @ Rx

    @staticmethod
    def _rotation_matrix_to_rpy_deg(R: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix to roll, pitch, yaw in degrees.

        Convention matches:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {R.shape}")

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0

        return np.degrees(np.array([roll, pitch, yaw], dtype=float))

    @staticmethod
    def _desired_pose_to_transform(position_m: list[float], rpy_deg: list[float]) -> np.ndarray:
        """
        Build desired tag->camera transform from config.
        """
        R = AprilTagCalculations._rpy_deg_to_rotation_matrix(rpy_deg)
        t = np.array(position_m, dtype=float).reshape(3)

        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _get_tool_config(self, tool_name: str) -> dict[str, Any]:
        tools = self.config.get("tools", {})
        if tool_name not in tools:
            raise KeyError(f"Tool '{tool_name}' not found in config file.")
        return tools[tool_name]

    def calculate_pose_error(
            self,
            tool_name: str,
            detected_tags: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        For the requested tool:
        1. find the required tag
        2. get current camera pose relative to tag
        3. get desired camera pose relative to tag
        4. compute delta transform
        5. return homogeneous transform matrices

        Returns:
            {
                "tool": ...,
                "tag_id": ...,
                "tag_visible": ...,
                "current_T_tag_cam": [[...], [...], [...], [...]],
                "desired_T_tag_cam": [[...], [...], [...], [...]],
                "T_error": [[...], [...], [...], [...]]
            }
        """
        tool_config = self._get_tool_config(tool_name)

        tag_id = int(tool_config["tag_id"])
        desired_pose = tool_config["desired_camera_pose_wrt_tag"]

        if tag_id not in detected_tags:
            return {
                "tool": tool_name,
                "tag_id": tag_id,
                "tag_visible": False,
                "error": "Required AprilTag not detected",
            }

        tag_data = detected_tags[tag_id]

        current_T_cam_tag = self._pose_to_transform(
            rotation_matrix=tag_data["rotation_matrix"],
            translation_m=tag_data["translation_m"],
        )

        current_T_tag_cam = self._invert_transform(current_T_cam_tag)

        desired_T_tag_cam = self._desired_pose_to_transform(
            position_m=desired_pose["position_m"],
            rpy_deg=desired_pose["rpy_deg"],
        )

        T_error = self._invert_transform(current_T_tag_cam) @ desired_T_tag_cam

        return {
            "tool": tool_name,
            "tag_id": tag_id,
            "tag_visible": True,
            "current_T_tag_cam": current_T_tag_cam.tolist(),
            "desired_T_tag_cam": desired_T_tag_cam.tolist(),
            "T_error": T_error.tolist(),
        }