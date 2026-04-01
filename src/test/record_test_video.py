from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 is not installed.")
    print("Install it with: pip install pyrealsense2")
    sys.exit(1)


@dataclass
class RecorderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    output_dir: Path = Path("testing_videos")
    codec: str = "mp4v"   # good default for .mp4


class RealSenseVideoRecorder:
    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )

        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.current_output_path: Optional[Path] = None

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def start_camera(self) -> None:
        self.pipeline.start(self.rs_config)

    def stop_camera(self) -> None:
        self.stop_recording()
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def get_frame(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        frame = np.asanyarray(color_frame.get_data())
        return frame

    def _generate_output_path(self) -> Path:
        """
        Generate a timestamped filename and avoid overwrite even if a file
        already exists with the same second-level timestamp.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"test_video_{timestamp}"
        output_path = self.config.output_dir / f"{base_name}.mp4"

        counter = 1
        while output_path.exists():
            output_path = self.config.output_dir / f"{base_name}_{counter}.mp4"
            counter += 1

        return output_path

    def start_recording(self) -> None:
        if self.is_recording:
            print("Already recording.")
            return

        output_path = self._generate_output_path()
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.config.fps,
            (self.config.width, self.config.height),
        )

        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Could not open video writer for: {output_path}")

        self.is_recording = True
        self.current_output_path = output_path
        print(f"Recording started: {output_path}")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        print(f"Recording stopped: {self.current_output_path}")
        self.is_recording = False
        self.current_output_path = None

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()

        status_text = "RECORDING" if self.is_recording else "LIVE"
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)

        cv2.putText(
            overlay,
            f"Status: {status_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            overlay,
            "Press 'r' to start/stop recording | Press 'q' to quit",
            (20, self.config.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return overlay

    def run(self) -> None:
        self.start_camera()
        print("Camera started.")
        print("Press 'r' to start/stop recording.")
        print("Press 'q' to quit.")

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    continue

                display_frame = self.draw_overlay(frame)

                if self.is_recording and self.writer is not None:
                    self.writer.write(frame)

                cv2.imshow("Arm Camera Test Recorder", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):
                    self.toggle_recording()
                elif key == ord("q"):
                    break

        finally:
            self.stop_camera()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record test videos from the arm-mounted RealSense camera."
    )
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="testing_videos",
        help="Folder where videos will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RecorderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=Path(args.output_dir),
    )

    recorder = RealSenseVideoRecorder(config)
    recorder.run()


if __name__ == "__main__":
    main()