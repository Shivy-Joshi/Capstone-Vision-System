import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """
    Thin wrapper around the Intel RealSense pipeline.

    Responsibilities:
    - configure and start the camera streams
    - align depth to color
    - return frames as NumPy arrays
    - provide camera intrinsics and depth scale
    - stop the pipeline cleanly
    """

    def __init__(
        self,
        color_width: int = 640,
        color_height: int = 480,
        color_fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: int = 30,
        align_to_color: bool = True,
    ) -> None:
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps

        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps

        self.align_to_color = align_to_color

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = None
        self.align = None
        self.depth_scale = None
        self.is_running = False

        self._configure_streams()

    def _configure_streams(self) -> None:
        """
        Tell the RealSense pipeline which streams to open.
        """
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.bgr8,
            self.color_fps,
        )

        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.depth_fps,
        )

    def start(self) -> None:
        """
        Start the camera pipeline and cache camera metadata needed later.
        """
        if self.is_running:
            return

        self.profile = self.pipeline.start(self.config)

        if self.align_to_color:
            self.align = rs.align(rs.stream.color)

        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.is_running = True

    def stop(self) -> None:
        """
        Stop the camera pipeline safely.
        """
        if not self.is_running:
            return

        self.pipeline.stop()
        self.is_running = False

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Wait for the next set of frames and return:
        - color image as HxWx3 NumPy array (BGR)
        - depth image as HxW NumPy array (uint16)

        Returns (None, None) if valid frames are not available.
        """
        if not self.is_running:
            raise RuntimeError("Camera pipeline is not running. Call start() first.")

        frames = self.pipeline.wait_for_frames()

        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def get_frames_and_frame_objects(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, rs.video_frame | None, rs.depth_frame | None]:
        """
        Same as get_frames(), but also returns the underlying RealSense frame objects.

        Useful when you need SDK functions like:
        - depth_frame.get_distance(x, y)
        """
        if not self.is_running:
            raise RuntimeError("Camera pipeline is not running. Call start() first.")

        frames = self.pipeline.wait_for_frames()

        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image, color_frame, depth_frame

    def get_color_intrinsics(self) -> dict:
        """
        Return color camera intrinsics in a simple dictionary.

        These are needed for pose estimation with AprilTags.
        """
        if not self.is_running:
            raise RuntimeError("Camera pipeline is not running. Call start() first.")

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        return {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "coeffs": list(intrinsics.coeffs),
            "model": str(intrinsics.model),
        }

    def get_depth_scale(self) -> float:
        """
        Return the depth scale in meters per depth unit.
        """
        if self.depth_scale is None:
            raise RuntimeError("Depth scale is not available. Call start() first.")

        return self.depth_scale