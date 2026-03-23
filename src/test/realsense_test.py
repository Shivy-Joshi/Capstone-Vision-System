import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} meters/unit")

    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            stacked = np.hstack((color_image, depth_vis))
            cv2.imshow("Color | Depth", stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()