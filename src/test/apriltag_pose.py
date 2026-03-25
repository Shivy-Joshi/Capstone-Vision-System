import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


def get_camera_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    return fx, fy, cx, cy


def draw_tag(frame, tag):
    corners = tag.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    center = tuple(tag.center.astype(int))
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"ID {tag.tag_id}",
        (center[0] + 10, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )


def main():
    tag_size_m = 0.0796  # 30 mm tag; change this to your actual printed tag size in m

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    fx, fy, cx, cy = get_camera_intrinsics(profile)
    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            tags = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=tag_size_m,
            )

            for tag in tags:
                draw_tag(image, tag)

                t = tag.pose_t.reshape(3)
                R = tag.pose_R

                cv2.putText(
                    image,
                    f"x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} m",
                    (20, 30 + 30 * (tag.tag_id % 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                print(f"Tag {tag.tag_id}: t={t}")
                print(f"Tag {tag.tag_id}: R=\n{R}\n")

            cv2.imshow("AprilTag Pose", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()