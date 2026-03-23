import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector


OUTPUT_CSV = Path("tag_poses.csv")
TAG_SIZE_M = 0.03  # change to real tag size


def get_camera_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    return intr.fx, intr.fy, intr.ppx, intr.ppy


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    fx, fy, cx, cy = get_camera_intrinsics(profile)

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    file_exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "tag_id",
                "tx_m", "ty_m", "tz_m",
                "r00", "r01", "r02",
                "r10", "r11", "r12",
                "r20", "r21", "r22",
            ])

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
                    tag_size=TAG_SIZE_M,
                )

                for tag in tags:
                    t = tag.pose_t.reshape(3)
                    R = tag.pose_R
                    writer.writerow([
                        time.time(),
                        int(tag.tag_id),
                        float(t[0]), float(t[1]), float(t[2]),
                        float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
                        float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
                        float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
                    ])
                    f.flush()

                cv2.imshow("Saving Tag Poses", image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()