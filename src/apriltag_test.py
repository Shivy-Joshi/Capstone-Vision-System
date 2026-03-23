import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

detector = Detector(families="tag36h11")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(gray)

    for tag in tags:
        corners = tag.corners.astype(int)

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(image, p1, p2, (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.circle(image, center, 5, (0, 0, 255), -1)

        cv2.putText(
            image,
            f"ID {tag.tag_id}",
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("AprilTag Detection", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipeline.stop()
cv2.destroyAllWindows()