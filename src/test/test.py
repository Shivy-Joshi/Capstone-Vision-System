import cv2

from src.camera.realsense_camera import RealSenseCamera
from src.detections.apriltag_detector import AprilTagDetector


def main() -> None:
    camera = RealSenseCamera()
    detector = AprilTagDetector(default_tag_size_m=0.08)

    camera.start()
    intrinsics = camera.get_color_intrinsics()

    try:
        while True:
            color_image, _ = camera.get_frames()
            if color_image is None:
                continue

            detected_tags = detector.detect(
                color_image=color_image,
                camera_intrinsics=intrinsics,
            )

            print(detected_tags)

            annotated_image = detector.draw_detections(color_image, detected_tags)
            cv2.imshow("AprilTag Detector Test", annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()