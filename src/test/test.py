import json

import cv2
import numpy as np

from src.camera.realsense_camera import RealSenseCamera
from src.detection.apriltag_detector import AprilTagDetector
from src.transformations.apriltag_calculations import AprilTagCalculations


def pretty_print_matrix(name: str, matrix_list: list[list[float]]) -> None:
    matrix = np.array(matrix_list, dtype=float)
    print(f"\n{name}:")
    print(np.array_str(matrix, precision=4, suppress_small=True))


def main() -> None:
    tool_name = "connector_tool"
    config_path = "src/config/tag_targets.json"
    default_tag_size_m = 0.03

    camera = RealSenseCamera()
    detector = AprilTagDetector(default_tag_size_m=default_tag_size_m)
    calculations = AprilTagCalculations(config_path=config_path)

    camera.start()
    intrinsics = camera.get_color_intrinsics()

    print("Camera started.")
    print(f"Testing tool: {tool_name}")
    print("Press 'q' or ESC to quit.")
    print("Press 'p' to print the current result.\n")

    try:
        while True:
            color_image, _ = camera.get_frames()
            if color_image is None:
                continue

            detected_tags = detector.detect(
                color_image=color_image,
                camera_intrinsics=intrinsics,
            )

            annotated_image = detector.draw_detections(color_image, detected_tags)

            result = calculations.calculate_pose_error(
                tool_name=tool_name,
                detected_tags=detected_tags,
            )

            if result.get("tag_visible", False):
                tag_id = result["tag_id"]

                cv2.putText(
                    annotated_image,
                    f"Tool: {tool_name}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated_image,
                    f"Tracking tag ID: {tag_id}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                t_error = np.array(result["T_error"], dtype=float)
                tx, ty, tz = t_error[0, 3], t_error[1, 3], t_error[2, 3]

                cv2.putText(
                    annotated_image,
                    f"T_error translation: x={tx:.3f} y={ty:.3f} z={tz:.3f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    annotated_image,
                    f"Tool: {tool_name}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated_image,
                    result.get("error", "Required tag not visible"),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("AprilTag Calculations Live Test", annotated_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("p"):
                print("\n" + "=" * 80)
                print("Current result:")
                print(json.dumps(result, indent=2))

                if result.get("tag_visible", False):
                    pretty_print_matrix("current_T_tag_cam", result["current_T_tag_cam"])
                    pretty_print_matrix("desired_T_tag_cam", result["desired_T_tag_cam"])
                    pretty_print_matrix("T_error", result["T_error"])
                print("=" * 80)

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()