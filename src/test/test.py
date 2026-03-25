from ..camera.realsense_camera import RealSenseCamera


def main():
    camera = RealSenseCamera()
    camera.start()

    print("Depth scale:", camera.get_depth_scale())
    print("Intrinsics:", camera.get_color_intrinsics())

    color_image, depth_image = camera.get_frames()
    print("Color shape:", color_image.shape if color_image is not None else None)
    print("Depth shape:", depth_image.shape if depth_image is not None else None)

    camera.stop()


if __name__ == "__main__":
    main()