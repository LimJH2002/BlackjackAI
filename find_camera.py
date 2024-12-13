import cv2


def list_cameras():
    """Try to open each camera index until we hit an unopenable one."""
    available_cameras = []
    index = 0

    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if ret:
            # Get camera name if possible
            name = cap.getBackendName()
            # Get resolution
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            available_cameras.append(
                {
                    "index": index,
                    "name": name,
                    "resolution": f"{int(width)}x{int(height)}",
                }
            )

        cap.release()
        index += 1

    return available_cameras


def main():
    print("Searching for cameras...")
    cameras = list_cameras()

    if not cameras:
        print("No cameras found!")
        return

    print("\nAvailable cameras:")
    for camera in cameras:
        print(f"\nCamera {camera['index']}:")
        print(f"  Backend: {camera['name']}")
        print(f"  Resolution: {camera['resolution']}")

    print("\nTo use a specific camera, update the camera index in your main script:")
    print("cap = cv2.VideoCapture(X)  # Replace X with the desired camera index")


if __name__ == "__main__":
    main()
