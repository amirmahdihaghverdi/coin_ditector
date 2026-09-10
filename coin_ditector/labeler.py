import cv2
import os
import glob

IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

CLASS_ID = 0

# اندازه پنجره
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

drawing = False
start_x = 0
start_y = 0
box = None

original_image = None
display_image = None

scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global drawing
    global start_x, start_y
    global box
    global display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        box = None

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            temp = display_image.copy()

            cv2.rectangle(
                temp,
                (start_x, start_y),
                (x, y),
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLO Labeler", temp)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)

        x2 = max(start_x, x)
        y2 = max(start_y, y)

        box = (x1, y1, x2, y2)

        cv2.rectangle(
            display_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLO Labeler", display_image)


# --------------------------------
# Find images
# --------------------------------

image_files = []

for ext in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]:

    image_files.extend(
        glob.glob(os.path.join(IMAGE_DIR, ext))
    )

image_files.sort()

print(f"Found {len(image_files)} images.")

if len(image_files) == 0:
    print("No images found!")
    exit()


# --------------------------------
# Create window
# --------------------------------

cv2.namedWindow(
    "YOLO Labeler",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "YOLO Labeler",
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)

cv2.setMouseCallback(
    "YOLO Labeler",
    mouse_callback
)


# --------------------------------
# Process images
# --------------------------------

for index, image_path in enumerate(image_files):

    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Could not read:", image_path)
        continue

    original_height, original_width = original_image.shape[:2]

    # --------------------------------
    # Calculate display scale
    # --------------------------------

    scale_x = WINDOW_WIDTH / original_width
    scale_y = WINDOW_HEIGHT / original_height

    scale = min(scale_x, scale_y, 1.0)

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    display_image = cv2.resize(
        original_image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    box = None

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    print()
    print("==============================")
    print(f"Image {index + 1}/{len(image_files)}")
    print(filename)
    print(f"Original: {original_width} x {original_height}")
    print(f"Display:  {display_width} x {display_height}")
    print("==============================")

    while True:

        temp = display_image.copy()

        cv2.putText(
            temp,
            "Draw box around the coin",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            temp,
            "ENTER = Save | R = Redo | Q = Quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "YOLO Labeler",
            temp
        )

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------
        # ENTER = Save
        # --------------------------------

        if key == 13:

            if box is None:

                print("Please draw a box first!")

                continue

            # Display coordinates
            x1, y1, x2, y2 = box

            # --------------------------------
            # Convert display coordinates
            # back to ORIGINAL image
            # --------------------------------

            x1_original = int(x1 / scale)
            y1_original = int(y1 / scale)

            x2_original = int(x2 / scale)
            y2_original = int(y2 / scale)

            # Clamp coordinates

            x1_original = max(
                0,
                min(x1_original, original_width - 1)
            )

            y1_original = max(
                0,
                min(y1_original, original_height - 1)
            )

            x2_original = max(
                0,
                min(x2_original, original_width - 1)
            )

            y2_original = max(
                0,
                min(y2_original, original_height - 1)
            )

            # --------------------------------
            # YOLO coordinates
            # --------------------------------

            x_center = (
                (x1_original + x2_original) / 2
            ) / original_width

            y_center = (
                (y1_original + y2_original) / 2
            ) / original_height

            box_width = (
                x2_original - x1_original
            ) / original_width

            box_height = (
                y2_original - y1_original
            ) / original_height

            # --------------------------------
            # Save label
            # --------------------------------

            label_path = os.path.join(
                LABEL_DIR,
                name + ".txt"
            )

            with open(
                label_path,
                "w"
            ) as f:

                f.write(
                    f"{CLASS_ID} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}\n"
                )

            print(
                "Saved:",
                label_path
            )

            break

        # --------------------------------
        # R = Redo
        # --------------------------------

        elif key == ord("r"):

            display_image = cv2.resize(
                original_image,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA
            )

            box = None

            print("Redo.")

        # --------------------------------
        # Q = Quit
        # --------------------------------

        elif key == ord("q"):

            print("Stopped.")

            cv2.destroyAllWindows()

            exit()


cv2.destroyAllWindows()

print()
print("==============================")
print("DONE!")
print(f"Processed {len(image_files)} images.")
print("==============================")