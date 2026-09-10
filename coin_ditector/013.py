import cv2
import os

# پوشه ذخیره عکس‌ها
save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    # نمایش تعداد عکس‌ها
    cv2.putText(
        frame,
        f"Images: {counter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("10p Dataset Creator", frame)

    key = cv2.waitKey(1) & 0xFF

    # S = Save
    if key == ord("s"):
        filename = os.path.join(
            save_dir,
            f"10p_{counter:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        counter += 1

    # Q = Quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()