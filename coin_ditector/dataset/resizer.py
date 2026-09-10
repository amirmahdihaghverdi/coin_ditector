import cv2
import os

input_folder = r"C:\Users\Amir\Downloads\ML 07-20260824T134241Z-1-001\ML 07\coin_ditector\dataset\negetive"
output_folder = "resized_images_negetive"
max_dim = 1280

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imwrite(os.path.join(output_folder, filename), img)