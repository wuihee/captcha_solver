import pathlib

import cv2

width = 100
col_width = 25
data_dir = pathlib.Path('./input/captcha_images')

for image_path in data_dir.iterdir():
    image = cv2.imread(str(image_path))
    label = image_path.name.split(".png")[0]

    for i in range(4):
        col = image[:, i * col_width:(i + 1) * col_width]
        cv2.imwrite(f"./input/captcha_letters/{label[i]}_{label}.png", col)
