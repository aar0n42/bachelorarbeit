import cv2
import numpy as np
import os
def fill_holes(binary_mask):

    _, binary_mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    return filled_mask

path = 'eval/'
allimg = os.listdir('eval')

target = 'eval/'
for i in allimg:
    input_image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)
    print(i)

    filled_mask = fill_holes(binary_mask)

    cv2.imwrite(target + i, filled_mask)
