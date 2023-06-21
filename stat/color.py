from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import cv2
import glob

path = '/home/aaron.goecke/DFUC2022_/train/images/*.jpg'

def combine_histograms(hist_list):
    combined_hist = np.zeros_like(hist_list[0])
    for hist in hist_list:
        combined_hist += hist
    return combined_hist

image_files = glob.glob(path)
images = [cv2.imread(img) for img in image_files]


hist_list_r = []
hist_list_g = []
hist_list_b = []


for image in images:
    channels = cv2.split(image)

    for idx, channel in enumerate(channels):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        if idx == 0:
            hist_list_b.append(hist)
        elif idx == 1:
            hist_list_g.append(hist)
        else:
            hist_list_r.append(hist)

combined_hist_r = combine_histograms(hist_list_r)
combined_hist_g = combine_histograms(hist_list_g)
combined_hist_b = combine_histograms(hist_list_b)
plt.plot(combined_hist_r, color='r', label='Red')
plt.plot(combined_hist_g, color='g', label='Green')
plt.plot(combined_hist_b, color='b', label='Blue')
plt.xlim([0, 256])
plt.xlabel('Color Value')
plt.ylabel('Frequency')
plt.title('Combined Color Histograms (RGB)')
plt.legend()
plt.show()
'''
image_array = np.array(image)

# Compute the color histogram
histogram = image.histogram()

# Compute the color mean
color_mean = np.mean(image_array, axis=(0,1))

# Compute the color standard deviation
color_std = np.std(image_array, axis=(0,1))

print("Color Histogram:", histogram)
print("Color Mean:", color_mean)
print("Color Standard Deviation:", color_std)
'''
