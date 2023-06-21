import enum
import numpy as np
import cv2 as cv
from PIL import Image
import cv2
import numpy
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from collections import Counter

transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)

stat = []
for label in os.listdir("train/labels/"):
    image = cv2.imread('train/labels/' + label)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    topLeft = image[0:cY, 0:cX]
    topRight = image[0:cY, cX:w]
    bottomLeft = image[cY:h, 0:cX]
    bottomRight = image[cY:h, cX:w]
    try:
        tl = torch.from_numpy(topLeft).unique(return_counts=True)[1][1]
    except IndexError:
        tl = 0

    try:
        tr = torch.from_numpy(topRight).unique(return_counts=True)[1][1]
    except IndexError:
        tr = 0

    try:
        bl = torch.from_numpy(bottomLeft).unique(return_counts=True)[1][1]
    except IndexError:
        bl = 0

    try:
        br = torch.from_numpy(bottomRight).unique(return_counts=True)[1][1]
    except IndexError:
        br = 0

    stats = {}
    stats['Top left'] = tl
    stats['Top right'] = tr
    stats['Bottom left'] = bl
    stats['Bottom right'] = br
    stat.append(max(stats, key=stats.get))

print(stat)
print(len(stat))

print(Counter(stat))
a = dict(Counter(stat))

labels = []
sizes = []

for x, y in a.items():
    labels.append(x)
    sizes.append(y)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d} ({p:.2f}%)'.format(p=pct,v=val)
    return my_autopct

colours = {'Top left': '#90f1ef',
           'Top right': '#ffd6e0',
           'Bottom left': '#c1fba4',
           'Bottom right': '#7bf1a8'}

# Plot

plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), colors=[colours[key] for key in labels],startangle=0,counterclock=False)

plt.axis('equal')
plt.show()
"""
cv2.imshow("Top Left Corner", topLeft)
cv2.imshow("Top Right Corner", topRight)
cv2.imshow("Bottom Right Corner", bottomLeft)
cv2.imshow("Bottom Left Corner", bottomRight)
cv2.waitKey(0)
"""
