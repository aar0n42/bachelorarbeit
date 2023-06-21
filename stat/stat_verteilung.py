import enum
import numpy as np
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
import numpy as np
import torchvision

plt.figure(figsize=(8, 6),dpi=600)
transform = transforms.Compose([transforms.ToTensor()])
transformToPIL = transforms.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)
mini = 1000
maxx = 0
summe = 0
count = 0


alle = []
for label in os.listdir("train/labels/"):
    tsr = torchvision.io.read_image("train/labels/" + label)
    #tsr = asdf(tsr)
    tsr = (tsr>0.5).float()
    count += 1
    a = tsr.unique(return_counts=True)
    print(a)
    if a[1][0].item() == 307200: #wenn alle pixel 0 sind
        continue
    area = a[1][1].item()
    summe += area
    alle.append(area)
    if area < mini:
        loc = label
        mini = area
    elif area > maxx:
        f = label
        maxx = area


alle.sort()

print(alle)
for i, num in enumerate(alle):
    alle[i] = round(num/307200,3)


print(alle)

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
j = 0
k = 0
l = 0
m = 0
n = 0
o = 0
y = 0
p = 0
for i in alle:
    if i < 0.01:
        a += 1
    elif  i < 0.02:
        b += 1
    elif i < 0.03:
        c += 1
    elif i < 0.04:
        d += 1
    elif i < 0.05:
        e += 1
    elif i < 0.06:
        f += 1
    elif i < 0.07:
        g += 1
    elif i < 0.08:
        h += 1
    elif i < 0.09:
        j += 1
    elif i < 0.1:
        k += 1
    elif i < 0.11:
        l += 1
    elif i < 0.12:
        m += 1
    else:
        y +=1

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(j)
print(k)
print(l)
print(m)
print(n)
print(o)
print(p)
print(y)
print(a+b+c+d+e+f+g+h+j+k+l+m+n+o+p+y)

print(len(alle))


height = [a,b,c,d,e,f,g,h,j,k,l,m,y]
bars = ('<1%', '1≥ <2%', '1≥ <3%', '1,5≥ <4%', '2≥ <5%', '2,5≥ <6%', '3≥ <7%', '3,5≥ <8%', '4≥ <9%', '4,5≥ <10%', '5≥ ≤11%', '5,5≥ <12', '≥12%')
y_pos = np.arange(len(bars))


plt.bar(y_pos, height)

plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.show()


