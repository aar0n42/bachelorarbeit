import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

dataset_path = 'DFUC2022_/train/images'

to_tensor_transform = ToTensor()

image_files = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

mean = np.zeros(3)
std = np.zeros(3)
num_images = len(image_files)

for image_file in image_files:

    image = Image.open(image_file)
    
    image_tensor = to_tensor_transform(image)

    mean += image_tensor.mean(axis=[1, 2]).numpy()
    std += image_tensor.std(axis=[1, 2]).numpy()

mean /= num_images
std /= num_images

print(f'Mean: {mean}')
print(f'Standard Deviation: {std}')
