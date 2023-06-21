from torchvision import transforms
import os
from PIL import Image
import random

#wenn nur zahl dann wird von 1-zahl bis 1+zahl ein faktor bestimmt
#wenn tupel dann zb (0.4,0.4) wird 0,4 genommen
augmentation = transforms.ColorJitter(brightness=(0.6,0.6), contrast=0, saturation=0, hue=0) # transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0))

trans =  transforms.ToTensor()
back = transforms.ToPILImage()


path = '../DFUC2022_/train/images/'
dir = os.listdir('../DFUC2022_/train/images')

rand = random.randint(0,len(dir))
rand = 145
print(rand)
img_org = Image.open(path + dir[rand])
img = trans(img_org)

img_aug = augmentation(img)

img_aug = back(img_aug)

img_aug.show()

img_org.show()