import os
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import torchvision
transform_label = transforms.Compose([transforms.ToTensor(),])

transformToPIL = T.ToPILImage()
asdf = torchvision.transforms.Grayscale(num_output_channels=1)
i = 0
for label in os.listdir("train/labels/"):
    i = i+1
    #img_label = Image.open('train/labels/'+ label).convert('L')
    #tsr = transform_label(img_label)
    tsr = torchvision.io.read_image("train/labels/" + label)
    print(tsr.shape)

    a = tsr.unique(return_counts=True)
    print(a)
    tsr2 = (tsr>0.5).float()
    b = tsr2.unique(return_counts=True)
    print(b)
    tsr3 = (tsr>128).float()
    c = tsr3.unique(return_counts=True)
    print(c)
    tsr2 = transformToPIL(tsr2)
    tsr3 = transformToPIL(tsr3)
    tsr2.save('threshold/' + str(i) + "_" + str(label) + '_5.png')
    tsr3.save('threshold/' + str(i) + "_" + str(label) +'_128.png')
    if i == 10:
        break