from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
import random
import torchvision.transforms.functional as TF
from torchvision import transforms as TU


class UlcerData(Dataset):
    def __init__(self, imagePath, maskPath, transforms, transform2):
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.transforms = transforms
        self.transform2 = transform2
        self.all_images = sorted(os.listdir(imagePath))
        self.all_labels = sorted(os.listdir(maskPath))
        self.norm = TU.Normalize((0.626, 0.573, 0.551), (0.155, 0.181, 0.193))
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.imagePath, self.all_images[idx])
        label_loc = os.path.join(self.maskPath, self.all_labels[idx])
        image = Image.open(img_loc)
        label = Image.open(label_loc).convert('L')
        tsr = self.transforms(image)
        tsr2 = self.transform2(label)
        tsr2 = (tsr2>0).float()
        if random.uniform(0.0, 1.0) > 0.5:
                tsr = TF.hflip(tsr)
                tsr2 = TF.hflip(tsr2)
        if random.uniform(0.0, 1.0) > 0.5:
                tsr = TF.vflip(tsr)
                tsr2 = TF.vflip(tsr2)
        angle = random.uniform(-180.0, 180.0)
        h_trans = random.uniform(-352 / 8, 352 / 8)
        v_trans = random.uniform(-352 / 8, 352 / 8)
        scale = random.uniform(0.5, 1.5)
        shear = random.uniform(-22.5, 22.5)
        tsr = TF.affine(tsr, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
        tsr2 = TF.affine(tsr2, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        tsr = self.norm(tsr)
        return (tsr,tsr2)
