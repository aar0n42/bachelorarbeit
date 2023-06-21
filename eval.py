import os
import numpy as np
import cv2
import torch
from Models import ssformer
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda'
model = ssformer.mit_PLD_b4()
model_state_dict = torch.load('')
model.load_state_dict(model_state_dict)
model.to(device)
print('loaded', flush=True)
model.eval()

folder_path = 'eval'
output_folder = 'eval/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

testfolder = 'test/images'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.626, 0.573, 0.551), (0.155, 0.181, 0.193))
])

allimg = os.listdir(testfolder)
allimg.sort()

tta = False

for file_name in allimg:
    img_path = os.path.join(testfolder, file_name)
    img = Image.open(img_path).convert('RGB')
    
    if tta: 
        img_list = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM), img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT) ]
    else:
        img_list = [img]
        
    predictions = []

    for img in img_list:
        x = transform(img)
        x = x.unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            prediction = model(x)
            prediction = (prediction>0).float()

        predictions.append(prediction.cpu().numpy())

    # Averaging predictions
    avg_prediction = np.mean(predictions, axis=0)
    avg_prediction = np.squeeze(avg_prediction)
    avg_prediction = avg_prediction.astype('uint8')

    output_path = os.path.join(output_folder, f"{file_name.split('.')[0]}.png")
    print(output_path)
    cv2.imwrite(output_path, avg_prediction*255)
