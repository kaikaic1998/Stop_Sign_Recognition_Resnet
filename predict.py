import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import cv2

import torch
from torchvision import transforms, models
import torch.nn as nn

import os

def imshow_(inp, title=None):
    """Imshow for Tensor."""
    inp = inp .permute(1, 2, 0).numpy() 
    print(inp.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
    plt.show()
    

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load( "20230512_resnet18_model_30epochs.pt"))
model.eval()

os.chdir('./test_set_stop_not_stop')
imageNames = ['stop_1.jpeg','stop_2.jpeg','stop_3.jpeg','stop_4.jpeg',
              'not_stop_1.jpeg','not_stop_2.jpeg','not_stop_3.jpeg','not_stop_4.jpeg',
              'stop_sign_online1.jpg', 'stop_sign_online2.jpg']
imageNames = []

for imageName in imageNames:
    image = Image.open(imageName)
    transform = composed = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(image)
    z=model(x.unsqueeze_(0))
    _,yhat=torch.max(z.data, 1)
    # print(yhat)
    prediction = "Not Stop"
    if yhat == 1:
        prediction ="Stop"
    imshow_(transform(image),imageName+": Prediction = "+prediction)