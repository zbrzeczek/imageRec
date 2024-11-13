
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import glob
import json
import os 
from PIL import Image

import utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()


DATA_LABELS = ["buildings", "forest", "glacier", "mountain", "sea", "street"] 
    
pre_trans = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#klasa dla naszego dataset
class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []
        
        #tworzymy ten nasz zbior przy inicjalizacji 
        for l_idx, label in enumerate(DATA_LABELS):
            label_dir = os.path.join(data_dir, label)
            data_paths = glob.glob(data_dir + label + '/*.jpg', recursive=True)

            if not data_paths:
                print(f"Warning: No images found in {label_dir}")

            for path in data_paths:
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)  # Use tv_io to read image

                # Ensure the image is of float32 type before applying transforms
                img = img.float()  # Convert to float32

                # Apply the transformations (resize, tensor, normalize) and move to the device
                img = pre_trans(img).to(device)
                self.imgs.append(img)
                self.labels.append(torch.tensor(l_idx).to(device)) 

        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {data_dir}")

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)
    

#tutaj stworzenie wlasnego modeklu (wcale nie poosilkowane z neta ael costam czaje)
n_classes = 24
kernel_size = 3
flattened_img_size = 75 * 3 * 3
IMG_CHS = 1

model = nn.Sequential(
    # First convolution
    nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),  # 25 x 28 x 28
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
    # Second convolution
    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 x 14 x 14
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.MaxPool2d(2, stride=2),  # 50 x 7 x 7
    # Third convolution
    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 x 7 x 7
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 75 x 3 x 3
    # Flatten to Dense
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, n_classes)
)


#stwarzamy loss function/ crossentropy bo categorisation problem 
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
model = torch.compile(model.to(device))

#IMG_WIDTH, IMG_HEIGHT = (150, 150)


#batch size
n = 32

train_path = "data/train/"
#stworzenie dataset train
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)


valid_path = "data/test/"
#stworzenie dataset valid
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n, shuffle=False)
valid_N = len(valid_loader.dataset)


epochs = 10

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(model, train_loader, train_N, optimizer, loss_function)
    utils.validate(model, valid_loader, valid_N, loss_function)