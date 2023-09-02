import os

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from PIL import Image


# 数据集类
class SlurryDataset(Dataset):
    def __init__(self, image_folder, label_file=None, transform=None):
        self.image_folder = image_folder
        self.label_df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = f"image{int(self.label_df.iloc[idx, 0])}.jpg"
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.label_df.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32).view(1)  # 调整标签的维度

        return image, label


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 80 * 60, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        # 将特征图展平成向量
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
