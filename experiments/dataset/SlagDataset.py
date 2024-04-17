import os

import pandas as pd
import torch

from torch.utils.data import Dataset
from PIL import Image


# 数据集类
class SlagDataset(Dataset):
    def __init__(self, image_path, label_path=None, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

        # 获取图片ID集合
        self.img_ids = os.listdir(image_path)
        # 获取标签DF
        if self.label_path:
            self.label_df = pd.read_csv(label_path)
            self.label_df.set_index('Image_Id', inplace=True)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 获取图片的ID
        img_id = self.img_ids[idx]
        # 读取图片
        image = Image.open(os.path.join(self.image_path, img_id))
        # 图片处理
        if self.transform:
            image = self.transform(image)

        # 标签处理
        if self.label_path:
            # 分离文件扩展名
            _img = img_id.split('.')[0]  # _img = os.path.splitext(img_id)[0]
            # 在df中找到 _img 对应的标签
            label = self.label_df.loc[_img, 'SF']
            # 转化成张量
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        else:
            label = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)

        return image, label
