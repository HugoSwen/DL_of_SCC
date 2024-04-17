import os
import random
import shutil

# 原始图片文件夹路径
original_folder = '../experiments/newData'

# 创建目标文件夹，用于存放随机抽取的图片
train_folder = '../experiments/train'
val_folder = '../experiments/val'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 获取原始文件夹中所有图片文件的列表
all_images = os.listdir(original_folder)

# 设置要拆分的图片数量
num_train_images = 192
num_val_images = 48

# 随机抽取训练集图片
train_images = random.sample(all_images, num_train_images)

# 从剩余图片中随机抽取验证集图片
val_images = set(all_images) - set(train_images)

# 将训练集图片复制到目标文件夹
for image in train_images:
    original_path = os.path.join(original_folder, image)
    target_path = os.path.join(train_folder, image)
    shutil.copy(original_path, target_path)

# 将验证集图片复制到目标文件夹
for image in val_images:
    original_path = os.path.join(original_folder, image)
    target_path = os.path.join(val_folder, image)
    shutil.copy(original_path, target_path)

print(f'Successfully split the dataset into {num_train_images} training images and {num_val_images} validation images.')
