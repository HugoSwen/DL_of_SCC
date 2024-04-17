import torch
from torchvision.transforms import transforms

from experiments.dataset.SlurryDataset import SlurryDataset
from torch.utils.data import DataLoader, Subset


def getstat(dataset):
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros(1)  # 因为我的数据集是单通道的，只包含目标（1）和背景（0），所以我只需要计算一个通道的mean和std
    std = torch.zeros(1)
    for x, _ in loader:  # 计算loader中所有数据的mean和atd的累积
        mean += x.mean()
        std += x.std()
    mean = torch.div(mean, len(dataset))  # 得到整体数据集mean的平均
    std = torch.div(std, len(dataset))
    return list(mean.numpy()), list(std.numpy())  # 返回mean和std的list


transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
])

image_folder = "newData"  # 图像文件夹路径
label_file = "../experiments/labels/slurryLabels.csv"  # 标签文件路径
dataset = SlurryDataset(image_folder, label_file, transform)

# 定义划分训练集的索引
train_indices = list(range(0, 60)) + list(range(80, 140)) + list(range(160, 220))
train_dataset = Subset(dataset, train_indices)

# 定义划分验证集的索引
valid_indices = list(range(60, 80)) + list(range(140, 160)) + list(range(220, 240))
valid_dataset = Subset(dataset, valid_indices)

mean, std = getstat(train_dataset)  # 调用getstat
mean_, std_ = getstat(valid_dataset)
print(mean, std)
print(mean_, std_)
