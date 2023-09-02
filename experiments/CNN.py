import time
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

from experiments.Model import *

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# --------------------------------------------------

# 加载数据并进行预处理

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize([0.485], [0.229])  # 图像归一化
])
# ----------------------------------------

# 设置超参数

image_folder = "experiments/newdata"  # 图像文件夹路径
label_file = "experiments/labels/labels.csv"  # 标签文件路径
batch_size = 20
num_epochs = 20
learning_rate = 0.001

# -------------------------------------------
dataset = SlurryDataset(image_folder, label_file, transform=transform)

# 定义划分训练集的索引
train_indices = list(range(0, 60)) + list(range(80, 140)) + list(range(180, 240))
train_dataset = Subset(dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# 定义划分验证集的索引
test_indices = list(range(60, 80)) + list(range(140, 160)) + list(range(160, 180))
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------
def draw_train_loss(Batch, all_train_loss):
    plt.title("Training Loss", fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batch, all_train_loss, color='green', label='training loss')
    plt.ylim(0, 3000)
    plt.yticks(np.arange(0, 3000, 100))
    plt.legend()
    plt.grid()
    plt.savefig(f"experiments/Results/train{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    plt.close()


def draw_test_result(predictions, ground_truths):
    plt.title("Test Result", fontsize=24)
    x = np.arange(0, 350)
    y = x
    plt.plot(x, y)
    plt.xlabel("ground truths", fontsize=14)
    plt.ylabel("predictions", fontsize=14)
    plt.scatter(ground_truths, predictions, color='green', label='training cost')
    plt.axis('equal')
    plt.grid()
    plt.savefig(f"experiments/Results/val{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    plt.close()


# -----------------------------------------------------
# 训练模型
def train_model(model):
    Batch_num = 0
    Batch = []
    all_train_loss = []

    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Batch_num = Batch_num + 1
            Batch.append(Batch_num)
            all_train_loss.append(float(loss))

            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/8], Loss: {float(loss):.4f}")

    draw_train_loss(Batch, all_train_loss)


# ----------------------------------------------
# 验证模型
def evaluate_model(model):
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            predictions.extend(outputs.tolist())
            ground_truths.extend(labels.tolist())

    predictions = [float(item[0]) for item in predictions]
    ground_truths = [float(item[0]) for item in ground_truths]

    # 输出预测结果和真实标签
    print("预测结果和真实标签")
    print("predictions\t\tground_truths")
    for p, l in zip(predictions, ground_truths):
        print("{:<12.4f}\t{:<12.4f}".format(p, l))

    # 计算均方误差
    mse = mean_squared_error(ground_truths, predictions)
    print("均方误差:Mean Squared Error (MSE):", mse)

    # 计算均根误差
    rmse = np.sqrt(mse)
    print("均根误差:Root Mean Squared Error (RMSE):", rmse)

    # 计算平均绝对误差
    mae = mean_absolute_error(ground_truths, predictions)
    print("平均绝对误差:Mean Absolute Error (MAE):", mae)

    # 计算决定系数
    r2 = r2_score(ground_truths, predictions)
    print("决定系数:R-squared (R2):", r2)

    draw_test_result(predictions, ground_truths)


def model_predict(model, image):
    image = Image.open(image).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        print(outputs.tolist()[0][0])

    return outputs.tolist()[0][0]
