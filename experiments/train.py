import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torchvision import transforms
from tqdm import tqdm

from experiments.model.model import resnet34

from experiments.dataset.SlurryDataset import SlurryDataset
from utils.result import draw_valid_result


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([  # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.643], [0.353])]),
        "val": transforms.Compose([  # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.669], [0.357])])}

    data_root = os.getcwd()  # get data root path
    image_path = os.path.join(data_root, "slagData")  # flower data set path
    label_path = os.path.join(data_root, "labels", "slagLabels.csv")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_dataset = SlurryDataset(image_path=os.path.join(image_path, "train"),
                                  label_path=label_path,
                                  transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = SlurryDataset(image_path=os.path.join(image_path, "val"),
                                     label_path=label_path,
                                     transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth

    model_weight_path = "model/resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 1)
    net.to(device)

    # define loss function
    loss_function = nn.MSELoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)

    epochs = 10
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        predictions = []
        ground_truths = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predictions.extend(outputs.tolist())
                ground_truths.extend(val_labels.tolist())

            # val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
            #                                            epochs)

        print('[epoch %d] train_loss: %.3f' %
              (epoch + 1, running_loss / train_steps))

        # 输出预测结果和真实标签
        # print("预测结果和真实标签")
        # print("predictions\t\tground_truths")
        # for p, l in zip(predictions, ground_truths):
        #     print("{:<12.4f}\t{:<12.4f}".format(p, l))

        # 计算均方误差
        mse = mean_squared_error(ground_truths, predictions)
        print("均方误差:Mean Squared Error (MSE):", mse)

        # 计算平均绝对误差
        mae = mean_absolute_error(ground_truths, predictions)
        print("平均绝对误差:Mean Absolute Error (MAE):", mae)

        # 计算决定系数
        r2 = r2_score(ground_truths, predictions)
        print("决定系数:R-squared (R2):", r2)

        draw_valid_result(predictions, ground_truths)

        # torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
