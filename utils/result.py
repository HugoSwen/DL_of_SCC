import time

import numpy as np
from matplotlib import pyplot as plt


def draw_train_loss(Batch, all_train_loss):
    plt.title("Training Loss", fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batch, all_train_loss, color='green', label='training loss')
    plt.ylim(0, 3000)
    plt.yticks(np.arange(0, 3000, 100))
    plt.legend()
    plt.grid()
    plt.savefig(f"Results/train{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    plt.close()


def draw_valid_result(predictions, ground_truths):
    plt.title("Valid Result", fontsize=24)
    x = np.arange(0, 350)
    y = x
    plt.plot(x, y)
    plt.xlabel("ground truths", fontsize=14)
    plt.ylabel("predictions", fontsize=14)
    plt.scatter(ground_truths, predictions, color='green')
    plt.axis('equal')
    plt.grid()
    # plt.savefig(f"Results/val{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    plt.close()
