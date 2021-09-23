import sys
import torch
from visdom import Visdom


def plot(file):
    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    epoch_record = checkpoint['epoch_record']
    train_loss, val_loss = checkpoint['train_loss'], checkpoint['val_loss']

    losses = [[train_loss[i], val_loss[i]] for i in range(len(epoch_record))]

    # 将窗口类实例化
    viz = Visdom()

    # 创建窗口并显示曲线
    viz.line(losses, epoch_record, win='loss',
             opts=dict(title='loss', legend=['train_loss', 'val_loss']))


if __name__ == '__main__':
    file = sys.argv[1]
    plot(file)
