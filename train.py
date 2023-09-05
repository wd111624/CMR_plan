import argparse
import os
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import time
import copy
import numpy as np

from dataset.cmr_slice_dataset import CMRSliceDataset
from dataset.cmr_stack_dataset import CMRStackDataset
from network.UNet import UNet
from network.HGNet import HGNet, HGMSELoss

SAVE_FREQ = 5


def arg_parser():
    parser = argparse.ArgumentParser(description='Baseline regressor with train/val split')
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('--train_worker', default=7, type=int,
                        help="number of workers for train data loader (default: 7)")
    parser.add_argument('--test_worker', default=7, type=int,
                        help="number of workers for test data loader (default: 7)")
    parser.add_argument('--root', default=r'data/',
                        type=str, help='directory to the image data and ground truth labels')
    parser.add_argument('--view', default='', type=str,
                        help='view id train and test data (default: ''); allowed views: 2C, 4C, and SAX')
    parser.add_argument('--type', default='HT0.5', type=str,
                        help='type of ground truth label to use, i.e., sigma of the Gaussian (default: HT0.5)')
    parser.add_argument('--split', default=r'prep_data/dummy_split.npz', type=str,
                        help='npz file containing split of train and test data')
    parser.add_argument('--net', default='HGNet', type=str, help='network architecture; options: UNet/HGNet')
    parser.add_argument('--n_stacks', default=2, type=int, help='number of stacks for hourglass network')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer to use (default: Adam)')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--schedule', default='plateau', type=str,
                        help='learning rate scheduler to use (default: ReduceLROnPlateau)')
    parser.add_argument('--gamma', default=0.5, type=float, metavar='GAMMA',
                        help='gamma param in lr scheduler (default: 0.5)')
    parser.add_argument('--max_epoch', default=100, type=int, help="default: 100")
    parser.add_argument('--step_epoch', nargs='+', default=[50, 75], type=int,
                        help="Epochs at which the lr should step down (default: [50, 75])")
    parser.add_argument('--prev', default='', type=str, help='directory to previously trained models (default: empty)')
    return parser


def main(args):
    print("Args: {}".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Currently using GPU {}".format(args.gpu))

    # view-dependent variables
    if args.view in ('2C', '4C'):
        cmr_dataset = CMRSliceDataset
        batch_size = 8
        drop_last = False
        n_cls = 3
    else:
        cmr_dataset = CMRStackDataset
        batch_size = 1
        drop_last = False
        n_cls = 3 if args.view == 'SAX' else 1

    print('==> Preparing data..')
    data_split = np.load(args.split)
    print(data_split['train'])
    train_set = cmr_dataset('train', data_split['train'], args.root + 'view_plan_data_%s_loc' % args.view, args.type)
    val_set = cmr_dataset('test', data_split['test'], args.root + 'view_plan_data_%s_loc' % args.view, args.type)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.train_worker, shuffle=True,
                              pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.test_worker, shuffle=False, pin_memory=True)

    print('==> Building regressing network..')
    if args.net == 'UNet':
        net = UNet(n_classes=n_cls).cuda()
        criterion = torch.nn.MSELoss().cuda()  # loss
    else:
        net = HGNet(n_stacks=args.n_stacks, n_classes=n_cls).cuda()
        criterion = HGMSELoss(n_stacks=args.n_stacks).cuda()  # loss
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    start_epoch = 0
    epoch_record = []
    train_loss = []
    val_loss = []
    best_loss = 1.
    best_epoch = -1
    best_model_wts = None
    if args.prev:
        print("\tResuming from existing model parameters in:", args.prev)
        load_model = torch.load(args.prev)
        net.load_state_dict(load_model['state_dict'])
        start_epoch = load_model['epoch_record'][-1] + 1
        epoch_record = load_model['epoch_record']
        train_loss = load_model['train_loss']
        val_loss = load_model['val_loss']
        best_loss = load_model['best_loss']
        best_epoch = load_model['best_epoch']
        best_model_wts = load_model['best_model_wts']

    if args.optim == 'SGD':
        optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': args.lr}],
                              lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': args.lr}],
                               lr=args.lr, weight_decay=5e-4)

    if args.schedule == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step_epoch, gamma=args.gamma,
                                                   last_epoch=start_epoch - 1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.gamma, verbose=True)

    max_epoch = args.max_epoch
    if not os.path.exists('stats'):
        os.mkdir('stats')
    print('==> Training model..')
    for epoch in range(start_epoch, max_epoch):
        since = time.time()
        epoch_record.append(epoch)
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))

        # print('\tLearning rate:', scheduler.get_lr())

        # training
        net.train()
        running_count = 0
        running_loss = 0.0

        for inputs, labels, _ in train_loader:
            if args.view in ('2C', '4C'):
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs[0].cuda(), labels[0].cuda()

            optimizer.zero_grad()
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # mini-batch stats
            running_count += labels.nelement()
            running_loss += loss.item() * labels.nelement()

        # epoch stats
        epoch_loss = running_loss / running_count
        train_loss.append(epoch_loss)
        print('\tTrain loss: {:.4f}'.format(epoch_loss))

        # validation
        net.eval()
        running_count = 0
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                if args.view in ('2C', '4C'):
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs[0].cuda(), labels[0].cuda()

                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)

                # mini-batch stats
                running_count += labels.nelement()
                running_loss += loss.item() * labels.nelement()

        # epoch stats
        epoch_loss = running_loss / running_count
        val_loss.append(epoch_loss)
        print('\tVal loss: {:.4f}'.format(epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(net.state_dict())

        print('\tBest val loss so far: {:.4f} @Ep-{:d}'.format(best_loss, best_epoch))

        if (epoch + 1) % SAVE_FREQ == 0 or (epoch + 1) == args.max_epoch:
            print("Periodically saving..")
            torch.save({'state_dict': net.state_dict(),
                        'epoch_record': epoch_record,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_loss': best_loss,
                        'best_epoch': best_epoch,
                        'best_model_wts': best_model_wts,
                        }, os.path.join('stats', '%s_Ep-%03d_checkpoint.pth.tar' % (args.type, epoch)))

        elapsed = time.time() - since
        print('Epoch completed in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))

        scheduler.step(epoch_loss)

    print('==> Training completed!')
    print('Best val loss: {:.4f} at epoch No. {:d}'.format(best_loss, best_epoch))

    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(best_model_wts, os.path.join('models', '%s_%s_%s_Ep-%03d_BestModel.pth.tar' %
                                            (args.view, args.type, args.split[-15:-4], best_epoch)))


if __name__ == '__main__':
    main(arg_parser().parse_args())
