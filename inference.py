import argparse
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from dataset.cmr_slice_dataset import CMRSliceDataset
from dataset.cmr_stack_dataset import CMRStackDataset
from network.UNet import UNet
from network.HGNet import HGNet
from utils.util import arr_to_8bit_img


def save_results(dirs, inputs, outputs, losses, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for list_single_dir, img, pred, loss in zip(dirs, inputs, outputs, losses):
        img_name = os.path.basename(list_single_dir)[:-4]
        np.save(os.path.join(save_dir, img_name + '-[%d].npy' % loss.sum()), pred)

        img = arr_to_8bit_img(img[0])
        pred = arr_to_8bit_img(pred.max(axis=0))
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        pred = cv2.addWeighted(src1=np.tile(img[..., None], [1, 1, 3]), alpha=1., src2=pred, beta=.2, gamma=0)

        cv2.imwrite(os.path.join(save_dir, img_name + '-[%d].png' % loss.sum()), pred,  # save for visual debug
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def arg_parser():
    parser = argparse.ArgumentParser(description='Predict orientation heatmap')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('--root', default=r'data/', type=str, help='directory to the image data and ground truth labels')
    parser.add_argument('--view', default='', type=str,
                        help='base view for prediction (default: empty); allowed views: 2C, 4C, and SAX')
    parser.add_argument('--type', default='HT0.5', type=str,
                        help='type of ground truth label to use, i.e., sigma of the Gaussian (default: HT0.5)')
    parser.add_argument('--split', default=r'prep_data/dummy_split.npz', type=str,
                        help='npz file containing split of train and test data')
    parser.add_argument('--part', default='test', type=str, help='if predict train or test (default) data')
    parser.add_argument('--net', default='HGNet', type=str, help='network architecture; options: UNet/HGNet')
    parser.add_argument('--n_stacks', default=2, type=int, help='number of stacks for hourglass network')
    parser.add_argument('--prev', default='', type=str, help='directory to previously trained models (default: empty)')
    return parser


def main(args):
    print("Args: {}".format(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Currently using GPU {}".format(args.gpu))

    # view-dependent variables
    if args.view in ('2C', '4C'):
        cmr_dataset = CMRSliceDataset
    else:
        cmr_dataset = CMRStackDataset
    n_cls = 1 if args.view == 'ax' else 3

    print('==> Preparing data..')
    data_split = np.load(args.split)
    test_set = cmr_dataset('test', data_split[args.part], os.path.join(args.root, 'view_plan_data_%s_loc' % args.view),
                           args.type)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=6, shuffle=False, pin_memory=True)

    print('==> Building inference network..')
    if args.net == 'UNet':
        net = UNet(n_classes=n_cls).cuda()
    else:
        net = HGNet(n_stacks=args.n_stacks, n_classes=n_cls).cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    print("\tLoading existing model parameters in:", args.prev)
    net.load_state_dict(torch.load(args.prev))
    net.eval()

    criterion = torch.nn.MSELoss(reduction='none').cuda()  # loss

    print('==> Inferring..')
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(test_loader):
            print("No.", i, paths[0])

            if args.view in ('2C', '4C'):
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs[0].cuda(), labels[0].cuda()
                paths = [path[0] for path in paths]

            outputs, _ = net(inputs)
            if args.net == 'HGNet':
                outputs = outputs[..., -1]
            loss = criterion(outputs, labels)

            save_results(paths, inputs.cpu().numpy(), outputs.cpu().numpy(), loss.cpu().data.numpy(),
                         os.path.join(args.root, 'view_plan_pred_%s_loc' % args.view))


if __name__ == '__main__':
    main(arg_parser().parse_args())
