from torch.utils.data import Dataset
import numpy as np
import os
import glob
import torch

from .util import z_score, aug_image_heatmap_pair, convert_volume_label_pair_2_img_n_save, est_data_loading_speed

CONFIG = {'SCALE': True,
          'SCALE_RANGE': [0.9, 1.1],
          'ROTATE': True,
          'ROTATE_RANGE': [-10, 10],
          'FLIP': False}


class CMRStackDataset(Dataset):
    def __init__(self, phase, study_ids, data_dir, sigma='HT0.5'):
        if phase not in ['train', 'test']:
            raise ValueError("phase must be either 'train' or 'test'")
        self.phase = phase

        self.samples, self.labels = [], []
        self.slc_cnt = 0
        for sid in study_ids:
            tmp_list = glob.glob(os.path.join(data_dir, sid + '*IMG*.npy'))
            if len(tmp_list) == 0:
                print("Warning: data not found for %s; skipped." % sid)
                continue
            tmp_list.sort()
            self.samples.append(tmp_list)

            tmp_list = glob.glob(os.path.join(data_dir, sid + '*%s*.npy' % sigma))
            tmp_list.sort()
            self.labels.append(tmp_list)

            assert len(self.samples[-1]) == len(self.labels[-1])
            # print("%d slices found for %s" % (len(self.samples[-1]), sid))
            self.slc_cnt += len(self.samples[-1])

        print(phase + ' dataset')
        print("\tTotal number of MRI exams:", len(self.samples))
        print("\tTotal number of slices:", self.slc_cnt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        volume, label = [], []
        for dir_img, dir_map in zip(self.samples[item], self.labels[item]):
            volume.append(np.load(dir_img)[None, ...])
            label.append(np.load(dir_map)[None, ...])
        volume = np.concatenate(volume, axis=0)
        label = np.concatenate(label, axis=0)

        # print("Image volume:", volume.shape, volume.dtype)
        # print("Label volume:", label.shape, label.dtype)

        volume = z_score(volume)  # z-score standardization
        label = 10. * label  # scale for more familiar magnitude during optimization

        # then data augmentation (per slice) if training
        if self.phase == 'train':
            for i, (image, htmap) in enumerate(zip(volume, label)):
                image, htmap = aug_image_heatmap_pair(CONFIG, image, htmap)  # augmentation
                volume[i], label[i] = image, htmap  # put the slice back in volume

        # add a dimension for channel, if necessary
        volume = volume[:, None, :, :]
        if label.ndim < 4:
            label = label[:, None, :, :]

        return torch.from_numpy(volume).float(), torch.from_numpy(label).float(), self.labels[item]


def _est_data_loading_speed(dataset, shuffle, w_max=40, w_min=0, repeats=3):
    est_data_loading_speed(dataset, batch_size=1, shuffle=shuffle, drop_last=False, w_max=w_max, w_min=w_min,
                           repeats=repeats)


if __name__ == '__main__':
    # unit test
    split_file = os.path.join('..', 'prep_data', 'group_5-fold_split_f2.npz')
    data_split = np.load(split_file)
    data_dir = r'/apdcephfs/private_donwei/data/CMR/view_plan_data_SAX_loc'

    trainset = CMRStackDataset('train', data_split['train'], data_dir, 'HT0.5')
    print("%d MRI exams with %d total slice" % (len(trainset.samples), trainset.slc_cnt))

    volume, label, _ = trainset.__getitem__(2)
    convert_volume_label_pair_2_img_n_save(volume.squeeze().numpy(), label.squeeze().numpy(), 'vis_trainset')

    print("Estimating train set loading efficiency..")
    _est_data_loading_speed(trainset, shuffle=True)

    valset = CMRStackDataset('test', data_split['test'], data_dir, 'HT0.5')
    print("%d MRI exams with %d total slice" % (len(valset.samples), valset.slc_cnt))

    volume, label, _ = valset.__getitem__(2)
    convert_volume_label_pair_2_img_n_save(volume.squeeze().numpy(), label.squeeze().numpy(), 'vis_valset')

    print("Estimating test set loading efficiency..")
    _est_data_loading_speed(valset, shuffle=False)
