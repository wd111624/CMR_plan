from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torch


from .util import z_score, aug_image_heatmap_pair, convert_image_heatmap_pair_to_img_n_save, est_data_loading_speed

CONFIG = {'SCALE': True,
          'SCALE_RANGE': [0.9, 1.1],
          'ROTATE': True,
          'ROTATE_RANGE': [-10, 10],
          'FLIP': True}


class CMRSliceDataset(Dataset):
    def __init__(self, phase, study_ids, data_dir, sigma='HT0.5'):
        if phase not in ['train', 'test']:
            raise ValueError("phase must be either 'train' or 'test'")
        self.phase = phase

        samples, labels = [], []
        for sid in study_ids:
            tmp_list = glob.glob(os.path.join(data_dir, sid + '*IMG*.npy'))
            if len(tmp_list) == 0:
                print("Warning: data not found for %s; skipped." % sid)
                continue
            assert len(tmp_list) == 1, "More than one slice found for single-slice data!"
            samples.append(tmp_list[0])

            tmp_list = glob.glob(os.path.join(data_dir, sid + '*%s*.npy' % sigma))
            assert len(tmp_list) == 1, "More than one heatmap found for single-slice data!"
            labels.append(tmp_list[0])

        self.samples, self.labels = np.asarray(samples), np.asarray(labels)
        print(phase + ' dataset')
        print("\tTotal number of MRI studies (slices):", len(self.samples))

        self.config = CONFIG
        if '2C' in data_dir:
            self.config['FLIP_WAY'] = 'LR'
            self.height, self.width = 192, 176
        elif '4C' in data_dir:
            self.config['FLIP_WAY'] = 'UD'
            self.height, self.width = 160, 192
        else:
            raise Exception("Unknown view for single-slice CMR localizer!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        image, htmap = np.load(self.samples[item]), np.load(self.labels[item])

        image = z_score(image)  # z-score standardization
        htmap = 10. * htmap  # scale for more familiar magnitude during optimization

        # data augmentation if training
        if self.phase == 'train':
            image, htmap = aug_image_heatmap_pair(self.config, image, htmap, self.height, self.width)

        return torch.from_numpy(image[None, ...]).float(), torch.from_numpy(htmap).float(), self.labels[item]


if __name__ == '__main__':
    # unit test
    split_file = os.path.join('..', 'prep_data', 'group_5-fold_split_f2.npz')
    data_split = np.load(split_file)
    data_dir = r'F:\CMR\view_plan_data_2C_loc'

    train_set = CMRSliceDataset('train', data_split['train'], data_dir, 'HT0.5')
    print("\t%d MRI studies (slices)" % len(train_set))

    for repeat in range(3):
        image, htmap, _ = train_set.__getitem__(0)
        convert_image_heatmap_pair_to_img_n_save(image[0].numpy(), htmap.numpy(), 'train_sample-r%d.png' % repeat)

    print("Estimating train set loading efficiency..")
    est_data_loading_speed(train_set, batch_size=8, shuffle=True, drop_last=True)

    test_set = CMRSliceDataset('test', data_split['test'], data_dir, 'HT0.5')
    print("\t%d MRI studies (slices)" % len(test_set))

    for repeat in range(3):
        image, htmap, _ = test_set.__getitem__(0)
        convert_image_heatmap_pair_to_img_n_save(image[0].numpy(), htmap.numpy(), 'test_sample-r%d.png' % repeat)

    print("Estimating test set loading efficiency..")
    est_data_loading_speed(test_set, batch_size=1, shuffle=False, drop_last=False)
