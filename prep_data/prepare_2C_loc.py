import os
import glob
import numpy as np

from util import trace_to_dcm_dir, process_LAX_localizer, process_LAX_cine, process_SAX_cine, save_heatmap


def prepare_2C_loc(study_dir, save_dirs, width, verbose=True):
    # first dig down to the root directory of data
    dcm_dir, sid, pid, date = trace_to_dcm_dir(study_dir)

    save_name = sid + '-[' + pid + ']-[' + date + ']'

    # process 2C localizer
    info_loc, img_loc = process_LAX_localizer(dcm_dir, 'trufi_2-chamber', -1, save_dirs['loc'], save_name + '-[IMG]',
                                              verbose=verbose)

    # process 3C cine
    heatmap_3C = process_LAX_cine(dcm_dir, 'TF Cine Retro_iPAT LVOT', -1, save_dirs['cine'], save_name + '-[3C]',
                                  info_loc, width, verbose=verbose)

    # process 4C cine
    heatmap_4C = process_LAX_cine(dcm_dir, 'TF Cine Retro_iPAT 4C', -1, save_dirs['cine'], save_name + '-[4C]',
                                  info_loc, width, verbose=verbose)

    # process SA cine (only the basal slice)
    heatmap_SA = process_SAX_cine(dcm_dir, save_dirs['cine'], save_name + '-[SA]', info_loc, width, verbose=verbose)

    for j, alpha in enumerate(width):
        heatmap = np.concatenate((heatmap_3C[j][None, ...], heatmap_4C[j][None, ...], heatmap_SA[j][None, ...]), axis=0)
        save_heatmap(heatmap, img_loc, save_dirs['loc'], save_name, alpha)


if __name__ == '__main__':
    root_dir = r'..\data\origin_sorted'
    exam_dirs = glob.glob(os.path.join(root_dir, '0*'))
    save_dirs = {'loc': r'..\data\view_plan_data_2C_loc',
                 'cine': r'..\data\view_plan_data_cine'}
    for save_dir in save_dirs.values():
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    width = (.5,)

    for study_dir in exam_dirs:
        print("Processing", study_dir)
        prepare_2C_loc(study_dir, save_dirs, width, True)
