import os
import glob
import numpy as np

from util import trace_to_dcm_dir, handle_localizer_slice, process_LAX_cine, sort_stack, save_heatmap


def prepare_SAX_loc(study_dir, save_dirs, width, verbose=True):
    # first dig down to the root directory of data
    dcm_dir, sid, pid, date = trace_to_dcm_dir(study_dir)

    save_name = sid + '-[' + pid + ']-[' + date + ']'

    # locate the SAX localizer
    dir_SAX = glob.glob(os.path.join(dcm_dir, '*trufi_short-axis'))
    assert len(dir_SAX) > 0, "No SAX localizer sequence is found!"
    if len(dir_SAX) > 1:
        print("\tWARNING: more than one SAX localizer sequence found; using the last one.")
    dir_SAX = dir_SAX[-1]

    list_SAX = glob.glob(os.path.join(dir_SAX, '*'))
    assert len(list_SAX) > 0, "Sth wrong with identifying the SAX localizer."

    # sort the SAX loc stack
    list_SAX = sort_stack(list_SAX)[::-1]

    for ind, file_SAX in enumerate(list_SAX):
        info_loc, img_loc = handle_localizer_slice(file_SAX, save_dirs['loc'], save_name + '-[%02d]-[IMG]' % ind)

        # process 2C cine
        heatmap_2C = process_LAX_cine(dcm_dir, 'TF Cine Retro_iPAT 2C', -1, save_dirs['cine'], save_name + '-[2C]',
                                      info_loc, width, verbose=verbose)

        # process 3C cine
        heatmap_3C = process_LAX_cine(dcm_dir, 'TF Cine Retro_iPAT LVOT', -1, save_dirs['cine'], save_name + '-[3C]',
                                      info_loc, width, verbose=verbose)

        # process 4C cine
        heatmap_4C = process_LAX_cine(dcm_dir, 'TF Cine Retro_iPAT 4C', -1, save_dirs['cine'], save_name + '-[4C]',
                                      info_loc, width, verbose=verbose)

        for j, alpha in enumerate(width):
            heatmap = np.concatenate((heatmap_2C[j][None, ...], heatmap_3C[j][None, ...], heatmap_4C[j][None, ...]),
                                     axis=0)
            save_heatmap(heatmap, img_loc, save_dirs['loc'], save_name + '-[%02d]' % ind, alpha)


if __name__ == '__main__':
    root_dir = r'..\data\origin_sorted'
    exam_dirs = glob.glob(os.path.join(root_dir, '0*'))
    save_dirs = {'loc': r'..\data\view_plan_data_SAX_loc',
                 'cine': r'..\data\view_plan_data_cine'}
    for save_dir in save_dirs.values():
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    width = (.5,)

    for study_dir in exam_dirs:
        print("Processing", study_dir)
        prepare_SAX_loc(study_dir, save_dirs, width, True)
