import os
import glob
import numpy as np
from math import pi
import cv2
from pydicom import dcmread
import pickle

import sys
sys.path.append("..")
from utils.geometry import plane_intersect, project_line_to_plane, distance_map_to_line
from utils.util import EPS, overlay_heatmap_on_image


def sort_stack(list_stack):
    slice_locations = []
    IOP0 = None
    for file_slice in list_stack:
        # load slice file
        ds = dcmread(file_slice, specific_tags=['ImagePositionPatient', 'ImageOrientationPatient'])
        IPP = np.array(ds.ImagePositionPatient)
        IOP = np.array(ds.ImageOrientationPatient)
        if IOP0 is None:
            IOP0 = IOP
        else:
            assert np.abs((IOP - IOP0)).mean() < EPS, "Inconsistent IOP in axial stack."
        slice_locations.append(np.dot(IPP, np.cross(IOP[0:3], IOP[3:])))
    indices = np.argsort(slice_locations)
    return np.array(list_stack)[indices]


def handle_cine_frame(list_cine, ind, save_dir, save_name, info_loc, width, verbose=True):
    ds = dcmread(list_cine[ind])  # load the designated frame of the cine sequence

    # extract DICOM info
    info_cine = extract_dcm_info(ds)
    target_file = os.path.join(save_dir, save_name + '.pkl')
    if not os.path.exists(target_file):
        with open(target_file, 'wb') as f:
            pickle.dump(info_cine, f, pickle.HIGHEST_PROTOCOL)  # save to disc
        # save image array for visual debug
        save_array_to_image(ds.pixel_array, os.path.join(save_dir, save_name + '.png'))

    return gen_heatmap(info_loc, info_cine, width, verbose=verbose)


def process_SAX_cine(dcm_dir, save_dir, save_name, info_loc, width, verbose=True):
    # locate SAX cine (only the basal slice)
    list_SAX_cine = locate_cine_basal_SA(dcm_dir)
    if verbose:
        print("Found basal SAX cine sequence:", list_SAX_cine)

    # handle the first frame of the SAX cine sequence
    return handle_cine_frame(list_SAX_cine, 0, save_dir, save_name, info_loc, width, verbose=verbose)


def process_LAX_cine(dcm_dir, view_str, ind, save_dir, save_name, info_loc, width, verbose=True):
    # locate LAX cine
    list_LAX_cine = locate_LAX_cine(dcm_dir, view_str, ind)
    if verbose:
        print("Found %s cine sequence: %s" % (view_str, list_LAX_cine))

    # handle the first frame of the LAX cine sequence
    return handle_cine_frame(list_LAX_cine, 0, save_dir, save_name, info_loc, width, verbose=verbose)


def handle_localizer_slice(file_loc, save_dir, save_name):
    ds = dcmread(file_loc)  # load localizer

    # extract DICOM info
    info_loc = extract_dcm_info(ds)
    with open(os.path.join(save_dir, save_name + '.pkl'), 'wb') as f:
        pickle.dump(info_loc, f, pickle.HIGHEST_PROTOCOL)  # save to disc

    # extract image array
    img_loc = ds.pixel_array
    np.save(os.path.join(save_dir, save_name + '.npy'), img_loc)  # save for training
    img_loc = save_array_to_image(img_loc, os.path.join(save_dir, save_name + '.png'))  # save for visual debug

    return info_loc, img_loc


def process_LAX_localizer(dcm_dir, view, ind, save_dir, save_name, verbose=True):
    # locate the LAX localizer
    file_LAX_loc = locate_LAX_localizer(dcm_dir, view, ind)
    if verbose:
        print("Found %s localizer: %s" % (view, file_LAX_loc))

    return handle_localizer_slice(file_LAX_loc, save_dir, save_name)


def gen_heatmap(info_host, info_guest, widths, verbose=True):
    # find the intersection line
    [P0, N, flag] = plane_intersect(info_host['INP'], info_host['IPP'], info_guest['INP'], info_guest['IPP'])
    assert flag == 2, "The two planes either coincide or are in parallel."
    if verbose:
        print("Intersection line passes ", P0, " in direction ", N)

    # project the intersection line into the 2C localizer view plane
    A, B, C = project_line_to_plane(P0, N, info_host['IOP'], info_host['IPP'], info_host['PixelSpacing'])

    # compute distance map to the intersection line in the plane
    D = distance_map_to_line(A, B, C, (info_host['Rows'], info_host['Columns']))

    # compute the heatmaps
    assert np.abs(info_host['PixelSpacing'][0] - info_host['PixelSpacing'][1]) < EPS, \
        "Pixels in the dark-blood axial image is not isotropic."
    heatmaps = []
    for alpha in widths:
        sigma = alpha * float(info_guest['SliceThickness']) / info_host['PixelSpacing'][0]
        heatmap = np.exp(-D ** 2 / 2. / sigma ** 2) / sigma / np.sqrt(2. * pi)
        heatmaps.append(heatmap)

    return heatmaps


def save_heatmap(heatmap, image, save_dir, save_name, alpha):
    np.save(os.path.join(save_dir, save_name + '-[HT%.1f].npy' % alpha), heatmap)  # save for training
    overlay = overlay_heatmap_on_image(heatmap, image)
    cv2.imwrite(os.path.join(save_dir, save_name + '-[HT%.1f].png' % alpha), overlay,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # save for visual debug


def save_array_to_image(arr, path):
    arr = (arr.astype(np.float) - arr.min()) / (arr.max() - arr.min())
    arr = (255 * arr).astype(np.uint8)
    cv2.imwrite(path, arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return arr


def locate_cine_basal_SA(dcm_dir):
    dirs_SA = glob.glob(os.path.join(dcm_dir, '*TF Cine Retro_iPAT SAO'))
    assert len(dirs_SA) > 0, "No SAX cine sequence is found!"
    dirs_SA.sort()
    dir_basal = dirs_SA[0]  # The first SAX slice is supposed to be the basal.

    list_cine = glob.glob(os.path.join(dir_basal, '*'))
    assert len(list_cine) > 0, "Sth wrong with identifying slices in the basal SAX cine sequence."

    return list_cine


def locate_LAX_cine(dcm_dir, view, ind=-1):
    dir_cine = glob.glob(os.path.join(dcm_dir, '*%s' % view))
    assert len(dir_cine) > 0, "No %s sequence is found!" % view
    if len(dir_cine) > 1:
        print("\tWARNING: %d %s sequence found; using the designated (%d)." % (len(dir_cine), view, ind))
    dir_cine = dir_cine[ind]

    list_cine = glob.glob(os.path.join(dir_cine, '*'))
    assert len(list_cine) > 0, "Sth wrong with identifying slices in the %s sequence." % view

    return list_cine


def extract_dcm_info(ds):
    info = {'Rows': int(ds.Rows),
            'Columns': int(ds.Columns),
            'IPP': np.array(ds.ImagePositionPatient),
            'IOP': np.array(ds.ImageOrientationPatient),
            'PixelSpacing': np.array(ds.PixelSpacing),
            'SliceThickness': float(ds.SliceThickness)}
    info['INP'] = np.cross(info['IOP'][0:3], info['IOP'][3:])
    return info


def locate_LAX_localizer(dcm_dir, view, ind=-1):
    dir_LAX = glob.glob(os.path.join(dcm_dir, '*' + view))
    assert len(dir_LAX) > 0, "No %s sequence is found!" % view
    if len(dir_LAX) > 1:
        print("\tWARNING: %d %s localizer sequence found; using the designated (%d)." % (len(dir_LAX), view, ind))
    dir_LAX = dir_LAX[ind]

    list_LAX = glob.glob(os.path.join(dir_LAX, '*'))
    assert len(list_LAX) == 1, "Sth wrong with identifying the %s localizer." % view

    return list_LAX[0]


def trace_to_dcm_dir(study_dir):
    sid = study_dir.split(os.sep)[-1]

    pid_dir = glob.glob(os.path.join(study_dir, '*'))
    assert len(pid_dir) == 1, "Sth wrong with extracting PID."
    pid_dir = pid_dir[-1]
    pid = pid_dir.split(os.sep)[-1]

    dcm_dir = glob.glob(os.path.join(pid_dir, '*'))
    assert len(dcm_dir) == 1, "Sth wrong with extracting study date."
    dcm_dir = dcm_dir[-1]
    date = dcm_dir.split(os.sep)[-1]

    return dcm_dir, sid, pid, date
