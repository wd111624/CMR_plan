import os
import glob
import pickle
import numpy as np
import sys

from .util import locate_peaks_in_heatmap
sys.path.append("..")
from utils.geometry import im2patient


def extract_stack_peaks_2d(list_stack_pred, list_stack_info):
    vals = []
    x, y = [], []
    for pred, info in zip(list_stack_pred, list_stack_info):
        peaks = locate_peaks_in_heatmap(pred)
        x.extend(peaks[:, 1])
        y.extend(peaks[:, 0])
        vals.extend(peaks[:, 2])
    return x, y, vals


def extract_stack_peaks_3d(list_stack_pred, list_stack_info):
    vals = []
    x, y, z = [], [], []
    # go through the stack
    for pred, info in zip(list_stack_pred, list_stack_info):
        peaks = locate_peaks_in_heatmap(pred)
        vals.extend(peaks[:, 2])

        X, Y, Z = im2patient(peaks[:, 1], peaks[:, 0], info['IOP'], info['IPP'], info['PixelSpacing'])
        x.extend(X)
        y.extend(Y)
        z.extend(Z)
    return x, y, z, vals


def extract_stack_prediction(root_dir, view, sid, sigma):
    list_stack_pred = glob.glob(os.path.join(root_dir, 'view_plan_pred_%s' % view, sid + '*%s*.npy' % sigma))
    assert len(list_stack_pred) > 0, "Sth wrong with locating %s stack info for %s!" % (view, sid)
    list_stack_pred.sort()

    list_stack_info = glob.glob(os.path.join(root_dir, 'view_plan_data_%s' % view, sid + '*IMG*.pkl'))
    assert len(list_stack_info) == len(list_stack_pred), "%s info and prediction numbers are different for %s!" % (view, sid)
    list_stack_info.sort()

    pred_stack, info_stack = [], []
    for file_pred, file_info in zip(list_stack_pred, list_stack_info):
        pred_stack.append(np.load(file_pred))
        with open(file_info, 'rb') as f:
            info_stack.append(pickle.load(f))

    list_stack_img = glob.glob(os.path.join(root_dir, 'view_plan_data_%s' % view, sid + '*IMG*.npy'))
    assert len(list_stack_pred) == len(list_stack_img), "%s image and prediction numbers are different for %s!" % (view, sid)
    list_stack_img.sort()

    return pred_stack, info_stack, list_stack_img


def extract_LAX_prediction(root_dir, view, sid, sigma):
    """
    Extract prediction (and relevant information) in LAX localizer
    :return:
    """
    list_LAX_info = glob.glob(os.path.join(root_dir, 'view_plan_data_%s_loc' % view, sid + '*.pkl'))
    assert len(list_LAX_info) == 1, "Sth wrong with locating the %s loc info for %s!" % (view, sid)
    with open(list_LAX_info[0], 'rb') as f:
        info_LAX = pickle.load(f)

    list_LAX_img = glob.glob(os.path.join(root_dir, 'view_plan_data_%s_loc' % view, sid + '*IMG*.npy'))
    assert len(list_LAX_img) == 1, "Sth wrong with locating the %s loc image for %s!" % (view, sid)

    list_LAX_pred = glob.glob(os.path.join(root_dir, 'view_plan_pred_%s_loc' % view, sid + '*%s*.npy' % sigma))
    assert len(list_LAX_pred) == 1, "Sth wrong with locating the %s loc prediction for %s!" % (view, sid)

    pred_LAX = np.load(list_LAX_pred[0])

    return pred_LAX, info_LAX, list_LAX_img[0]


def extract_slice_peaks_3d(pred, info, x, y, z, vals, weight=2):
    peaks = locate_peaks_in_heatmap(pred)
    vals.extend(weight * peaks[:, 2])
    X, Y, Z = im2patient(peaks[:, 1], peaks[:, 0], info['IOP'], info['IPP'], info['PixelSpacing'])
    x.extend(X), y.extend(Y), z.extend(Z)
    return x, y, z, vals
