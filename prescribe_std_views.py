import numpy as np
import os
import glob
import pickle
import cv2

from prescribe_plane.handle_prediction import extract_LAX_prediction, extract_stack_prediction
from prescribe_plane.prescribe_cmr_plane import search_LAX_cine_plane, search_LVOT_cine_plane, search_SAX_cine_plane
from prescribe_plane.util import locate_intersection_line_in_view, compute_metrics
from utils.util import EPS, arr_to_8bit_img


def prescribe_std_views(root_dir, sid, sigma, save_dir):
    # extract prediction in LAX localizers
    pred_2C, info_2C, file_2C_img = extract_LAX_prediction(root_dir, '2C', sid, sigma)
    pred_4C, info_4C, file_4C_img = extract_LAX_prediction(root_dir, '4C', sid, sigma)

    if np.abs(np.dot(info_2C['INP'], info_4C['INP'])) < EPS:
        print("\t2C loc and 4C loc are orthogonal!")

    # extract prediction in SAX localizers
    pred_SA, info_SA, list_SA_img = extract_stack_prediction(root_dir, 'SAX_loc', sid, sigma)

    if np.abs(np.dot(info_2C['INP'], info_SA[0]['INP'])) < EPS:
        print("\t2C loc and SAX loc are orthogonal!")

    if np.abs(np.dot(info_4C['INP'], info_SA[0]['INP'])) < EPS:
        print("\t4C loc and SAX loc are orthogonal!")

    print("Prescribing 2C cine..")
    IPP_2C, INP_2C, _ = search_LAX_cine_plane(info_SA, [pred[0] for pred in pred_SA], info_4C, pred_4C[0])
    print("Prescribing 3C cine..")
    IPP_3C, INP_3C, _ = search_LVOT_cine_plane(info_SA, [pred[1] for pred in pred_SA], info_2C, pred_2C[0],
                                               info_4C, pred_4C[1])
    print("Prescribing 4C cine..")
    IPP_4C, INP_4C, _ = search_LAX_cine_plane(info_SA, [pred[2] for pred in pred_SA], info_2C, pred_2C[1])
    print("Prescribing SAX cine..")
    IPP_SAX, INP_SAX, _ = search_SAX_cine_plane(info_4C, pred_4C[-1], info_2C, pred_2C[-1])

    print("Locating, loading, and comparing to ground truth..")
    list_2C_cine = glob.glob(os.path.join(root_dir, 'view_plan_data_cine', sid + '*2C*.pkl'))
    assert len(list_2C_cine) == 1, "Sth wrong with locating 2C cine info for %s!" % sid
    with open(list_2C_cine[0], 'rb') as f:
        info_2C_cine = pickle.load(f)
    d, theta = compute_metrics(info_2C_cine, IPP_2C, INP_2C)
    print("For cine 2C, normal deviation: %.2f degree and point-to-plane distance: %.2f mm" % (theta, d))

    list_3C_cine = glob.glob(os.path.join(root_dir, 'view_plan_data_cine', sid + '*[3C*.pkl'))
    assert len(list_3C_cine) == 1, "Sth wrong with locating LVOT cine info for %s!" % sid
    with open(list_3C_cine[0], 'rb') as f:
        info_3C_cine = pickle.load(f)
    d, theta = compute_metrics(info_3C_cine, IPP_3C, INP_3C)
    print("For cine 3C, normal deviation: %.2f degree and point-to-plane distance: %.2f mm" % (theta, d))

    list_4C_cine = glob.glob(os.path.join(root_dir, 'view_plan_data_cine', sid + '*4C*.pkl'))
    assert len(list_4C_cine) == 1, "Sth wrong with locating 2C cine info for %s!" % sid
    with open(list_4C_cine[0], 'rb') as f:
        info_4C_cine = pickle.load(f)
    d, theta = compute_metrics(info_4C_cine, IPP_4C, INP_4C)
    print("For cine 4C, normal deviation: %.2f degree and point-to-plane distance: %.2f mm" % (theta, d))

    list_SAX_cine = glob.glob(os.path.join(root_dir, 'view_plan_data_cine', sid + '*SA*.pkl'))
    assert len(list_SAX_cine) == 1, "Sth wrong with locating SAX cine info for %s!" % sid
    with open(list_SAX_cine[0], 'rb') as f:
        info_SAX_cine = pickle.load(f)
    d, theta = compute_metrics(info_SAX_cine, IPP_SAX, INP_SAX)
    print("For cine SAX, normal deviation: %.2f degree and point-to-plane distance: %.2f mm" % (theta, d))

    print("Saving visualizations..")
    visualize_a_loc_view((file_2C_img,), (info_2C,), (INP_4C, INP_SAX), (IPP_4C, IPP_SAX),
                         (info_4C_cine, info_SAX_cine), save_dir, '2C')
    visualize_a_loc_view((file_4C_img,), (info_4C,), (INP_2C, INP_SAX), (IPP_2C, IPP_SAX),
                         (info_2C_cine, info_SAX_cine), save_dir, '4C')
    visualize_a_loc_view(list_SA_img, info_SA, (INP_2C, INP_3C, INP_4C), (IPP_2C, IPP_3C, IPP_4C),
                         (info_2C_cine, info_3C_cine, info_4C_cine), save_dir, 'SAX')

    print("Done!")


def visualize_a_loc_view(list_img, list_info, INPs, IPPs, info_GTs, save_dir, loc_view):
    for i, (img_file, img_info) in enumerate(zip(list_img, list_info)):
        img = np.load(img_file)
        img = arr_to_8bit_img(img)
        # channel sequence below is different from the final image, but convenient for operation
        img = np.tile(img[None, ...], [3, 1, 1])

        for (INP, IPP, info_gt) in zip(INPs, IPPs, info_GTs):
            intersect = locate_intersection_line_in_view(INP, IPP, img_info)
            img[2][intersect] = 255
            intersect = locate_intersection_line_in_view(info_gt['INP'], info_gt['IPP'], img_info)
            img[1][intersect] = 255

        img = img.transpose((1, 2, 0))
        # save visualization
        cv2.imwrite(os.path.join(save_dir, loc_view + '-%s.png' % i), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    root_dir = r'data/'
    sigma = 'HT0.5'
    sid = '056B'
    print("Processing %s" % sid)
    save_dir = r'data/prescribe_std_views_%s' % sid
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    prescribe_std_views(root_dir, sid, sigma, save_dir)
