import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import os
import cv2
from skimage.filters import threshold_otsu
import time
from joblib import Parallel, delayed

import sys
sys.path.append('..')
from utils.geometry import distance_map_to_line, im2patient, plane_intersect, project_line_to_plane
from utils.util import arr_to_8bit_img


def compute_plane_param_2d(x, y, vals, ref_info, weight):
    if weight:
        A, B, C, score = regress_line(x, y, vals)
    else:
        A, B, C, score = regress_line(x, y)
    # IOP_012_2d = np.array([B, -A]) / np.sqrt(A ** 2 + B ** 2)
    IOP_012_3d = ref_info['IOP'][0:3] * B - ref_info['IOP'][3:] * A
    IOP_012_3d = IOP_012_3d / np.linalg.norm(IOP_012_3d)
    INP = np.cross(IOP_012_3d, ref_info['INP'])
    IPP = im2patient(0., -C / B, ref_info['IOP'], ref_info['IPP'], ref_info['PixelSpacing'])
    return INP, IPP, score


def locate_intersection_line_in_view(INP_2C, pt_2C, info_ax):
    # find the intersection line
    [P0, N, flag] = plane_intersect(INP_2C, pt_2C, info_ax['INP'], info_ax['IPP'])
    assert flag == 2, "The two planes either coincide or are in parallel."

    # project the intersection line into dark-blood axial view plane
    A, B, C = project_line_to_plane(P0, N, info_ax['IOP'], info_ax['IPP'], info_ax['PixelSpacing'])

    # compute distance map to the intersection line in the plane
    D = distance_map_to_line(A, B, C, (info_ax['Rows'], info_ax['Columns']))

    return D < 1.


def save_slice_prescription_result_as_img(file_img, info_img, INP, IPP, save_dir, info_gt=None):
    save_name = os.path.basename(file_img)[:-10]
    if '2C' in file_img:
        save_name += '-[2C]'
    if '4C' in file_img:
        save_name += '-[4C]'

    img = np.load(file_img)
    img = arr_to_8bit_img(img)
    # channel sequence below is different from the final image, but convenient for operation
    img = np.tile(img[None, ...], [3, 1, 1])

    intersect = locate_intersection_line_in_view(INP, IPP, info_img)
    img[2][intersect] = 255

    if info_gt is not None:
        intersect = locate_intersection_line_in_view(info_gt['INP'], info_gt['IPP'], info_img)
        img[1][intersect] = 255

    img = img.transpose((1, 2, 0))
    cv2.imwrite(os.path.join(save_dir, save_name + '-[PRE].png'), img,  # save for visual debug
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def save_stack_prescription_result_as_img(list_img, list_info, INP, IPP, save_dir, info_gt=None):
    for i, (img_file, img_info) in enumerate(zip(list_img, list_info)):
        save_slice_prescription_result_as_img(img_file, img_info, INP, IPP, save_dir, info_gt)


def compute_metrics(info_gt, IPP, INP):
    # compute coordinate of the central pixel of the ground truth view in the patient coordinate system
    x_im = (1. + info_gt['Columns']) / 2.
    y_im = (1. + info_gt['Rows']) / 2.
    x_patient, y_patient, z_patient = \
        im2patient(x_im, y_im, info_gt['IOP'], info_gt['IPP'], info_gt['PixelSpacing'])
    d = np.abs(np.dot(IPP - np.array([x_patient, y_patient, z_patient]), INP))
    theta = 180. * np.arccos(np.dot(INP, info_gt['INP'])) / np.pi
    if theta > 90:
        theta = 180. - theta
    return d, theta


def locate_peaks_in_heatmap(heatmap):
    v0 = heatmap.max(axis=0)
    r0 = heatmap.argmax(axis=0)
    c0 = np.linspace(0, heatmap.shape[1] - 1, heatmap.shape[1], dtype=int)
    P0 = list(zip(r0, c0))

    c1 = heatmap.argmax(axis=1)
    r1 = np.linspace(0, heatmap.shape[0] - 1, heatmap.shape[0], dtype=int)
    P1 = list(zip(r1, c1))

    P = [list(pt) + [v] for (pt, v) in zip(P0, v0) if pt in P1]
    return np.asarray(P)


def regress_line(x, y, w=None):
    """
    Line equation: y = cx + i -> Ax + By + C = 0
    :param pts:
    :param w:
    :return:
    """
    x = x.reshape(-1, 1)
    reg = LinearRegression().fit(x, y, w)
    # reg = RANSACRegressor().fit(x, y, w).estimator_
    return reg.coef_[0], -1., reg.intercept_, reg.score(x, y, w)


def regress_plane(x, y, z, w=None):
    """
    Plane equation: y = ax + cz + d -> Ax + By + Cz + D = 0
    :param pts:
    :param w:
    :return:
    """
    X = np.hstack((x[..., None], z[..., None]))
    reg = LinearRegression().fit(X, y, w)
    # reg = RANSACRegressor().fit(X, y, w).estimator_
    return reg.coef_[0], -1., reg.coef_[1], reg.intercept_, reg.score(X, y, w)


def compute_plane_param_3d(x, y, z, vals, weight):
    if weight:
        A, B, C, D, score = regress_plane(x, y, z, vals)
    else:
        A, B, C, D, score = regress_plane(x, y, z)
    INP = np.array([A, B, C]) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
    # project origin into the prescribed plane
    IPP = - D / np.sqrt(A ** 2 + B ** 2 + C ** 2) * INP
    return INP, IPP, score


def init_line_in_heatmap(heatmap):
    X_im = np.linspace(0, heatmap.shape[1] - 1, heatmap.shape[1])
    Y_im = np.linspace(0, heatmap.shape[0] - 1, heatmap.shape[0])
    X_im, Y_im = np.meshgrid(X_im, Y_im)

    thresh = threshold_otsu(heatmap)
    x0 = (X_im[heatmap > thresh] * heatmap[heatmap > thresh]).sum() / heatmap[heatmap > thresh].sum()
    y0 = (Y_im[heatmap > thresh] * heatmap[heatmap > thresh]).sum() / heatmap[heatmap > thresh].sum()

    max_score = 0.
    theta = 0.
    line_param = (0., 0., 0.)
    distance_map = np.zeros(heatmap.shape)
    for dir_angle in range(180):
        normal = (-np.sin(np.pi * dir_angle / 180.), np.cos(np.pi * dir_angle / 180.))

        # convert the line equation from point + normal to general form
        A, B, C = normal[0], normal[1], -normal[0] * x0 - normal[1] * y0

        D = distance_map_to_line(A, B, C, heatmap.shape)
        score = heatmap[D < 1.].sum()

        if score > max_score:
            max_score = score
            theta = dir_angle
            line_param = (A, B, C)
            distance_map = D

    return x0, y0, theta, max_score, X_im, Y_im, line_param, distance_map


def grid_search_in_heatmap(heatmap, x_range, y_range, t_range):
    max_score = 0.
    theta, x0, y0 = 0., 0., 0.
    line_param = (0., 0., 0.)
    distance_map = np.zeros(heatmap.shape)
    for dir_angle in range(*t_range):
        for x in range(*x_range):
            for y in range(*y_range):
                normal = (-np.sin(np.pi * dir_angle / 180.), np.cos(np.pi * dir_angle / 180.))

                # convert the line equation from point + normal to general form
                A, B, C = normal[0], normal[1], -normal[0] * x - normal[1] * y

                D = distance_map_to_line(A, B, C, heatmap.shape)
                score = heatmap[D < 1.].sum()

                if score > max_score:
                    max_score = score
                    theta, x0, y0 = dir_angle, x, y
                    line_param = (A, B, C)
                    distance_map = D
    return x0, y0, theta, max_score, line_param, distance_map


def coarse_to_fine_line_search_in_heatmap(heatmap, steps=(15, 5, 1)):
    since = time.time()
    # coarse search
    x0, y0, theta, max_score, line_param, distance_map = grid_search_in_heatmap(
        heatmap, (0, heatmap.shape[1] - 1, steps[0]), (0, heatmap.shape[0] - 1, steps[0]), (0, 180, steps[0]))

    # (iterative) refine
    for i in range(1, len(steps)):
        x0, y0, theta, max_score, line_param, distance_map = grid_search_in_heatmap(
            heatmap, (x0 - steps[i-1], x0 + steps[i-1], steps[i]), (y0 - steps[i-1], y0 + steps[i-1], steps[i]),
            (theta - steps[i-1], theta + steps[i-1], steps[i]))
    print('\tLine search time: %.2fs' % (time.time() - since))

    return x0, y0, theta, max_score, line_param, distance_map


def coarse_to_fine_plane_search(x_range, y_range, z_range, info_dicts, predictions, steps=(15, 5, 1)):
    since = time.time()
    # coarse search
    pt_idx, theta, phi, score, IPP, INP = grid_plane_search_par(
        x_range[::steps[0]], y_range[::steps[0]], z_range[::steps[0]], (0, 180, steps[0]), (0, 180, steps[0]),
        info_dicts, predictions)

    # (iterative) refine
    for i in range(1, len(steps)):
        lb, ub = max(pt_idx * steps[i-1] - steps[i-1], 0), min(pt_idx * steps[i-1] + steps[i-1], len(x_range))
        x_range, y_range, z_range = x_range[lb:ub], y_range[lb:ub], z_range[lb:ub]
        pt_idx, theta, phi, score, IPP, INP = grid_plane_search_par(
            x_range[::steps[i]], y_range[::steps[i]], z_range[::steps[i]],
            (theta - steps[i-1], theta + steps[i-1], steps[i]), (phi - steps[i-1], phi + steps[i-1], steps[i]),
            info_dicts, predictions)
    print('\tPlane searching time: %.2fs' % (time.time() - since))

    return IPP, INP, score


def grid_plane_search_par(X_range, Y_range, Z_range, theta_range, phi_range, info_dicts, predictions):
    theta_range = range(*theta_range)
    phi_range = range(*phi_range)
    scores = Parallel(n_jobs=6)(
        delayed(grid_plane_search_worker)(
            X_range, Y_range, Z_range, theta_range, phi_range, info_dicts, predictions, i)
        for i in range(len(X_range) * len(theta_range) * len(phi_range)))
    max_score = max(scores)
    max_idx = scores.index(max_score)
    pt_idx, theta_idx, phi_idx = ind2sub(len(theta_range), len(phi_range), max_idx)
    IPP0, INP0, theta0, phi0 = sub_2_plane_specs(
        pt_idx, theta_idx, phi_idx, X_range, Y_range, Z_range, theta_range, phi_range)
    return pt_idx, theta0, phi0, max_score, IPP0, INP0


def grid_plane_search_worker(X_range, Y_range, Z_range, theta_range, phi_range, info_dicts, predictions, ind):
    pt_idx, theta_idx, phi_idx = ind2sub(len(theta_range), len(phi_range), ind)
    IPP, INP, _, _ = sub_2_plane_specs(
        pt_idx, theta_idx, phi_idx, X_range, Y_range, Z_range, theta_range, phi_range)
    score = 0.
    for info, pred in zip(info_dicts, predictions):
        score += calc_plane_score_per_view(IPP, INP, info, pred)
    return score


def ind2sub(theta_num, phi_num, ind):
    pt_idx = ind // (theta_num * phi_num)
    theta_idx = ind % (theta_num * phi_num) // phi_num
    phi_idx = ind % (theta_num * phi_num) % phi_num
    return pt_idx, theta_idx, phi_idx


def sub_2_plane_specs(pt_idx, theta_idx, phi_idx, X_range, Y_range, Z_range, theta_range, phi_range):
    theta, phi = theta_range[theta_idx], phi_range[phi_idx]
    IPP = np.array([X_range[pt_idx], Y_range[pt_idx], Z_range[pt_idx]])
    INP = np.array([np.sin(theta / 180. * np.pi) * np.cos(phi / 180. * np.pi),
                    np.sin(theta / 180. * np.pi) * np.sin(phi / 180. * np.pi),
                    np.cos(theta / 180. * np.pi)])
    return IPP, INP, theta, phi


def grid_plane_search(X_range, Y_range, Z_range, theta_range, phi_range, info_dicts, predictions):
    max_score = 0.
    pt_idx = 0.
    theta0, phi0 = 0., 0.
    IPP0, INP0 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
    for i, (x, y, z) in enumerate(zip(X_range, Y_range, Z_range)):
        IPP = np.array([x, y, z])
        for theta in range(*theta_range):
            for phi in range(*phi_range):
                INP = np.array([np.sin(theta / 180. * np.pi) * np.cos(phi / 180. * np.pi),
                                np.sin(theta / 180. * np.pi) * np.sin(phi / 180. * np.pi),
                                np.cos(theta / 180. * np.pi)])
                score = 0.
                for info, pred in zip(info_dicts, predictions):
                    score += calc_plane_score_per_view(IPP, INP, info, pred)
                if score > max_score:
                    max_score = score
                    pt_idx = i
                    theta0, phi0 = theta, phi
                    IPP0, INP0 = IPP, INP
    return pt_idx, theta0, phi0, max_score, IPP0, INP0


def calc_plane_score_per_view(IPP, INP, info, pred):
    # find the intersection line
    [P0, N, flag] = plane_intersect(INP, IPP, info['INP'], info['IPP'])
    if not flag == 2:
        # print("\tThe two planes either coincide or are in parallel.")
        return 0.

    # project the intersection line into the view plane
    A, B, C = project_line_to_plane(P0, N, info['IOP'], info['IPP'], info['PixelSpacing'])

    # compute distance map to the intersection line in the view
    D = distance_map_to_line(A, B, C, (info['Rows'], info['Columns']))

    return pred[D < 1.].sum()


if __name__ == '__main__':
    file = r'F:\CMR\view_plan_pred_4C_loc\078A-[S0782528E]-[20090730]-[HT1.0]-[2778].npy'
    heatmap = np.load(file)[-1]
    colormap = np.tile(heatmap[None, ...], [3, 1, 1])

    x0, y0, theta, score, line_param, D = coarse_to_fine_line_search_in_heatmap(heatmap)
    colormap[0][D < 1.] = 0.
    colormap[1][D < 1.] = 0.
    colormap[2][D < 1.] = 1.
    colormap[1][y0, x0] = 1.

    colormap = colormap.transpose((1, 2, 0))

    print('Done.')
