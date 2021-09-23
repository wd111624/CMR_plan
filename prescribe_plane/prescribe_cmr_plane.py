import numpy as np

from .util import coarse_to_fine_plane_search
import sys
sys.path.append("..")
from utils.geometry import im2patient, plane_intersect, project_line_to_plane, distance_map_to_line


def get_pt_range(info_host, info_guest, thresh=1.):
    # find the intersection line
    [P0, N, flag] = plane_intersect(info_host['INP'], info_host['IPP'], info_guest['INP'], info_guest['IPP'])
    assert flag == 2, "The two planes either coincide or are in parallel."

    # project the intersection line into the 2C localizer view plane
    A, B, C = project_line_to_plane(P0, N, info_host['IOP'], info_host['IPP'], info_host['PixelSpacing'])

    # compute distance map to the intersection line in the plane
    D = distance_map_to_line(A, B, C, (info_host['Rows'], info_host['Columns']))

    # extract planar points on the line segment contained within the 2C view
    X_im = np.linspace(0, info_host['Columns'] - 1, info_host['Columns'])
    Y_im = np.linspace(0, info_host['Rows'] - 1, info_host['Rows'])
    X_im, Y_im = np.meshgrid(X_im, Y_im)
    X_im, Y_im = X_im[D < thresh], Y_im[D < thresh]
    X_patient, Y_patient, Z_patient = im2patient(
        X_im.flatten(), Y_im.flatten(), info_host['IOP'], info_host['IPP'], info_host['PixelSpacing'])
    sort_ind = X_patient.argsort()
    return X_patient[sort_ind], Y_patient[sort_ind], Z_patient[sort_ind]


def search_LAX_cine_plane(info_SA, pred_SA, info_LAX, pred_LAX):
    # get coordinate search range for point on the target plane
    x_range, y_range, z_range = get_pt_range(info_SA[int(len(info_SA) / 2)], info_LAX)
    # search for optimal plane
    return coarse_to_fine_plane_search(x_range, y_range, z_range, info_SA + [info_LAX], pred_SA + [pred_LAX])


def search_SAX_cine_plane(info_4C, pred_4C, info_2C, pred_2C):
    # get coordinate search range for point on the target plane
    X_patient, Y_patient, Z_patient = get_pt_range(info_4C, info_2C)

    # search for optimal plane
    return coarse_to_fine_plane_search(X_patient, Y_patient, Z_patient, (info_2C, info_4C), (pred_2C, pred_4C))


def search_LVOT_cine_plane(info_SA, pred_SA, info_2C, pred_2C, info_4C, pred_4C):
    # get coordinate search range for point on the target plane
    x_range, y_range, z_range = get_pt_range(info_SA[int(len(info_SA) / 2)], info_2C)
    # search for optimal plane
    return coarse_to_fine_plane_search(
        x_range, y_range, z_range, info_SA + [info_2C, info_4C], pred_SA + [pred_2C, pred_4C])
