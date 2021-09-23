import numpy as np

import sys
sys.path.append("..")
from utils.util import EPS


def im2patient(X_im, Y_im, IOP, IPP, PS):
    X_patient = IOP[0] * X_im * PS[1] + IOP[3] * Y_im * PS[0] + IPP[0]
    Y_patient = IOP[1] * X_im * PS[1] + IOP[4] * Y_im * PS[0] + IPP[1]
    Z_patient = IOP[2] * X_im * PS[1] + IOP[5] * Y_im * PS[0] + IPP[2]
    return X_patient, Y_patient, Z_patient


def distance_map_to_line(A, B, C, shape):
    """
    Line equation: Ax + By + C = 0
    :param A:
    :param B:
    :param C:
    :param shape: An iterable whose 0th and 1st elements indicate rows and columns of the view.
    :return:
    """
    X_im = np.linspace(0, shape[1] - 1, shape[1])
    Y_im = np.linspace(0, shape[0] - 1, shape[0])
    X_im, Y_im = np.meshgrid(X_im, Y_im)
    D = np.abs(A * X_im + B * Y_im + C) / np.sqrt(A ** 2. + B ** 2.)
    return D


def patient2im(X_p, Y_p, Z_p, IOP, IPP, PS):
    X_vector = X_p - IPP[0]
    Y_vector = Y_p - IPP[1]
    Z_vector = Z_p - IPP[2]

    project2imgX = X_vector * IOP[0] + Y_vector * IOP[1] + Z_vector * IOP[2]
    project2imgY = X_vector * IOP[3] + Y_vector * IOP[4] + Z_vector * IOP[5]

    X_im = project2imgX / PS[1]
    Y_im = project2imgY / PS[0]

    return X_im, Y_im


def project_line_to_plane(P0, N, IOP, IPP, PS):
    # get another point on the line
    P1 = P0 + N

    # project both points to the image plane
    x0, y0 = patient2im(P0[0], P0[1], P0[2], IOP, IPP, PS)
    x1, y1 = patient2im(P1[0], P1[1], P1[2], IOP, IPP, PS)

    # compute the projected line in the image plane
    """
    Ax + By + C = 0:
        -> A = y1 - y0    
        -> B = x0 - x1     
        -> C = x1*y0 - x0*y1
    """
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1

    return A, B, C


def plane_intersect(N1, A1, N2, A2):
    """
    computes the intersection of two planes(if any)
    :param N1: normal vector to Plane 1
    :param A1: any point that belongs to Plane 1
    :param N2: normal vector to Plane 2
    :param A2: any point that belongs to Plane 2
    :return:
        P: a point that lies on the interection straight line
        N: the direction vector of the straight line
        check: a flag (0:Plane 1 and Plane 2 are parallel;
                       1:Plane 1 and Plane 2 coincide;
                       2:Plane 1 and Plane 2 intersect)

    Example:
    Determine the intersection of these two planes:
    2x - 5y + 3z = 12 and 3x + 4y - 3z = 6
    The first plane is represented by the normal vector N1=[2 -5 3]
    and any arbitrary point that lies on the plane, ex: A1=[0 0 4]
    The second plane is represented by the normal vector N2=[3 4 -3]
    and any arbitrary point that lies on the plane, ex: A2=[0 0 -2]
    [P,N,check]=plane_intersect([2 -5 3],[0 0 4],[3 4 -3],[0 0 -2]);

    This function is rewritten by Dong Wei (weidong111624@gmail.com)
    from the MATLAB function originally written by:
        Nassim Khaled
        Wayne State University
        Research Assistant and Phd candidate
    """

    P = np.array([0., 0., 0.])
    N = np.cross(N1, N2)

    # test if the two planes are parallel
    if np.linalg.norm(N) < EPS:  # Plane 1 and Plane 2 are near parallel
        V = A1 - A2
        if np.dot(N1, V) < EPS:
            check = 1  # Plane 1 and Plane 2 coincide
        else:
            check = 0  # Plane 1 and Plane 2 are disjoint
        return P, N, check

    check = 2

    # Plane 1 and Plane 2 intersect in a line
    # first determine max abs coordinate of cross product
    maxc = np.flatnonzero(np.abs(N) == np.max(np.abs(N)))[0]

    # next, to get a point on the intersection line and zero the max coord,
    # and solve for the other two
    d1 = -np.dot(N1, A1)  # the constants in the Plane 1 equations
    d2 = -np.dot(N2, A2)  # the constants in the Plane 2 equations

    if maxc == 0:  # intersect with x=0
        P[1] = (d2 * N1[2] - d1 * N2[2]) / N[0]
        P[2] = (d1 * N2[1] - d2 * N1[1]) / N[0]
    elif maxc == 1:  # intersect with y=0
        P[0] = (d1 * N2[2] - d2 * N1[2]) / N[1]
        P[2] = (d2 * N1[0] - d1 * N2[0]) / N[1]
    elif maxc == 2:  # intersect with z=0
        P[0] = (d2 * N1[1] - d1 * N2[1]) / N[2]
        P[1] = (d1 * N2[0] - d2 * N1[0]) / N[2]

    return P, N, check
