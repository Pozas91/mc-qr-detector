import cv2
import numpy as np
import math

# The smallest contour of FIP is the contour of a 3*3 modules.
t = 3 * 3


def center_contours(contour: np.ndarray) -> (int, int):
    """
    Returns centroid of the contour given.
    :param contour:
    :return:
    """
    moment = cv2.moments(contour)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])

    return cx, cy


def distance_between_centers(center_1: (int, int), center_2: (int, int)) -> float:
    """
    Returns the distance between two centers given.
    :param center_1:
    :param center_2:
    :return:
    """
    return math.sqrt((center_2[0] - center_1[0]) ** 2 + (center_2[1] - center_1[1]) ** 2)


def ratio_criterion(ratio: float, epsilon: float) -> bool:
    """
    Check if ratio given satisfy ratio criterion
    :param ratio:
    :param epsilon:
    :return:
    """
    return 1 - epsilon < ratio < 1 + epsilon


def length_criterion(length: float) -> bool:
    """
    Check if the length given satisfy length criterion
    :param length:
    :param t:
    :return:
    """
    return length > t


def overlap_criterion(center_i: (int, int), center_j: (int, int), center_k: (int, int), d: float) -> bool:
    """
    Check if the three centers satisfy the overlap criterion.
    :param center_i:
    :param center_j:
    :param center_k:
    :param d:
    :return:
    """
    distance_i_j = distance_between_centers(center_i, center_j)
    distance_i_k = distance_between_centers(center_i, center_k)
    return distance_i_j < d and distance_i_k < d


def contour_sifting(contours: list, epsilon=0.2, distance=10) -> list:
    """
    Function to simplify contours
    :param contours:
    :param epsilon:
    :param distance:
    :return:
    """

    f = list()
    valid_contours = list()

    for i, contour in enumerate(contours):

        f.append(0)

        x, y, w, h = cv2.boundingRect(contour)
        ratio = float(w) / float(h)
        area = cv2.contourArea(contour)

        if not ratio_criterion(ratio, epsilon):
            f[i] = -1

        if not length_criterion(area):
            f[i] = -1

    valid_indexes = set([i for i, _ in enumerate(f) if f[i] != -1])

    for i in valid_indexes:
        for j in (valid_indexes - {i}):
            for k in (valid_indexes - {i, j}):

                contour_i, contour_j, contour_k = contours[i], contours[j], contours[k]
                center_i, center_j, center_k = center_contours(contour_i), center_contours(contour_j), center_contours(
                    contour_k)

                if overlap_criterion(center_i, center_j, center_k, distance):
                    valid_contours.append(contours[i])

    return valid_contours
