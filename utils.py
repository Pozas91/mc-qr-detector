from operator import itemgetter
import cv2
import numpy as np
import math
import imutils

# El menor contorno de un FIP is el contorno que tiene un módulo de 3*3.
t = 3 * 3


def center_contour(contour: np.ndarray) -> (int, int):
    """
    Devolvemos el centro del contorno dado.
    :param contour:
    :return:
    """
    moment = cv2.moments(contour)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])

    return cx, cy


def distance_between_centers(center_1: (int, int), center_2: (int, int)) -> float:
    """
    Devolvemos la distancia entre los dos centros dados.
    :param center_1:
    :param center_2:
    :return:
    """
    return math.sqrt((center_2[0] - center_1[0]) ** 2 + (center_2[1] - center_1[1]) ** 2)


def ratio_criterion(ratio: float, epsilon: float) -> bool:
    """
    Comprobamos si el radio dado satisface el criterio del ratio.
    :param ratio:
    :param epsilon:
    :return:
    """
    return 1 - epsilon < ratio < 1 + epsilon


def length_criterion(length: float) -> bool:
    """
    Comprobamos si el area dada satisface el criterio de longitud.
    :param length:
    :param t:
    :return:
    """
    return length > t


def overlap_criterion(center_i: (int, int), center_j: (int, int), center_k: (int, int), d: float) -> bool:
    """
    Comprobamos si los tres centros satisfacen el criterio de superposición.
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
    Devuelve los contornos simplificados.
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

        if not ratio_criterion(ratio, epsilon):
            f[i] = -1

        if f[i] is not -1:

            area = cv2.contourArea(contour)

            if not length_criterion(area):
                f[i] = -1

    valid_indexes = set([i for i, _ in enumerate(f) if f[i] != -1])

    for i in valid_indexes:
        for j in (valid_indexes - {i}):
            for k in (valid_indexes - {i, j}):

                contour_i, contour_j, contour_k = contours[i], contours[j], contours[k]
                center_i, center_j, center_k = center_contour(contour_i), center_contour(contour_j), center_contour(
                    contour_k)

                if overlap_criterion(center_i, center_j, center_k, distance):
                    valid_contours.append(contours[i])

    return valid_contours


def remove_duplicates(contours: list) -> list:
    """
    Elimina los contornos duplicados.
    :param contours:
    :return:
    """
    return list({contour.tostring(): contour for contour in contours}.values())


def contours_order_by_area(contours: list) -> list:
    """
    Devuelve los contornos ordenados por area de manera descendente.
    :param contours:
    :return:
    """

    return sorted(contours, key=cv2.contourArea, reverse=True)


def get_qr_fips(contours: list) -> list:
    """
    Devuelve los contornos que sean parecidos entre sí.
    :param contours:
    :return:
    """

    fips = 3

    contours_selected = list()

    # Por cada contorno
    for a in contours:

        # Añadimos el contorno como base
        contours_selected.append(a)

        for b in contours:

            # Comprobamos que sea diferente al primero
            if a is not b:

                # Calculamos el área y el perímetro de los dos
                area_a = cv2.contourArea(a)
                area_b = cv2.contourArea(b)
                perimeter_a = cv2.arcLength(a, True)
                perimeter_b = cv2.arcLength(b, True)

                # Comprobamos si más o menos (+-10%) tienen el mismo área y el mismo perímetro, si esto es así, podemos
                # intuir que son la misma figura del mismo tamaño.
                if (area_b - (area_b * 0.1)) < area_a < (area_b + (area_b * 0.1)) and (perimeter_b - (perimeter_b * 0.1)) < perimeter_a < (perimeter_b + (perimeter_b * 0.1)):
                    contours_selected.append(b)

        # Si ya lo hemos encontrado dejamos de buscar, si no, vaciamos la lista y empezamos a buscar de nuevo
        if len(contours_selected) < fips:
            contours_selected = list()
        else:
            break

    # Devolvemos el número de contornos indicados
    return contours_selected[:fips]


def delimiter_and_rotate_rectangle(contours: list, img: np.ndarray, draw=True) -> (np.ndarray, np.ndarray):
    # Hacemos una copia de la imagen para no afectar a la original
    img_copy = img.copy()

    # Sacamos el primer contorno
    points = contours[0]

    # Unimos los puntos de todos los contornos
    for contour in contours[1:]:
        points = np.concatenate((points, contour))

    # Sacamos el mínimo cuadrado que recubre todos los puntos
    rectangle = cv2.minAreaRect(points)
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)

    # Si queremos dibujarlo, lo dibujamos
    if draw:
        cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)

    (center_x, center_y), (width, height), angle = rectangle

    # Sacamos el ángulo para enderezar la imagen
    new_angle = 0.0

    # Hacemos unas pequeñas comprobamos para corregir el ángulo
    if angle < -45.0:
        new_angle = -(90 + angle)
    elif angle >= -45.0:
        new_angle = -angle

    # Devolvemos la imagen rotada
    return imutils.rotate_bound(img_copy, new_angle), rectangle

# def crop_image(img: np.ndarray) -> np.ndarray:
#     x = int(center_x - width / 2)
#     y = int(center_y - height / 2)
#
#     img_copy = img_copy[y: y + int(height), x:x + int(width)]
