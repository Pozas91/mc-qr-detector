import os
import cv2
import utils as u

qr_detected = set()
files_read = set()

directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

for root, dirs, files in os.walk(directory):
    for file_name in files:
        """
        SIGUIENDO EL ARTÍCULO
        """

        # Establecemos el fichero
        file = os.path.join(directory, file_name)

        # Leemos la imagen
        img = cv2.imread(file)

        # La convertimos a escala de grises.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarizamos la imagen
        ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Buscamos los contornos
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Simplificamos los contornos con el algoritmo que se indica en el artículo
        contours = u.contour_sifting(contours)

        """
        EXTRA
        """

        # Eliminamos los contornos duplicados
        contours = u.remove_duplicates(contours)

        # Ordenamos los contornos por area
        contours_with_areas = u.contours_order_by_area(contours)

        # Podemos coger los 9 mayores, estos correspondrán con los 9 FIPs del código QR, además si tenemos varios QR,
        # cogerán el más centrado que será el más grande.
        contours = u.take_firsts_contours(contours_with_areas, 3)

        # Sacamos una copia de la imagen
        img_qr = img.copy()

        # Comprobamos si tenemos contornos que cumplen con las restricciones dadas
        if len(contours) > 2:

            # Dibujamos el rectángulo que envuelve a los contornos
            img_rotated, rectangle = u.delimiter_and_rotate_rectangle(contours, img_qr, draw=False)

            # Sacamos el ratio del rectángulo dibujado
            (center_x, center_y), (width, height), angle = rectangle
            ratio = width / height

            # Comprobamos si el rectángulo pintado tiene la proporción de más o menos 1
            if 0.8 < ratio < 1.2:
                qr_detected.add(file_name)

        # Añadimos el fichero leido
        files_read.add(file_name)

        # Obtenemos los códigos QR no detectados
        qr_not_detected = files_read - qr_detected

        # Mostramos el resultado actual
        print('\n-----------------------------------------------------------------------------------------------------')
        print(' Fichero "{0}" leído.'.format(file_name))
        print(' Total ficheros leídos: {0}.'.format(len(files_read)))
        print(' Códigos detectados ({0}): \n\t {1}'.format(len(qr_detected), qr_detected))
        print(' Códigos no detectados ({0}): \n\t {1}'.format(len(qr_not_detected), qr_not_detected))
        print(' Porcentaje de aciertos: {0:.2f}%.'.format((len(qr_detected) / len(files_read)) * 100))
        print('-----------------------------------------------------------------------------------------------------\n')
