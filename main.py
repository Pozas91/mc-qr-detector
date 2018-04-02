import cv2
import utils as u
import os

"""
SIGUIENDO EL ARTÍCULO
"""

# Establecemos el fichero
file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/File 005.bmp')

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

# Dibujamos los contornos
cv2.drawContours(img_qr, contours, -1, (0, 255, 0), 2)

# Comprobamos si tenemos contornos que cumplen con las restricciones dadas
if len(contours) <= 2:
    raise ValueError('No se han encontrado contornos que cumplan con las restricciones dadas.')

# Dibujamos el rectángulo que envuelve a los contornos
img_rotated, rectangle = u.delimiter_and_rotate_rectangle(contours, img_qr)

# Mostramos las imagenes
cv2.imshow("Original", img)
cv2.imshow("Gray", img_gray)
cv2.imshow("Binary", thresh)
cv2.imshow("Detected QR", img_qr)
cv2.imshow("Rotated QR", img_rotated)

# Mantenemos abiertas las ventanas hasta que se pulse alguna tecla
cv2.waitKey(0)
cv2.destroyAllWindows()
