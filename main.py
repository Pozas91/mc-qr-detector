import cv2
import utils as u

"""
SIGUIENDO EL ARTÍCULO
"""

# Leemos la imagen
img = cv2.imread("assets/File 001.bmp")

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
contours = u.take_firsts_contours(contours_with_areas, 9)

# Sacamos una copia de la imagen
img_qr = img.copy()

# Dibujamos los contornos
cv2.drawContours(img_qr, contours, -1, (0, 255, 0), 2)

# Dibujamos el rectángulo que envuelve a los contornos
u.draw_delimiter_rectangle(contours, img_qr, draw=False)

# Mostramos las imagenes
cv2.imshow("Original", img)
cv2.imshow("Gray", img_gray)
cv2.imshow("Binary", thresh)
cv2.imshow("Detected QR", img_qr)

# Mantenemos abiertas las ventanas hasta que se pulse alguna tecla
cv2.waitKey(0)
cv2.destroyAllWindows()
