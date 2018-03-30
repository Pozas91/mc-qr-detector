import cv2
import numpy as np
import utils as u

# Read image
img = cv2.imread("assets/qr.jpg")

# Convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binary image
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find Contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = u.contour_sifting(contours)

# Draw contours
cv2.drawContours(img, contours, 0, (0, 255, 0), 1)

# Show image2
cv2.imshow("Original", img)
cv2.imshow("Gray", img_gray)
cv2.imshow("Binary", thresh)

# Keep open until press a key
cv2.waitKey(0)
cv2.destroyAllWindows()
