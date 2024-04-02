import cv2

THRESH = 150
VALUE = 255

image = cv2.imread('variant-10.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(image, THRESH, VALUE, cv2.THRESH_BINARY)

cv2.imshow('Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()