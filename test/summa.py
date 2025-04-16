import cv2


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


image = cv2.imread("./Kavin Subramaniam/face_50.jpg")
# denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
denoised_image = apply_clahe(image)
cv2.imshow("test", denoised_image)
cv2.imshow("og", image)
cv2.waitKey(0)
