import cv2
import numpy as np
import matplotlib.pyplot as plt


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (28, 28,), interpolation=cv2.INTER_NEAREST)


def select_roi(image_color, image_bin):
    img, contours, hierarchy = cv2.findContours(plate_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    contours_plate = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > 20 and w < 90 and h > 30 and h < 100 and area > 200:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_color, sorted_regions


def prepare_for_nn(regions):
    ready_for_nn = []
    for region in regions:
        scale = region / 255
        ready_for_nn.append(region.flatten())
    return ready_for_nn


plate_color = cv2.imread('images/plate3.png')
plate_color = cv2.resize(plate_color, (450, 105), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('color',plate_color)
plate_color = cv2.GaussianBlur(plate_color, (3, 3), 0)
plate_gray = cv2.cvtColor(plate_color, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', plate_gray)
plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 30)
kernel = np.ones((4, 4))
plate_bin = cv2.erode(plate_bin, kernel, iterations=1)
# plate_bin = erode(dilate(plate_bin))
cv2.imshow('binary', plate_bin)

# izdvajanje regiona od interesa sa tablice
selected_regions, chars = select_roi(plate_color, plate_bin)
cv2.imshow('regions', selected_regions)

# priprema za neuronsku mrezu
inputs = prepare_for_nn(chars)

plt.show()
cv2.waitKey()
