import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import prediction as pred

img_width, img_height = 28, 28


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((4, 4))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def vertical_erode(image):
    cols = image.shape[1]
    vertical_size = cols / 60
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size)))
    return cv2.erode(image, verticalStructure)
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
        if w > 15 and w < 90 and h > 30 and h < 100 and area > 200:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_color, sorted_regions

# def correct_char(sorted_regions):
#     for i,region in enumerate(sorted_regions):
#         x1,y1,w1,h1 = cv2.boundingRect(sorted_regions[i])
#         x2, y2, w2, h2 = cv2.boundingRect(sorted_regions[i+1])
#         print(x1,x2)

def prepare_for_nn(regions):
    ready_for_nn = []
    for region in regions:
        scale = region / 255
        ready_for_nn.append(region.flatten())
    return ready_for_nn


def check_result(string):
    p = re.compile('[A-Z]{2}[0-9]{3,4}[A-Z]{2}')
    return p.match(string)

file_object = open('plates.txt', 'r+')
plate_color = cv2.imread('images/plate10.png')
# lower = np.array([0,0,0], dtype="uint8")
# upper = np.array([20,20,20], dtype="uint8")
# mask = cv2.inRange(plate_color,lower,upper)
# plate_color = cv2.bitwise_and(plate_color,plate_color,mask=mask)

plate_color = cv2.resize(plate_color, (450, 105), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('color',plate_color)
plate_color = cv2.GaussianBlur(plate_color, (3, 3), 0)
plate_gray = cv2.cvtColor(plate_color, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', plate_gray)
plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 30)

#vertikalna erozija
plate_bin= vertical_erode(plate_bin)
cv2.imshow('binary', plate_bin)

# izdvajanje regiona od interesa sa tablice
selected_regions, chars = select_roi(plate_color, plate_bin)
cv2.imshow('regions', selected_regions)

# predikcija karaktera
types = []
if (len(chars) == 7 or len(chars) == 8 and len(chars) > 6):

    if (len(chars) == 7):
        types = [False, False, True, True, True, False, False]
        print(types)
    elif (len(chars) == 8):
        types = [False, False, True, True, True, True, False, False]
        print(types)

    result = []
    for i, char in enumerate(chars):
        imgs = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
        cv2.imshow('test', imgs)
        model = pred.create_model()
        result.append(pred.prediction(model, imgs, types[i]))
    print(result)

    licence_plate = ''.join(result) + '\n'

    checked = check_result(licence_plate)
    if not checked:
        licence_plates = file_object.readlines()
        if licence_plate not in licence_plates and len(licence_plate) > 1:
            file_object.write(licence_plate)

file_object.close()
plt.show()
cv2.waitKey()
