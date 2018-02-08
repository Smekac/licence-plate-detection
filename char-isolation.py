import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import prediction as pred

img_width, img_height = 28, 28


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((4, 4))
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
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > 15 and w < 90 and h > 30 and h < 100 and area > 600:
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


def check_result(string):
    p = re.compile('[A-Z]{2}[0-9]{3,4}[A-Z]{2}')
    return p.match(string)


def load_cities(file_cities):
    cities = {}
    for line in file_cities:
        (key, val) = line.split()
        cities[key] = val
    return cities

#funkcija menja oznaku grada ako ne postoji u bazi gradova sa najverovatnijim
def validate_plate(cities, licence_plate):
    city = licence_plate[0:2];
    cSimilar = {}
    if city in cities:
        return licence_plate
    else:
        for ci in cities:
            if(city[0] in ci[0] or city[1] in ci[1]):
                c1Ver = ci
                break
        if(len(c1Ver) == 2):
            licence_plate = c1Ver + licence_plate[2:]
    return licence_plate


# ucitavanje tekstualnih falova
file_object = open('plates.txt', 'r+')
file_cities = open('cities.txt', 'r')
plate_color = cv2.imread('images/plate4.png')

cities = load_cities(file_cities)

plate_color = cv2.resize(plate_color, (450, 105), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('color',plate_color)
plate_color = cv2.GaussianBlur(plate_color, (3, 3), 0)
plate_gray = cv2.cvtColor(plate_color, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', plate_gray)
plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 30)
cv2.imshow('binary', plate_bin)
# vertikalna erozija
plate_bin = vertical_erode(plate_bin)
cv2.imshow('erode', plate_bin)

# izdvajanje regiona od interesa sa tablice
selected_regions, chars = select_roi(plate_color, plate_bin)
cv2.imshow('regions', selected_regions)

# predikcija karaktera ako je pronadjen dovoljan broj regiona
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
        model = pred.create_model()
        result.append(pred.prediction(model, imgs, types[i]))

    licence_plate = ''.join(result) + '\n'
    print('Pre validacije ' + licence_plate)

    checked = check_result(licence_plate)
    # provera da li su prva dva karaktera grada validna ako nisu menja ih u najverovatnije
    valid_plate = validate_plate(cities, licence_plate)
    print('Posle validacije ' + valid_plate)
    if checked:
        licence_plates = file_object.readlines()
        if valid_plate not in licence_plates and len(valid_plate) > 1:
            file_object.write(valid_plate)

file_object.close()
plt.show()
cv2.waitKey()
