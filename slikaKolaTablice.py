# Proba detekcija tablica na slici(frame) koju smo dobili iz videa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

nizSlika = ['ImagesFrame\slika1.jpg', 'ImagesFrame\slika7.jpg']
index = 0


while index < len(nizSlika):

    SlikaKola = cv2.imread(nizSlika[index])  # Slika1

    # print(SlikaKola.shape)
    # SlikaKola = cv2.resize(SlikaKola,(480,640),interpolation = cv2.INTER_CUBIC)

    SlikaKola = cv2.cvtColor(SlikaKola, cv2.COLOR_BGR2RGB)

    # pts = np.array([[150, 190], [150, 420], [380, 420], [380, 190]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(SlikaKola, [pts], True, (255, 0, 0))

    pts = np.array([[200, 210], [200, 380], [400, 380], [400, 210]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(SlikaKola, [pts], True, (255, 0, 255))

    img_barcode_gs = cv2.cvtColor(SlikaKola, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    plt.imshow(img_barcode_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_barcode_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61,
                                              15)
    plt.imshow(image_barcode_bin, 'gray')
    img, contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = SlikaKola.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size  # Obrnuo sam parametre , Prvo ide visina ??!!?!?!
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if width > 15 and width < 120 and height > 35 and height < 100:  # uslov da kontura pripada bar-kodu
            if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu

    img = SlikaKola.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()

    # y,x
    x, y, w, h = cv2.boundingRect(contours_Tablica[0])
    print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

    #if(SlikaKola.shape == (480, 640, 3)):
    plt.imshow(SlikaKola[y:y + h, x:x + w])  # SlikaKola[x:x+w,y:y+h])
    # KropovanaSlika = SlikaKola[252:272,288:348]
    # plt.imshow(KropovanaSlika)
    plt.show()
    cv2.imwrite('Images\slika%d.jpg' % index, SlikaKola[y:y + h, x:x + w])

    index = index + 1

    print(str(len(contours_Tablica)))

