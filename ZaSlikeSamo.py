import cv2
import numpy as np
import matplotlib.pyplot as plt


# from ZaSlikeSamo import Kola1


def ZaDetekciju(Okvir):
    img_GRAY_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    plt.imshow(img_GRAY_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_GRAY_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61,
                                              15)
    plt.imshow(image_barcode_bin, 'gray')
    plt.show()
    img, contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = Okvir.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 130 and w < 220 and h > 36 and h < 90:  # uslov da kontura pripada bar-kodu
            # if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
            contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
        # print("X kordinata je: " + str(xx) + " A ovo je y kordinata: "+ str(yy) )
        # print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 0, 255), 4)
    plt.imshow(img)
    plt.show()

    return img, contours_Tablica


# Kola1 = cv2.imread('ImagesFrame\kola1.jpg')
Kola1 = cv2.imread('ImagesOfCars\kola2.jpg')  # kolaMrak.png') #kola2.jpg')

if (Kola1.shape is not (533, 800, 3)):
    Kola1 = cv2.resize(Kola1, (800, 533))

print('Velicina oblika ' + str(Kola1.shape))

Kola1 = cv2.cvtColor(Kola1, cv2.COLOR_BGR2RGB)
iscrtana = ZaDetekciju(Kola1)

lower = np.array([190, 190, 190], dtype="uint8")
upper = np.array([255, 255, 255], dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(Kola1, lower, upper)
output = cv2.bitwise_and(Kola1, Kola1, mask=mask)
output2 = 255 - output  # Ako su kola u mraku NE treba invertovati (TREBA DODATNI USLOV)
plt.imshow(output)
plt.figure()
plt.imshow(output2, 'gray')
plt.show()

iscrtana, contours_Tablica = ZaDetekciju(output2)

x, y, w, h = cv2.boundingRect(contours_Tablica[0])
print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

# if(SlikaKola.shape == (480, 640, 3)):
plt.imshow(Kola1[y:y + h, x:x + w])  # SlikaKola[x:x+w,y:y+h])
cv2.imwrite('Images\SamoSlika3.jpg', Kola1[y:y + h, x:x + w])

# KropovanaSlika = SlikaKola[252:272,288:348]
# plt.imshow(KropovanaSlika)
plt.show()
# cv2.imwrite('Images\slika%d.jpg' % index, SlikaKola[y:y + h, x:x + w])
# plt.imshow(iscrtana)
# plt.show()
