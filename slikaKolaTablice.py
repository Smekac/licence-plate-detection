# Proba detekcija tablica na slici(frame) koju smo dobili iz videa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
                                                                                          #'ImagesFrame2\slika39.jpg'
nizSlika = ['ImagesFrame\slika1.jpg', 'ImagesFrame\slika7.jpg','ImagesFrame2\slika41.jpg','ImagesFrame\slika333.jpg']
index = 0

### Predefinisana funkcija ......
def ZaDetekciju(Okvir):
    img_GRAY_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    plt.imshow(img_GRAY_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_GRAY_gs, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61,15)

    ####
    retSlike, image_bin = cv2.threshold(img_GRAY_gs, 0, 255, cv2.THRESH_OTSU)
    print("pRAG JE: " + str(retSlike))
    ####

    plt.imshow(image_barcode_bin, 'gray')
    #plt.show()
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
        if w > 35 and w < 120 and h > 10 and h < 90:  # uslov da kontura pripada bar-kodu //36 bila
            # if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
            if (xx > 200 and xx < 400 and yy > 200 and yy < 380):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
        # print("X kordinata je: " + str(xx) + " A ovo je y kordinata: "+ str(yy) )
        # print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 0, 255), 4)
    plt.imshow(img)
    plt.show()

    return img, contours_Tablica, retSlike
###

while index < len(nizSlika):

    SlikaKola = cv2.imread(nizSlika[index])  # Slika1

    # print(SlikaKola.shape)
    # SlikaKola = cv2.resize(SlikaKola,(480,640),interpolation = cv2.INTER_CUBIC)

    SlikaKola = cv2.cvtColor(SlikaKola, cv2.COLOR_BGR2RGB)

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
        if w > 35 and w < 180 and h > 10 and h < 80:  # uslov da kontura pripada bar-kodu
            if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu

    img = SlikaKola.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()

    #Index kada imamo vise ...........

    if(len(contours_Tablica) >= 1):
        x, y, w, h = cv2.boundingRect(contours_Tablica[0])
        print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

        plt.imshow(SlikaKola[y:y + h, x:x + w])  # SlikaKola[x:x+w,y:y+h])
        plt.show()
        for ii in range (0,len(contours_Tablica)):
           x, y, w, h = cv2.boundingRect(contours_Tablica[ii])
           if(contours_Tablica[ii] is not None ):
             cv2.imwrite('plates\slikaKola%dkontura%d.png' %(index ,ii) , SlikaKola[y:y + h, x:x + w])

        index = index + 1

        print(str(len(contours_Tablica)))
    else:
        print("Nema nista teraj dalje ......")
        index = index + 1
        #SlikaKola2 = cv2.cvtColor(SlikaKola, cv2.COLOR_RGB2BGR)
        # SlikaKola2 = cv2.cvtColor(SlikaKola, cv2.COLOR_BGR2RGB)
        #
        # iscrtana, konturee, VrednostPraga = ZaDetekciju(SlikaKola2)
        #
        # if (VrednostPraga < 116 and VrednostPraga != 0):
        #     lower = np.array([140, 140, 140], dtype="uint8")  # Za bolje detektovanje ....!!!! 160
        # elif VrednostPraga > 116 and VrednostPraga < 150:
        #     lower = np.array([170, 170, 170], dtype="uint8")  # 160 malo spustiti granicu
        # elif (VrednostPraga > 150 and VrednostPraga < 190):
        #     lower = np.array([170, 170, 170], dtype="uint8")  # Zbog boje asfalta
        # else:
        #     lower = np.array([200, 200, 200], dtype="uint8")  # Izuzetak zbog bolje binarizacije !!!!!!!!!!
        #
        # upper = np.array([255, 255, 255], dtype="uint8")
        #
        # # find the colors within the specified boundaries and apply
        # # the mask
        # mask = cv2.inRange(SlikaKola2, lower, upper)
        # output = cv2.bitwise_and(SlikaKola2, SlikaKola2, mask=mask)
        # # output2 = 255 - output  # Ako su kola u mraku NE treba invertovati (TREBA DODATNI USLOV)
        # plt.imshow(output)
        # plt.figure()
        # plt.imshow(output, 'gray')
        # plt.show()
        #
        # iscrtana, contours_Tablica, retSlike = ZaDetekciju(output)
        #
        # # if(VrednostPraga < 140 and VrednostPraga > 109  and  contours_Tablica is None ):
        # if (len(contours_Tablica) != 0):  # Da ne bi pucao program provera !!!!!!!!!!!!!!!!!!!!!!!!!
        #     x, y, w, h = cv2.boundingRect(contours_Tablica[0])
        #     print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))
        #     center, size, angle = cv2.minAreaRect(contours_Tablica[0])
        #     print(" A ugao je: " + str(angle))
        #     if (angle < -30):  # Nekad bude slika okrenuta po difoltu pa provera potrebna
        #         angle = angle + 90
        #
        #     # if(SlikaKola.shape == (480, 640, 3)):
        #     if  w > 35 and w < 180 and h > 10 and h < 100:
        #      if (x > 200 and x < 400 and y> 200 and y < 380):
        #         #  if(x > 100 and x < 650 and y > 150 and y < 450):
        #         print("Dobrodosaooo....")
        #         rotated = ndimage.rotate(output, angle, reshape=True)
        #         RotiranaBezBoja = ndimage.rotate(SlikaKola2, angle, reshape=True)
        #         imgRotirana, KonturaKadjeSlikaRotirana, retSlike = ZaDetekciju(rotated)
        #         xxx, yyy, www, hhh = cv2.boundingRect(KonturaKadjeSlikaRotirana[0])
        #         center, size, angle = cv2.minAreaRect(KonturaKadjeSlikaRotirana[0])
        #         plt.imshow(imgRotirana[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
        #         plt.figure()
        #         cv2.imwrite('plates\slika%d.png' % index, RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])
        #         plt.imshow(RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
        #         index = index + 1
        #         plt.show()