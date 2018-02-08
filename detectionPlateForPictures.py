#Rade svee !!!!!!!!!!!!!!!!!!!!!!
# Mozda dodati da menja po z osi perspektivu ako dohvati kola u daljini
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


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
    #plt.show()

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 130 and w < 320 and h > 36 and h < 120:  # uslov da kontura pripada bar-kodu //36 bila
            # if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
            if (xx > 100 and xx < 650 and yy > 210 and yy < 450):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
        # print("X kordinata je: " + str(xx) + " A ovo je y kordinata: "+ str(yy) )
        # print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))

    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 0, 255), 4)
    plt.imshow(img)
    #plt.show()

    return img, contours_Tablica, retSlike

#############################################################################################################

#Kola1 = cv2.imread('ImagesFrame\slika2.jpg')
#for index in range (0,15) : #Ako treba sve od jednom
#Kola1 = cv2.imread('ImagesOfCars\PA210245.jpg') # P1010128.jpg')      #slika1.jpg')    #PA210245.jpg') #slika5.jpg')  # kolaMrak.png') #kola2.jpg')

#for index in range(1,15):
index =1
while index < 17:
    Kola1 = cv2.imread('ImagesOfCars\kola%d.jpg' %index)
    #index = 15

    if (Kola1.shape is not (533, 800, 3)):
        Kola1 = cv2.resize(Kola1, (800, 533))

    # pts = np.array([[150, 190], [150, 420], [410, 420], [410, 190]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # plt.imshow(pts)
    # plt.show()

    print('Velicina oblika ' + str(Kola1.shape))

    #Kola1 = cv2.cvtColor(Kola1, cv2.COLOR_BGR2RGB)
    iscrtana, konturee, VrednostPraga = ZaDetekciju(Kola1)

    if( VrednostPraga < 116 and VrednostPraga != 0):
        lower = np.array([140, 140, 140], dtype="uint8")     # Za bolje detektovanje ....!!!! 160
    elif VrednostPraga > 116 and VrednostPraga < 150:
        lower = np.array([90, 90, 90], dtype="uint8")     # 160 malo spustiti granicu
    elif(VrednostPraga >150 and VrednostPraga < 190):
        lower = np.array([170, 170, 170], dtype="uint8")        #Zbog boje asfalta
    else:
        lower = np.array([200, 200, 200], dtype="uint8")        # Izuzetak zbog bolje binarizacije !!!!!!!!!!

    upper = np.array([255, 255, 255], dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(Kola1, lower, upper)
    output = cv2.bitwise_and(Kola1, Kola1, mask=mask)
    #output2 = 255 - output  # Ako su kola u mraku NE treba invertovati (TREBA DODATNI USLOV)
    plt.imshow(output)
    plt.figure()
    plt.imshow(output, 'gray')
    plt.show()

    iscrtana, contours_Tablica, retSlike = ZaDetekciju(output)

    #if(VrednostPraga < 140 and VrednostPraga > 109  and  contours_Tablica is None ):
    if(len(contours_Tablica) != 0 ):    #Da ne bi pucao program provera !!!!!!!!!!!!!!!!!!!!!!!!!
        x, y, w, h = cv2.boundingRect(contours_Tablica[0])
        print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))
        center, size, angle = cv2.minAreaRect(contours_Tablica[0])
        print(" A ugao je: " + str(angle))
        if (angle < -30):                       #Nekad bude slika okrenuta po difoltu pa provera potrebna
            angle = angle + 90

        # if(SlikaKola.shape == (480, 640, 3)):
        if w > 130 and w < 320 and h >36 and h < 110:
          #  if(x > 100 and x < 650 and y > 150 and y < 450):
                print("Dobrodosaooo....")
                rotated = ndimage.rotate(output, angle, reshape=True)
                RotiranaBezBoja = ndimage.rotate(Kola1, angle, reshape=True)
                imgRotirana, KonturaKadjeSlikaRotirana, retSlike = ZaDetekciju(rotated)
                xxx, yyy, www, hhh = cv2.boundingRect(KonturaKadjeSlikaRotirana[0])
                center, size, angle = cv2.minAreaRect(KonturaKadjeSlikaRotirana[0])
                plt.imshow(imgRotirana[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
                plt.figure()
                cv2.imwrite('plates\slika%d.png' % index, RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])
                plt.imshow(RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
                index = index + 1
                plt.show()
    else:
        iscrtana, contours_Tablica, retSlike = ZaDetekciju(Kola1)
        if( len(konturee) >  1):                #Ako zahvati vise od jedne konture dodatni uslovi se poostravaju !!!!!!
            center, size, angle = cv2.minAreaRect(contours_Tablica[0])
            print(" A ugao je: " + str(angle))
            if (angle < -30):  # Nekad bude slika okrenuta po difoltu pa provera potrebna
                angle = angle + 90

            for i in range (0,len(konturee)):
                x, y, w, h = cv2.boundingRect(konturee[i])  #bez boje !!!!!!!!!!
                print("Kontura " + str(i) + " _ _ ")
                print(" tacke x: " + str(x) + " A y  je: " + str(y) + " Sirina je::: " + str(w) + "  "+ str(h) )
                if w > 130 and w < 320 and h > 50 and h < 120:      # Ovde samo ojacavamo uslov za visinu da ne moze biti manja !!!! h > 63
                    if (x > 100 and x < 650 and y > 150 and y < 450):

                        rotated = ndimage.rotate(output, angle, reshape=True)
                        RotiranaBezBoja = ndimage.rotate(Kola1, angle, reshape=True)
                        cv2.imwrite('plates\slika%d.png' % index, RotiranaBezBoja[y:y + h, x:x + w])
                        plt.imshow(iscrtana[y:y + h, x:x + w])
                        index = index + 1
                        plt.show()
                        break

#Treba da radi !!!!!!!!!!!!!!!!!!!!!
#
# PutanjaNove = 'Images\SamoTablice3.jpg'
# cv2.imwrite(PutanjaNove, Kola1[y:y + h, x:x + w])
#
#
# Tablica = cv2.imread(PutanjaNove)
#
# lowerzZaTablicu =  np.array([0, 0, 0], dtype="uint8")         #np.array([0, 0, 0], dtype="uint8")
# upperZaTablicu  =  np.array([40, 40, 40], dtype="uint8")          #np.array([50, 50, 50], dtype="uint8")
#
# mask = cv2.inRange(Kola1[y:y + h, x:x + w], lowerzZaTablicu, upperZaTablicu)
# KrajnjaTablica = cv2.bitwise_and(Kola1[y:y + h, x:x + w], Kola1[y:y + h, x:x + w], mask=mask)
# plt.imshow(KrajnjaTablica)
# plt.show()


#Treba da radi !!!!!!!!!!!!!!!!!!!!!


# KropovanaSlika = SlikaKola[252:272,288:348]
# plt.imshow(KropovanaSlika)
#plt.show()
# cv2.imwrite('Images\slika%d.jpg' % index, SlikaKola[y:y + h, x:x + w])
# plt.imshow(iscrtana)
# plt.show()

# else:
#     Kola1 = cv2.cvtColor(Kola1, cv2.COLOR_BGR2RGB)
#     img, Kontura, retSlike = ZaDetekciju(Kola1)
#     x, y, w, h = cv2.boundingRect(Kontura[0])
#     print(str(x) + ' Ovo je x kordinsata ,vis: ' + str(y) + ' srina je: ' + str(w) + ' A Visina je: ' + str(h))
#
#     plt.imshow(img[y:y + h, x:x + w])
#     plt.show()