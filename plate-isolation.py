import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime


# Funkcija koja vraca detekciju na sliku
def ZaDetekciju(Okvir):
    img_Car_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    plt.imshow(img_Car_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_Car_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61,
                                              30)
    plt.imshow(image_barcode_bin, 'gray')
    img, contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = Okvir.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    plt.imshow(img)

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 35 and w < 220 and h > 10 and h < 100:  # uslov da kontura pripada bar-kodu
            if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu

    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 1)
    plt.imshow(img)
    # plt.show()
    return img


def diffImg(t0, t1, t2):  # Function to calculate difference between images.
    # t0 = t0[150:410,190:420]
    # t1 = t1[150:410, 190:420]
    # t2 = t2[150:410, 190:420]

    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


threshold = 81500
cap = cv2.VideoCapture("cars.mp4")

winName = "Movement Indicator"  # comment to hide window
cv2.namedWindow(winName)

t_minus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)



# Kropovati ih da se vrsi fetekcija na smo tom regiom !!!!!!!!!!!!!!

# t_minus = t_minus[150:410, 190:420]
# t = t[150:410, 190:420]
# t_plus = t_plus[150:410, 190:420]
# Da svake sekunde uzimamo sliku ............
timeCheck = datetime.now().strftime('%Ss')
index = 0

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    pts = np.array([[150, 190], [150, 420], [410, 420], [410, 190]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255))



    if ret == True and t_plus is not None:
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold and timeCheck != datetime.now().strftime('%Ss'):
            dimg = cap.read()[1]
            # cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
            cv2.imwrite('ImagesFrame2\slika%d.jpg' % index, dimg)
            index = index + 1
            # Sik = ZaDetekciju(dimg)
            # cv2.imshow(winName, Sik)

        timeCheck = datetime.now().strftime('%Ss')

        if( cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY) is not None ):
            # Read next image
            t_minus = t
            t = t_plus
            t_plus = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        frame = ZaDetekciju(frame)

        cv2.imshow(winName, frame)
        # cv2.imshow(winName, cap.read()[1])  # comment to hide window
        # Display the resulting frame
        if cv2.waitKey(85) & 0xFF == ord('q'):          #85 #1
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
