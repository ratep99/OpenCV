import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

#Definisemo text i font
def writeText(text:str, x:int, y:int):
    cv2.putText(img,  text, (int(x- len(text)* 4),int(y+3.5)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (10 ,10,10), 1)

#Definisemo putanju sa koje citamo sliku
img = cv2.imread("C:/OpenCV/17200_17339_Projekat2/OpenCVImage.png")

#Dictionary ("numberOfAngles","Name")
shapeDict:dict = {3:"triangle",4:"rectangle",5:"pengaton",10:"star"}

#Konverzija boja iz jednog u drugi "svet"
gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
#Numpy array opsezi
lower_yellow = np.array([25,70,120])
upper_yellow = np.array([30,255,255])

lower_green = np.array([40,70,80])
upper_green = np.array([70,255,255])

lower_red = np.array([0,50,120])
upper_red = np.array([10,255,255])

lower_red1 = np.array([170,50,50])
upper_red1 = np.array([180,255,255])

lower_blue = np.array([90,60,0])
upper_blue = np.array([121,255,255])

#Maske za propoznavanje boja u definisanim opsezima 

maskList:list = list()
maskList.append(("Yellow",cv2.inRange(hsv,lower_yellow,upper_yellow)))
maskList.append(("Green",cv2.inRange(hsv,lower_green,upper_green)))
maskList.append(("Red",cv2.inRange(hsv,lower_red,upper_red) + cv2.inRange(hsv, lower_red1, upper_red1)))
maskList.append(("Blue",cv2.inRange(hsv, lower_blue,upper_blue)))


#Pribavljanje liste kontura po bojama

lista:list = list()
lista = map(lambda x: (x[0], imutils.grab_contours(cv2.findContours(x[1],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE))),maskList)

#Lista lista
#Prolazak po konturama i po bojama
#counturs[0] <- Boja
#counturs[1] <- Lista kontura u toj boji

for contours in lista:
    for contour in contours[1]:
        approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [contour], 0, (0, 0, 0), 1)

        approx = len(cv2.approxPolyDP(  #approx <- pamti broj uglova mnogougla
        contour, 0.01 * cv2.arcLength(contour, True), True))
        M = cv2.moments(contour) #Trazi centar figure kako bi ispisao u njoj oblik i boju
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

       
        if approx in shapeDict.keys():
            writeText( contours[0] +" "+ shapeDict[approx],x,y)
        else:
            writeText( contours[0] +" circle",x,y)  

cv2.imshow('Projekat II - OpenCV', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
