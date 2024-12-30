
# importing the python open cv library
import cv2
import numpy as np
from PIL import Image
# intialize the webcam and pass a constant which is 0
import serial
import os
import time
from pathlib import Path

cam = cv2.VideoCapture(0)
#cv2.namedWindow('webcam screenshot ')
img_counter = 0
data = serial.Serial(
                    'COM3',
                    baudrate = 9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=1
                    )
# while loop
Detected=0

################define########################
def getPosOfThirdAndFourthPoints(y1, x1, y2, x2):
    r = int((x2 - x1) / 2)
    xc = x2 - r
    yc = y2
    x3 = int(r / np.sqrt(2))
    y3 = x3
    x4 = -x3
    y4 = -y3
    x3 = x3 + xc
    y3 = y3 + yc
    x4 = x4 + xc
    y4 = y4 + yc

    return [x4, y4], [x3, y3], r, [xc, yc]


def checkIfthosPointsIsOnACircle(p3, p4, boundary):
    height, width = boundary.shape[:2]
    row = boundary[p4[1]]
    i = -2
    point3IsOnCircle = False
    if p3[0] + 2 < width and p4[0] + 2 < width:
        while i != 2:
            row_value = row[p3[0] + i]
            if row_value == 255:
                point3IsOnCircle = True
                break
            else:
                i = i + 1
    if point3IsOnCircle:
        while i != 2:
            row_value = row[p4[0] + i]
            if row_value == 255:
                return True
            else:
                i = i + 1

    return False


def getCircles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3, 3), np.uint8)
    ret, img_binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(~img_binary, kernel)
    boundary = (~img_binary) - erode
    prev_column = 0
    RowIndex = 0
    ColumnIndex = 0
    p1 = [0, 0]
    p1IsObtained = False
    Circles = []
    Circles_count = 0
    for ROW in boundary:
        for colVal in ROW:
            if colVal == 255 or p1IsObtained:
                if p1IsObtained:
                    p2 = [RowIndex, ColumnIndex]
                    p3, p4, r, C = getPosOfThirdAndFourthPoints(p1[0], p1[1], p2[0], p2[1])
                    CircleIsDetected = checkIfthosPointsIsOnACircle(p3, p4, boundary)
                    if CircleIsDetected:
                        Circles.insert(Circles_count, [C, r])
                        Circles_count = Circles_count + 1
                    p1IsObtained = False
                if colVal == 255:
                    prev_column = 255

            else:
                if prev_column == 255:
                    p1 = [RowIndex, ColumnIndex]
                    p1IsObtained = True
            ColumnIndex = ColumnIndex + 1
        RowIndex = RowIndex + 1  # increment the ROW index
        ColumnIndex = 0

    return Circles


######################################
while True:


    if data.inWaiting() > 0:
      Data = data.readline()
      Data = Data.decode('utf-8', 'ignore')
      print("Raw data is ---- {}  ---".format(Data))

      if '\00' in str(Data):
        Detected=1
        print('Wall Detected ')
    # intializing the frame, ret
    ret, frame = cam.read()
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    #to get continuous live video feed from my laptops webcam
    k  = cv2.waitKey(1)
    # if the escape key is been pressed, the app will stop
    if k%256 == 27:
        print('escape hit, closing the app')
        break
    # if the spacebar key is been pressed
    # screenshots will be taken
    elif k%256  == 32:
        # the format for storing the images scrreenshotted
        img_name = f'opencv_frame_{img_counter}'
        # saves the image as a png file
        cv2.imwrite(img_name, frame)
        print('screenshot taken')
        # the number of images automaticallly increases by 1
        img_counter += 1
    elif Detected==1:
        Detected=0
        img_name = 'wall.jpg'
        # saves the image as a png file
        cv2.imwrite(img_name, frame)
        print('screenshot taken')
        break

# release the camera
cam.release()
from PIL import Image



from PIL import Image
#Read the two images
image1 = Image.open('images\Corner.png')
image1.show()
image2 = Image.open('wall.jpg')
image2.show()
#resize, first image
#image1 = image1.resize((426, 240))
image1_size = image1.size
#image2_size = image2.size
new_image = Image.new('RGB',image2.size, (250,250,250))
new_image.paste(image1,(0,0))
new_image.paste(image2,(image1_size[0],0))
new_image.save("images/merged_image.jpg","png")
new_image.show()


### Load the smaller and larger images
##small_image = Image.open("images\\Corner.png")
##small_image = small_image.convert("RGB")
##large_image = Image.open("wall.jpg")
##large_image = large_image.convert("RGB")
##
##
### Create a new image that is the same size as the larger image
##new_image = Image.new("RGB",large_image.size,color=(b, g, r))
### Save the new image
### Paste the smaller image at the top left corner of the new image
##new_image.paste(small_image, (0, 0))
##
##new_image.save("merged_image.png")
##
##

img = cv2.imread(r'images\merged_image.jpg')
circles = getCircles(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for circle in circles:
    center = circle[0]
    r = circle[1]
    cv2.circle(img, (center[0], center[1]), r, (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (center[0], center[1]), 2, (0, 0, 255), 3)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

