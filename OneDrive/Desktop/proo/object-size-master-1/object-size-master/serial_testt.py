
# importing the python open cv library
import cv2
import numpy as np
from PIL import Image
# intialize the webcam and pass a constant which is 0
import serial
import os
import time
from pathlib import Path
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
from tkinter import *
from PIL import Image,ImageTk
import os
win=Tk()
win.title("two way")
win.geometry("1500x750")
img=Image.open("aa.jpg")
img=img.resize((1500,750))
bg=ImageTk.PhotoImage(img)
lbl=Label(win,image=bg)
lbl.place(x=0,y=0)


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
global Data
# Data= None
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

def func():
######################################
    cam = cv2.VideoCapture(0)
    while True:
        Data=""
        global Detected

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
    cv2.destroyAllWindows()
    from PIL import Image



    from PIL import Image
    #Read the two images
    image1 = Image.open('images\Corner.png')
    ##image1.show()
    image2 = Image.open('wall.jpg')
    ##image2.show()
    #resize, first image
    #image1 = image1.resize((426, 240))
    image1_size = image1.size
    #image2_size = image2.size
    new_image = Image.new('RGB',image2.size, (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save("images/merged_image.jpg","png")
    ##new_image.show()


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
    cv2.imwrite("circle.jpg",img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()



    def show_images(images):
        for i, img in enumerate(images):
            cv2.imshow("image_" + str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img_path = "images/merged_image.jpg"

    # Read image and preprocess
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    #show_images([blur, edged])

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)

    #show_images([image, edged])
    #print(len(cnts))

    # Reference object dimensions
    # Here for reference I have used a 2cm x 2cm square
    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm




    def rectangle_diagonal(length, width):
        diagonal = math.sqrt(length**2 + width**2)
        return diagonal
    lenght=0
    width=0
    counter=0
    # Draw remaining contours
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        counter=counter+1
        if counter==2:
                    counter=0
                    ret=rectangle_diagonal(ht,wid)
                    ret=round(ret,1)
                    cv2.putText(image, 'Dia: '+str(ret), (70,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    print('Diameter : {}'.format(ret))

                    if 0.1<=ret<=1.0:
                        cv2.putText(image, 'Dia: '+str(ret), (100,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        print("The Associated Screw is:{} ".format('M1'))
                        res="The Associated Screw is: {}".format('M1')
                    if 1.1<=ret<=1.3:
                        print("The Associated Screw is:{} ".format('M1.2'))
                        res="The Associated Screw is:{}".format('M1.2',)	

                    if 1.4<ret<=1.5:
                        print("The Associated Screw is: {}  ".format('M1.4'))
                        res="The Associated Screw is:{}".format('M1.4')

                    if 1.6<=ret<=1.79:
                        print("The Associated Screw is:{} ".format('M1.6'))	
                        res="The Associated Screw is:{}  ".format('M1.6')


                    if 1.8<=ret<=1.9:
                        print("The Associated Screw is:{} ".format('M1.8'))
                        res="The Associated Screw is:{} ".format('M1.8')



                    if 2.0<=ret <=2.4:#2.0,2.4
                        print("The Associated Screw is:{} ".format('M2'))
                        res="The Associated Screw is:{}  ".format('M2')


                    if 2.5<=ret<=2.9:
                        print("The Associated Screw is:{}  ".format('M2.5',))
                        res="The Associated Screw is:{}".format('M2.5')

                    if 3.0<=ret<=3.4:
                        print("The Associated Screw is:{}  ".format('M3'))
                        res="The Associated Screw is:{}  ".format('M3')


                    if 3.5<=ret<=3.9:
                        print("The Associated Screw is:{} ".format('M3.5'))
                        res="The Associated Screw is:{}  ".format('M3.5')
                    if 4.0<=ret<=4.9:
                        print("The Associated Screw is:{} ".format('M4'))
                        res="The Associated Screw is:{}  ".format('M4')

                    if 5.0<=ret<=5.9:
                        print("The Associated Screw is:{} ".format('M5'))
                        res="The Associated Screw is:{}".format('M5')                                        


                    if 6.0<=ret<=6.9:
                        print("The Associated Screw is:{} ".format('M6'))
                        res="The Associated Screw is:{}  ".format('M6')
                    if 7.0<=ret<=7.9:
                        print("The Associated Screw is:{}  ".format('M7'))
                        res="The Associated Screw is:{}  ".format('M7')



                    if 8.0<=ret<=9.9:
                        print(" The Associated Screw is:{} ".format('M8'))
                        res="The Associated Screw is:{}  ".format('M8')

                    if 10.0<=ret<=11.9:
                        print("The Associated Screw is:{} ".format('M10' ))
                        res="The Associated Screw is:{}  ".format('M10')
                    output.configure(text=res)

   
                    
    show_images([image])



label=Label(win,text="Hole and Associated Fastener Identification",font=("times",32,"bold"),bg="black",fg="white")
label.place(x=430,y=180)

labelb=Button(win,text="Start Checking",command=func,font=("times",24,"bold"),bg="brown",fg="white",height=3,width=20)
labelb.place(x=630,y=300)
global output
output=Label(win,text="",font=("times",24,"bold"),bg="black",fg="white")
output.place(x=600,y=500)


win.mainloop()