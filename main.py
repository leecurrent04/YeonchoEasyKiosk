# https://github.com/smahesh29/Gender-and-Age-Detection
#A Gender and Age Detection program by Mahesh Sawant

# Gender&Age Detection
import cv2
import math
import argparse

# PyQt5
import os, sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

# ==================================================
# Higtlight Face with DNN

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# ==================================================

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="./resource/kernel/opencv_face_detector.pbtxt"
faceModel="./resource/kernel/opencv_face_detector_uint8.pb"
ageProto="./resource/kernel/age_deploy.prototxt"
ageModel="./resource/kernel/age_net.caffemodel"
genderProto="./resource/kernel/gender_deploy.prototxt"
genderModel="./resource/kernel/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=[1,2,3,4,5,6,7,8]
#ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=[1,0]
#genderList=["Male","FeMale"]

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# ==================================================

def LoadCustomersInfo():
    video=cv2.VideoCapture(args.image if args.image else 0)
    print(video.isOpened())

    padding=20

    gender=None
    age=None

    sum_age=0
    sum_gender=0
    repeat_num=5

    if video.isOpened():
        for i in range(0,repeat_num):
            hasFrame,frame=video.read()
            
            resultImg,faceBoxes=highlightFace(faceNet,frame)
            if not faceBoxes:
                continue

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Gender
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()

                # Age
                ageNet.setInput(blob)
                agePreds=ageNet.forward()

                # Calculate the Oldest, if many people are detected
                if(age != None)and(age < ageList[agePreds[0].argmax()]):
                    age = ageList[agePreds[0].argmax()]
                    gender = genderList[genderPreds[0].argmax()]
                elif age == None:
                    age = ageList[agePreds[0].argmax()]
                    gender = genderList[genderPreds[0].argmax()]

            #print(gender,age)
            sum_age+=age
            sum_gender += gender
        
        video.release()

        #print("{}\t{}".format(sum_gender/repeat_num,sum_age/repeat_num))
        return sum_gender/repeat_num, sum_age/repeat_num

# ==================================================
# Load UI

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path,"resource/UI/" ,relative_path)

form_class = uic.loadUiType(resource_path("food_main.ui"))[0]
form_receivement = uic.loadUiType(resource_path("food_receivement.ui"))[0]
form_chooseE = uic.loadUiType(resource_path("food_choose_E.ui"))[0]
form_chooseNE = uic.loadUiType(resource_path("food_choose_NE.ui"))[0]

# ==================================================
# FirstScreen

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.Btn_Start.clicked.connect(self.btn_start_click)

    def btn_start_click(self):
        CustomerGender, CustomerAge = LoadCustomersInfo()

        self.hide()                     # Hide main(this) Window

        if CustomerAge >= 6 or CustomerAge <= 2:
            self.second = chooseEasyWindow()
        else:
            self.second = chooseNEasyWindow()


        self.second.exec()              # wait until when Second window is closed
        self.show()                     # Show main window

# ==================================================
# choose Easy

class chooseEasyWindow(QDialog,QWidget,form_chooseE):
    def __init__(self):
        super(chooseEasyWindow,self).__init__()
        self.initUi()
        self.show()

        self.Btn_back.clicked.connect(self.BtnBackClicked)

        self.Btn_bulgogiBuger.clicked.connect(self.BtnBulgogiClicked)
        self.Btn_bulgogiBugerImg.clicked.connect(self.BtnBulgogiClicked)

        self.Btn_cheeseBuger.clicked.connect(self.BtnCheeseClicked)
        self.Btn_cheeseBugerImg.clicked.connect(self.BtnCheeseClicked)

        self.Btn_chickenBuger.clicked.connect(self.BtnChickenClicked)
        self.Btn_chickenBugerImg.clicked.connect(self.BtnChickenClicked)

    def initUi(self):
        self.setupUi(self)

    def BtnBackClicked(self):
        self.close()                    #클릭시 종료됨.

    def BtnBulgogiClicked(self):
        tempNum = int(self.Lbl_bulgogi.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_bulgogi.setText(str(tempNum))

    def BtnCheeseClicked(self):
        tempNum = int(self.Lbl_cheese.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_cheese.setText(str(tempNum))

    def BtnChickenClicked(self):
        tempNum = int(self.Lbl_chicken.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_chicken.setText(str(tempNum))

# ==================================================
# choode NEasy

class chooseNEasyWindow(QDialog,QWidget,form_chooseNE):
    def __init__(self):
        super(chooseNEasyWindow,self).__init__()
        self.initUi()
        self.show()

        self.Btn_back.clicked.connect(self.BtnBackClicked)

        self.Btn_bulgogiBugerImg.clicked.connect(self.BtnBulgogiClicked)
        self.Btn_cheeseBugerImg.clicked.connect(self.BtnCheeseClicked)
        self.Btn_chickenBugerImg.clicked.connect(self.BtnChickenClicked)
        self.Btn_filetofishBugerImg.clicked.connect(self.BtnFiletofishClicked)
        self.Btn_doublefiletofishBugerImg.clicked.connect(self.BtnDoubleFiletofishClicked)
        self.Btn_bigmacBugerImg.clicked.connect(self.BtnBigmacClicked)

    def initUi(self):
        self.setupUi(self)

    def BtnBackClicked(self):
        self.close()                    #클릭시 종료됨.

    def BtnBulgogiClicked(self):
        tempNum = int(self.Lbl_bulgogi.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_bulgogi.setText(str(tempNum))

    def BtnCheeseClicked(self):
        tempNum = int(self.Lbl_cheese.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_cheese.setText(str(tempNum))

    def BtnChickenClicked(self):
        tempNum = int(self.Lbl_chicken.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_chicken.setText(str(tempNum))

    def BtnFiletofishClicked(self):
        tempNum = int(self.Lbl_filetofish.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_filetofish.setText(str(tempNum))

    def BtnDoubleFiletofishClicked(self):
        tempNum = int(self.Lbl_doublefiletofish.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_doublefiletofish.setText(str(tempNum))

    def BtnBigmacClicked(self):
        tempNum = int(self.Lbl_bigmac.text())
        if not tempNum >= 9:
            tempNum+=1

        self.Lbl_bigmac.setText(str(tempNum))

# ==================================================
# from_receivement

class receivementWindow(QDialog,QWidget,form_receivement):
    def __init__(self):
        super(receivementWindow,self).__init__()
        self.initUi()
        self.show()

        self.Btn_inside.clicked.connect(self.BtnInsideClicked)
        self.Btn_outside.clicked.connect(self.BtnOutsideClicked)
        self.Btn_back.clicked.connect(self.BtnBackClicked)

    def initUi(self):
        self.setupUi(self)

    def BtnInsideClicked(self):
        self.close()                    #클릭시 종료됨.

    def BtnOutsideClicked(self):
        self.close()                    #클릭시 종료됨.

    def BtnBackClicked(self):
        self.close()                    #클릭시 종료됨.


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()


