from datetime import datetime

import cv2
import numpy as np
import face_recognition
import os


def findEncoding(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgEnc = face_recognition.face_encodings(img)[0]
        encodingList.append(imgEnc)
    return encodingList


def markAttendance(name):
    with open('Entries.csv', '+r') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in myDataList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

encodedList = findEncoding(images)
print(len(encodedList))

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    curFrameLoc = face_recognition.face_locations(imgs)
    curFrameEnc = face_recognition.face_encodings(imgs, curFrameLoc)

    for faceEnc, faceLoc in zip(curFrameEnc, curFrameLoc):
        compare = face_recognition.compare_faces(encodedList, faceEnc)
        distance = face_recognition.face_distance(encodedList, faceEnc)
        print(distance)
        matchIndex = np.argmin(distance)

        if compare[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            # y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        cv2.imshow("Webcam", img)
        cv2.waitKey(1)
