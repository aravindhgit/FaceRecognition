import cv2

import face_recognition
import numpy as np

orgElon = face_recognition.load_image_file('images/elon.jpg')
orgElon = cv2.cvtColor(orgElon,cv2.COLOR_BGR2RGB)

testElon = face_recognition.load_image_file('images/bill-gates.jpg')
testElon = cv2.cvtColor(testElon,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(orgElon)[0]
faceEnc = face_recognition.face_encodings(orgElon)[0]
cv2.rectangle(orgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255, 0, 255),2)

faceLocTest = face_recognition.face_locations(testElon)[0]
faceEncTest = face_recognition.face_encodings(testElon)[0]
cv2.rectangle(testElon, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([faceEnc],faceEncTest)
faceDist = face_recognition.face_distance([faceEnc],faceEncTest)
print(result,faceDist)

cv2.putText(orgElon,f'{result} {round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',orgElon)
cv2.imshow("Elon",testElon)

cv2.waitKey(0)