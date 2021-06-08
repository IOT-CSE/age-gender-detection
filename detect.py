# detecting age gender by classification
import cv2
import math
import argparse

ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
discriminator = argparse.ArgumentParser()
discriminator.add_argument('--image')
args = discriminator.parse_args()
genderList = ['Male', 'Female']


protoTypeFace = "opencv_face_detector.pbtxt"
modelFace = "opencv_face_detector_uint8.pb"
prototypeAge = "age_deploy.prototxt"
modelAge = "age_net.caffemodel"
protoTypegender = "gender_deploy.prototxt"
modelGender = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78, 87, 114)

faceNeural = cv2.dnn.readNet(modelFace, protoTypeFace)
ageNeural = cv2.dnn.readNet(modelAge, prototypeAge)
genderNeural = cv2.dnn.readNet(modelGender, protoTypegender)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

def calculateFace(net, frame, conf_threshold=0.7):
    fOpencvDnn = frame.copy()
    fHeight = fOpencvDnn.shape[0]
    fWidth = fOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(fOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * fWidth)
            y1 = int(detections[0, 0, i, 4] * fHeight)
            x2 = int(detections[0, 0, i, 5] * fWidth)
            y2 = int(detections[0, 0, i, 6] * fHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(fOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(fHeight / 150)), 8)
            
            
    return fOpencvDnn, faceBoxes






while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = calculateFace(faceNeural, frame)
    if not faceBoxes:
        print("No face")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding) :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNeural.setInput(blob)
        genderPreds = genderNeural.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNeural.setInput(blob)
        agePreds = ageNeural.forward()
        age = ages[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting", resultImg)


