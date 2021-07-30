import cv2

def faceBox (faceNet,frame):
    blob=cv2.dnn.blobFromImage (frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setINput(blob)
    detection=faceNet.forward()
    print(detection)
    return detection

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet=cv2.dnn.readNet (faceModel, faceProto)

video=cv2.VideoCapture (0)

while True:
    ret, frame=video.read()
    detect = faceBox(faceNet, frame)
    cv2.imshow("Age-Gender", frame)
    k=cv2.waitkey (1)
    if k==ord ('q'):
        break

video.release()
cv2.destroyAllWindows()