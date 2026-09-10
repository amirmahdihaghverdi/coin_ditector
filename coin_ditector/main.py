import cv2

face_detector = cv2.CascadeClassifier("cascade/cascade.xml")

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()

    faces = face_detector.detectMultiScale(frame)

    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) =="q":
        break

cv2.destroyAllWindows()
