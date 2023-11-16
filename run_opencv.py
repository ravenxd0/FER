import numpy as np
import cv2 as cv
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('my_mobilenetv2.h5')

emotion_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

cap = cv.VideoCapture(0)
if not cap.isOpened():
    cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Error Opening Video Capture")
    exit(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to Capture frame")
        break

    frame = cv.flip(frame, 1)
    labels = []
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray)

    if len(faces) == 0:
        cv.putText(frame, 'No Faces', (272,242), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv.resize(roi_gray, (224,224), interpolation=cv.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = np.expand_dims(roi_gray, axis=0)
            roi = preprocess_input(roi)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[np.argmax(prediction) ]
            label_position = (x,y)
            cv.putText(frame, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv.imshow("Emotion Recognition", frame)

    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == ord('q') or pressed_key == 27:
        break

cap.release()
cv.destroyAllWindows()
