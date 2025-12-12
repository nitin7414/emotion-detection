# import cv2
# from keras.models import load_model
# import numpy as np

# #Load pre trained face detector and emotion model

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# emotional_model = load_model(r"C:\Users\pc\Desktop\python\.venv\Lib\site-packages\emotional_model.h5")   # This is a pre trained CNN model
# emotional_labels = ['neutral','Angry','Disgust', 'Fear','Happy','Sad','Surprise']

# #start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     #Read video Frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     #Convert frame to grayscale
#     gray =cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

#     #Detect FAces
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:
#         #Draw rectrangular around face
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

#         #Extract the face region
#         face_roi = gray[y:y+h, x:x+w]
#         face_roi = cv2.resize(face_roi, (48,48))
#         face_roi = face_roi.astype('float') / 255.0
#         face_roi = np.expand_dims(face_roi, axis=0)
#         face_roi = np.expand_dims(face_roi, axis=-1)

#         # Predict emotions
#         preds = emotional_model.predict(face_roi)
#         label = emotional_labels[np.argmax(preds)]

#         #Display emotion label
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36,255,12),2)
#     #show result
#     cv2.imshow("Emotion Detector", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from fer.fer import FER
import cv2

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_emotions(frame)
    for face in result:
        (x, y, w, h) = face["box"]
        emotion, score = max(face["emotions"].items(), key=lambda item: item[1])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
