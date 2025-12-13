import cv2, json, time
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Load Emotion Model
model = tf.keras.models.load_model("models/emotion_model.hdf5", compile=False)

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Mediapipe initialization
mp_face = mp.solutions.face_detection
mp_landmarks = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

log_data = []
cap = cv2.VideoCapture(0)
face_data_list = []
predictions = []

with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection, \
     mp_landmarks.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Detect faces and Landmarks
        face_results = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)

        if face_results.detections:
            for det in face_results.detections:
                det_conf = det.score[0]
                if det_conf < 0.6:
                    continue
                
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = frame.shape

                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)

                
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)

                
                crop = frame[y:y + h_box, x:x + w_box]

                if crop.size != 0:
                    face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (64, 64))
                    face_gray = face_gray.astype("float") / 255.0
                    face_gray = np.reshape(face_gray, (1, 64, 64, 1))

                    # Predict emotion
                    preds = model.predict(face_gray)[0]
                    emotion_index = np.argmax(preds)
                    emotion = EMOTIONS[emotion_index]
                    confidence = float(preds[emotion_index])
                    predictions.append(confidence)
                    # Here we draw the emotion label
                    cv2.putText(frame, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    emotion = "Unknown"

                # Save data
                log_data.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    "Time per Frame": datetime.now().second,
                    "emotion": emotion,
                    "confidence": f"{confidence*100}%"
                })
                df= pd.DataFrame(log_data)
        cv2.imshow("Mediapipe Face Detection + Emotion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(np.array(predictions))
            print(np.mean(predictions))
            print(df)
            plt.plot(df["timestamp"], df["confidence"])
            plt.xlabel("Time")
            plt.ylabel("Confidence")
            plt.title("Emotion Model Confidence Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            break

cap.release()
cv2.destroyAllWindows()
