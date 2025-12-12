
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import time

# Load Emotion Model
# Load without compiling to avoid restoring optimizer args (e.g. 'lr') that may be incompatible
model = tf.keras.models.load_model("models/emotion_model.hdf5", compile=False)

# Emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Mediapipe initialization
mp_face = mp.solutions.face_detection
mp_landmarks = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

face_data_list = []

with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection, \
     mp_landmarks.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_results = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)

        if face_results.detections:
            for det in face_results.detections:
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Convert relative → absolute bounding box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)

                # Draw box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)

                # Crop face for emotion model
                crop = frame[y:y + h_box, x:x + w_box]

                if crop.size != 0:
                    face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (64, 64))
                    face_gray = face_gray.astype("float") / 255.0
                    face_gray = np.reshape(face_gray, (1, 64, 64, 1))

                    # Predict emotion
                    preds = model.predict(face_gray)
                    emotion = EMOTIONS[np.argmax(preds)]

                    # Draw emotion label
                    cv2.putText(frame, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    emotion = "Unknown"

                # Save data
                face_data = {
                    "timestamp": time.time(),
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": w_box,
                        "height": h_box
                    },
                    "emotion": emotion,
                    "confidence": float(det.score[0])
                }

                # Extract 468 landmark points
                landmark_points = []
                if mesh_results.multi_face_landmarks:
                    for face_landmark in mesh_results.multi_face_landmarks:
                        for lm in face_landmark.landmark:
                            landmark_points.append({
                                "x": lm.x,
                                "y": lm.y,
                                "z": lm.z
                            })

                face_data["landmarks"] = landmark_points
                face_data_list.append(face_data)

        cv2.imshow("Mediapipe Face Detection + Emotion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
