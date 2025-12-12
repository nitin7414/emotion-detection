import cv2 
import mediapipe as mp
import time

class handDetector():
     def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackingCon = 0.5):
          self.mode = mode
          self.maxHands = maxHands
          self.detectionCon = detectionCon
          self.trackingCon = trackingCon
          self.mphands = mp.solutions.hands
          self.hands = self.mphands.Hands(
              static_image_mode=self.mode,
max_num_hands=self.maxHands,
min_detection_confidence=self.detectionCon,
min_tracking_confidence=self.trackingCon
          )
          self.mpDraw = mp.solutions.drawing_utils

     def findHands(self, img, draw = True): 
            imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRgb)
            # print(results.multi_hand_landmarks)
            if self.results.multi_hand_landmarks:
              for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img,handLms, self.mphands.HAND_CONNECTIONS)
                
            return img   
     def findPosition(self,img, handNo = 0, draw = True):
         lmList = []
         if self.results.multi_hand_landmarks:
          myHand = self.results.multi_hand_landmarks[handNo]
          for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                 h,w,c = img.shape
                 cx, cy = int(lm.x*w),int(lm.y*h)
                 lmList.append([id, cx, cy])
                 if draw:
                  cv2.circle(img, (cx,cy), 17, (255,0,255), cv2.FILLED)
         return lmList

def main():
     cap = cv2.VideoCapture(0)
     cTime = 0
     pTime = 0
     cap.set(3, 640)  # width
     cap.set(4, 480)  # height
     while True:
      sucess, img = cap.read()
      detector = handDetector()
      img = detector.findHands(img)
      cTime = time.time()
      fps = 1 / (cTime-pTime)
      pTime = cTime
      cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
      cv2.imshow("image", img)
      cv2.waitKey(1)
if __name__ == "__main__":
     main()