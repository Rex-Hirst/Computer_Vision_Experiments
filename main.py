import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector

# Start the Video Capture on main camera
cap = cv2.VideoCapture(0)

# Creates an MOG2 Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

while True:

    # Sets ret (a boolean variable to check if frame is TRUE) and frame (an image array vector)
    ret, frame = cap.read()

    # Checks if there is a frame
    if ret:

        # Foreground mask = Background Subtraction -> applied to frame
        FG_Mask = backSub.apply(frame)

        # Approx simple takes 4 points
        contours, hierarchy = cv2.findContours(FG_Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on the contours of an image in blue
        frame_ct = cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)

        retval, mask_thresh = cv2.threshold(FG_Mask, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)
        cv2.imshow('Frame_final', frame_out)
    # Turn off Camera with "q" key
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()