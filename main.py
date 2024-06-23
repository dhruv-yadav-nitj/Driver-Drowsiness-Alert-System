import cv2
import time
import dlib
from scipy.spatial.distance import euclidean as distance
import pyttsx3
import numpy as np

# PRE-TRAINED FACE DETECTOR
detector = dlib.get_frontal_face_detector()

# TEXT-TO-SPEECH
speaker = pyttsx3.init()

# PRE-TRAINED FACE-LANDMARKS DETECTOR
landmark_detector = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

# ALERT DELAY
DELAY = 3

# TIMER
CURR_TIME = None

# EYE CLOSE THRESHOLD
THRESHOLD = 0.25


def eye_aspect_ratio(eye):
    """
    CALCULATES THE EYE ASPECT RATIO FOR EYE.
    THE RATIO DEFINES THE DEGREE OF EYE OPENING.
    :param eye
    :return EAR = (d1 + d2)/(2 * d3)
    """
    d1 = distance(eye[1], eye[5])
    d2 = distance(eye[2], eye[4])
    d3 = distance(eye[0], eye[3])

    ratio = (d1 + d2)/(2 * d3)
    return ratio


def main():
    """
    STEPS:
    CAPTURE VIDEO -> FACE DETECTION -> EYES LANDMARKS DETECTION -> EAR -> EYE OPEN/CLOSE -> ALERT SYSTEM
    """
    global CURR_TIME
    cap = cv2.VideoCapture(0)
    while True:
        flag, frame = cap.read()
        if flag:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # DETECT AT MOST 1 FACE
            faces = detector(gray, 1)

            for face in faces:
                landmarks = landmark_detector(image=gray, box=face)
                left, right = list(), list()

                # FOR LEFT EYE -> LANDMARK RANGE IS (42, 47]
                for i in range(42, 48):
                    # COORDINATES FOR ith LANDMARK
                    x1 = landmarks.part(i).x
                    y1 = landmarks.part(i).y
                    left.append((x1, y1))

                    # WE NEED TO DRAW THE CIRCUMFERENCE AROUND EYE
                    j = 42 if i == 47 else (i+1)
                    x2 = landmarks.part(j).x
                    y2 = landmarks.part(j).y

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # FOR RIGHT EYE -> LANDMARK RANGE IS (36, 41]
                for i in range(36, 42):
                    # COORDINATES FOR ith LANDMARK
                    x1 = landmarks.part(i).x
                    y1 = landmarks.part(i).y
                    right.append((x1, y1))

                    # WE NEED TO DRAW THE CIRCUMFERENCE AROUND EYE
                    j = 36 if i == 41 else (i+1)
                    x2 = landmarks.part(j).x
                    y2 = landmarks.part(j).y

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # CALCULATING AVG EAR FOR EYES
                right_eye_ratio = eye_aspect_ratio(right)
                left_eye_ratio = eye_aspect_ratio(left)
                mean = np.mean([left_eye_ratio, right_eye_ratio])

                # ALERT SYSTEM
                if mean < THRESHOLD:
                    if CURR_TIME is None:
                        CURR_TIME = time.time()
                    else:
                        time_elapsed = time.time() - CURR_TIME
                        if time_elapsed >= DELAY:
                            print('WAKE UP!')
                            cv2.putText(frame, 'WAKE UP!', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                            if not speaker._inLoop:  # CHECKS IF SPEAKER IS NOT BUSY
                                speaker.say("WAKE UP!")
                                speaker.runAndWait()
                else:
                    CURR_TIME = None

            cv2.imshow('Driver Cam', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
