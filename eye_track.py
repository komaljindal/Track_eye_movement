import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    print(ear)
    return ear


def calculate_gaze(left_points, right_points, leftxy, rightxy):
    try:
        gaze = 0
        A = (dist.euclidean(left_points[0], leftxy) + dist.euclidean(right_points[0], rightxy)) / 2.0
        B = (dist.euclidean(left_points[3], leftxy) + dist.euclidean(right_points[3], rightxy)) / 2.0
        if (0.75 * B > A):
            gaze = 1  # left
        elif (0.75 * A > B):
            gaze = 2  # right
        else:
            gaze = 3  # center
        return gaze
    except:
        return 0


def contouring(thresh, mid, img, right=False):
    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 255,0), 2)
        return [cx, cy]
    except:
        pass


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
EYE_AR_THRESH = 0.2
PREVIOUS = "center"
cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

#cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)


def nothing(x):
    pass


#cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while (True):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 75 # cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)
        leftxy = contouring(thresh[:, 0:mid], mid, img)
        rightxy = contouring(thresh[:, mid:], mid, img, True)
        # print(leftxy)
        earleft = eye_aspect_ratio(shape[36:42])
        earright = eye_aspect_ratio(shape[42:48])
        ear = (earleft + earright) / 2.0
        blink = 0
        state = None

        gaze = calculate_gaze(shape[36:42], shape[42:48], leftxy, rightxy)
        time.sleep(1)
        if (ear < EYE_AR_THRESH):
            blink = 1
        if (blink):
            state = "Blink"
        elif (gaze == 1):
            state = "Left"
        elif (gaze == 2):
            state = "Right"
        elif (gaze == 3):
            state = "Center"
        else:
            state = "Not detected"

        print(state)
        PREVIOUS = state
        cv2.putText(img, state, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 0), 2)
        #for (x, y) in shape[36:48]:
         #   cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('Saathi 2.0', img)
    #cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()