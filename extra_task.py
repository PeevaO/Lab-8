import cv2
import math

pattern = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
fly = cv2.imread('fly64.png')
fly_height, fly_width = fly.shape[:2]
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(pattern, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    matches = bf.match(des1, des2)
    get_distance = lambda x: x.distance
    matches = sorted(matches, key=get_distance)

    if len(matches) > 0:
        top_match = matches[0]
        ref_center_x = pattern.shape[1] // 2
        ref_center_y = pattern.shape[0] // 2
        match_center_x = int(kp2[top_match.trainIdx].pt[0])
        match_center_y = int(kp2[top_match.trainIdx].pt[1])

        distance = math.sqrt((ref_center_x - match_center_x) ** 2 + (ref_center_y - match_center_y) ** 2)
        x = match_center_x - fly_width // 2
        y = match_center_y - fly_height // 2

        roi = frame[y:y + fly_height, x:x + fly_width]
        overlay = cv2.addWeighted(roi, 0.5, fly, 0.5, 0)
        frame[y:y + fly_height, x:x + fly_width] = overlay

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()