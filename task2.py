import cv2

SIZE_SQUARE = 150

pattern = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)
flip = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    width, height = pattern.shape[::-1]

    if max_val > 0.60:
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        center_y = frame_height // 2
        rect_x = center_x - SIZE_SQUARE // 2
        rect_y = center_y - SIZE_SQUARE // 2
        center_rect = (rect_x, rect_y, )

        cv2.rectangle(frame, (rect_x, rect_y),
                      (rect_x + SIZE_SQUARE, rect_y + SIZE_SQUARE), (0, 255, 255), 2)

        label_center_x = top_left[0] + width // 2
        label_center_y = top_left[1] + height // 2

        if (rect_x <= label_center_x <= rect_x + SIZE_SQUARE
                and rect_y <= label_center_y <= rect_y + SIZE_SQUARE):
            frame = cv2.flip(frame, -1)

    cv2.imshow('Shot', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
