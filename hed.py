import cv2
import numpy as np

net = cv2.dnn.readNet("путь/к/yolov3.weights", "путь/к/yolov3.cfg")

layer_names = net.getUnconnectedOutLayersNames()

image1 = cv2.imread("beautiful_puppy.jpg")
height, width = image1.shape[:2]

blob = cv2.dnn.blobFromImage(image1, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

outs = net.forward(layer_names)

conf_threshold = 0.5

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            pattern_image1 = cv2.imread("dzen_infra.jpg")
            pattern_image1 = cv2.resize(pattern_image1, (w, h))
            image1[y:y+h, x:x+w] = pattern_image1

image2 = cv2.imread("purina.jpg")
blob2 = cv2.dnn.blobFromImage(image2, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob2)
outs2 = net.forward(layer_names)

for out in outs2:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            pattern_image2 = cv2.imread("puppy_overlay.jpg")
            pattern_image2 = cv2.resize(pattern_image2, (w, h))
            image2[y:y+h, x:x+w] = pattern_image2

cv2.imshow("Результат 1", image1)
cv2.imshow("Результат 2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
