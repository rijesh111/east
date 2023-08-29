from typing import List, Tuple

import cv2
import numpy as np

img = cv2.imread("C:\\Users\\rijes\\OCR\\nagrikta front.png")

model = cv2.dnn.readNet(r"C:\Users\rijes\OCR\frozen_east_text_detection.pb")


height, width, _ = img.shape
new_height = (height // 32) * 32
new_width = (width // 32) * 32

h_ratio = height / new_height
w_ratio = width / new_width

mean_values = (123, 68, 116, 78)
blob = cv2.dnn.blobFromImage(img, 1, (new_width, new_height), mean_values, True, False)


model.setInput(blob)
output_layer_names = model.getUnconnectedOutLayersNames()
model.forward(output_layer_names)
(geomtery, scores) = model.forward(output_layer_names)

rectangles = []
confidence_scores = []

for i in range(0, geomtery.shape[2]):
    for j in range(0, geomtery.shape[3]):
        if scores[0][0][i][j] < 0.1:
            continue
        bottom_x = int(j * 4 + geomtery[0][1][i][j])
        bottom_y = int(i * 4 + geomtery[0][1][i][j])
        top_x = int(j * 4 - geomtery[0][1][i][j])
        top_y = int(i * 4 - geomtery[0][1][i][j])
        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_scores.append(float(scores[0][0][i][j]))

indices = cv2.dnn.NMSBoxes(rectangles, confidence_scores, score_threshold=0.5, nms_threshold=0.5)

img_copy = img.copy()
for index in indices:
    x1, y1, x2, y2 = rectangles[index]
    x1 = int(x1 * w_ratio)
    y1 = int(y1 * h_ratio)
    x2 = int(x2 * w_ratio)
    y2 = int(y2 * h_ratio)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Original Image", img)
cv2.imshow("Text Detection", img_copy)
cv2.waitKey(0)











