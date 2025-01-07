import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.5)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            conf = box.conf[0]
            cls = int(box.cls[0])

            image_np = np.array(frame)
            cropped_plate = image_np[y1:y2, x1:x2]
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            resized_plate = cv2.resize(gray_plate, (320, 96))
            ocr_results = ocr.ocr(resized_plate, cls=True)

            if ocr_results and ocr_results[0]:  
                license_text = ''.join([line[1][0] for line in ocr_results[0]])
                label = f"{license_text}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                label = f"No text detected"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
