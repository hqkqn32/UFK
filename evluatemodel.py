import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/hqkqn/Desktop/yolov5/runs/train/exp/weights/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    results = model(frame)

    detections = results.pandas().xyxy[0]

    uav_detections = detections[(detections['confidence'] >= 0.70) & (detections['name'] == 'uav')]

    for _, detection in uav_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        label = f"UAV {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('UAV Tespiti', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
