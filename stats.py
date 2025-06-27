import os
import sys
import cv2
from ultralytics import YOLO

model_path = 'my_model.pt'  # Path to model
min_thresh = 0.50                      # Minimum detection threshold
cam_index = 0                          # Index of USB camera
imgW, imgH = 1280, 720                 # Resolution to run USB camera at
record = False                         # Record result video

nutrition_info = {'starburst': [40, 6.6],'lollipop': [20, 4], 'hichew': [21, 3.5], 'nerds': [70, 20]}

if (not os.path.exists(model_path)):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

model = YOLO(model_path, task='detect')
labels = model.names

cap = cv2.VideoCapture(cam_index)
ret = cap.set(3, imgW)
ret = cap.set(4, imgH)

if record == True:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (imgW,imgH))

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133)]

while True:
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break
    results = model.track(frame, verbose=False)
    detections = results[0].boxes
    candies_detected = []
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        if conf > 0.5:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            candies_detected.append(classname)
    total_calories = 0
    total_sugar = 0

    for candy_name in candies_detected:
        calories, sugar = nutrition_info[candy_name]
        total_calories += calories
        total_sugar += sugar
    cv2.rectangle(frame, (10, 10), (450, 130), (50,50,50), cv2.FILLED) # Rectangle to draw text on
    cv2.putText(frame, f'Number of candies: {len(candies_detected)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,102,51), 2)
    cv2.putText(frame, f'Total calories: {total_calories}', (20,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (51,204,51), 2)
    cv2.putText(frame, f'Total sugar (g): {total_sugar}', (20,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,204,255), 2)
    cv2.imshow('Candy detection results',frame) # Display image
    if record: recorder.write(frame) # Record frame to video (if enabled)
    key = cv2.waitKey(5)
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

cap.release()
if record: recorder.release()
cv2.destroyAllWindows()
