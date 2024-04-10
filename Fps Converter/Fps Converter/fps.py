import cv2
from darknet import load_net, load_meta, detect

# Load YOLO model and metadata
net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
meta = load_meta(b"cfg/coco.data")

# Open camera module
cap = cv2.VideoCapture('demo.mp4')  # Change the parameter if using a different camera

frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Check if the current frame is the 60th frame (1 FPS)
    if frame_count == 60:
        # Perform YOLO object detection inference
        detections = detect(net, meta, frame)
        
        # Draw bounding boxes around detected objects
        for detection in detections:
            label = detection[0]
            confidence = detection[1]
            bounds = detection[2]
            x, y, w, h = bounds
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h / 2)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label.decode() + ' ' + str(round(confidence, 2)), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with detections
        cv2.imshow('Frame', frame)
        
        # Reset frame count
        frame_count = 0

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()