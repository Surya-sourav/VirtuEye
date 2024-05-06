import cv2
import numpy as np
from sklearn.cluster import KMeans

def find_majority_group(centers):
    # Use KMeans clustering to find majority group of obstacles
    if len(centers) == 0:
        return []

    centers_np = np.array(centers)
    num_clusters = min(3, len(centers))  # Limit number of clusters to 3 for simplicity
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centers_np)
    
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    majority_cluster_id = unique[np.argmax(counts)]
    
    majority_group = centers_np[kmeans.labels_ == majority_cluster_id]
    return majority_group.tolist()

def is_prominent_obstacle(center, viewer_position, threshold_distance):
    # Check if the distance from the obstacle center to the viewer position is below the threshold
    distance = np.linalg.norm(np.array(center) - np.array(viewer_position))
    return distance < threshold_distance

def draw_obstacles(frame, centers):
    height, width = frame.shape[:2]
    viewer_position = (width // 2, height - 1)  # Viewer is assumed at the bottom center of the frame
    threshold_distance = height // 4  # Define the threshold distance for prominence


    line_width = 250  # Width of the parallel lines
    line_length = height // 2  # Length of the parallel lines
    line_center = (width // 2, height // 2 + height // 5)  # Center point of the parallel lines

    left_line_start = (line_center[0] - line_width // 2, line_center[1])
    left_line_end = (line_center[0] - line_width // 2, line_center[1] + line_length)
    right_line_start = (line_center[0] + line_width // 2, line_center[1])
    right_line_end = (line_center[0] + line_width // 2, line_center[1] + line_length)

    cv2.line(frame, left_line_start, left_line_end, (255, 255, 255), 5)
    cv2.line(frame, right_line_start, right_line_end, (255, 255, 255), 5)

    majority_group = find_majority_group(centers)

    if len(majority_group) > 0:
        # Calculate the bounding rectangle around the majority group
        x, y, w, h = cv2.boundingRect(np.array(majority_group))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      
        rect_center = (x + w // 2, y + h // 2)

        cv2.line(frame, rect_center, line_center, (255, 255, 255), 2)

        # Calculate distance from rectangle center to the parallel lines center in steps

        distance_to_line_pixels = np.linalg.norm(np.array(rect_center) - np.array(line_center))
        distance_in_meters = distance_to_line_pixels / 100  # Assuming 1 pixel = 0.01 meters
        distance_in_steps = distance_in_meters * 2  # 1 meter is approximately 2 steps

        cv2.putText(frame, f"{distance_in_steps:.1f} steps", line_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for center in centers:
        if is_prominent_obstacle(center, viewer_position, threshold_distance):

            cv2.circle(frame, center, 5, (255, 255, 255), -1)

    return frame

def main():
    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        frame_with_obstacles = draw_obstacles(frame, centers)

        cv2.imshow('Obstacles Visualization', frame_with_obstacles)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
