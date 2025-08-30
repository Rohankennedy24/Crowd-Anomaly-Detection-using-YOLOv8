# Import the necessary libraries
import cv2
from ultralytics import YOLO

# --- Configuration Section ---
# The YOLOv8n model will be downloaded automatically by the ultralytics library
model = YOLO('yolov8n.pt') 

# Define the path to your video file
# You will need to replace this with the path to your own video.
video_file_path = 'street_walk.mp4'

# Define the minimum confidence threshold to filter weak detections
confidence_threshold = 0.5

# Define a list of "anomalous" objects to detect
# The names are based on the COCO dataset classes that YOLOv8 was trained on.
anomalous_objects = ['fire hydrant', 'bus', 'truck', 'airplane', 'boat', 'car']

# --- Main Script Execution ---
def run_combined_detector():
    """
    Main function to run both anomaly detection and crowd counting on a single video stream.
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_file_path}'.")
            return

        # Loop through each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference on the frame
            results = model(frame, stream=True)
            
            anomaly_detected = False
            person_count = 0
            
            # Process results for each detection
            for r in results:
                for box in r.boxes:
                    # Get the class name of the detected object
                    detected_class_name = model.names[int(box.cls[0])]
                    confidence = box.conf[0]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crowd Counting: Check for a person
                    if detected_class_name == 'person' and confidence > confidence_threshold:
                        person_count += 1
                        # Draw a GREEN bounding box for people
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Anomaly Detection: Check if the detected object is in our list of anomalous objects
                    if detected_class_name in anomalous_objects and confidence > confidence_threshold:
                        anomaly_detected = True
                        
                        # Draw a RED bounding box and label for an anomaly
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"ANOMALY: {detected_class_name} ({confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display the total person count on the screen
            count_label = f"People Count: {person_count}"
            cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display a large warning message if an anomaly is detected
            if anomaly_detected:
                cv2.putText(frame, "ANOMALY DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Display the resulting frame
            cv2.imshow("Combined Detector", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: A required file was not found. Please ensure your video file exists.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    run_combined_detector()
