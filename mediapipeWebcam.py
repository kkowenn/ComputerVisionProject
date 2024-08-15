import cv2 as cv
import mediapipeWebcam as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Access webcam feed
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the BGR image to RGB
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(image_rgb)

    # Draw the pose annotation on the image
    image_height, image_width, _ = frame.shape
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            z = landmark.z  # z is in normalized coordinates
            cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw connections between landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image with pose annotations
    cv.imshow('3D Pose Estimation', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv.destroyAllWindows()
