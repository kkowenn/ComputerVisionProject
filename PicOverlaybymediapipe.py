import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to overlay the image
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to overlay
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

# Function to detect pose and overlay image
def poseDetector(frame, overlay_img):
    # Convert the image to RGB as MediaPipe expects RGB images
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract relevant landmarks
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Convert landmark positions to pixel coordinates
        h, w, _ = frame.shape
        right_wrist_coord = (int(right_wrist.x * w), int(right_wrist.y * h))
        left_wrist_coord = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_elbow_coord = (int(right_elbow.x * w), int(right_elbow.y * h))
        left_elbow_coord = (int(left_elbow.x * w), int(left_elbow.y * h))

        # Calculate the length of the hand region (between wrist and middle of elbow)
        right_hand_length = int(np.sqrt((right_wrist_coord[0] - right_elbow_coord[0])**2 + (right_wrist_coord[1] - right_elbow_coord[1])**2) * 0.5)
        left_hand_length = int(np.sqrt((left_wrist_coord[0] - left_elbow_coord[0])**2 + (left_wrist_coord[1] - left_elbow_coord[1])**2) * 0.5)

        # Resize overlay image to fit the hand length
        right_overlay_resized = cv.resize(overlay_img, (right_hand_length, right_hand_length))
        left_overlay_resized = cv.resize(overlay_img, (left_hand_length, left_hand_length))

        # Adjust position to overlay the image at the wrist
        right_position = (int(right_wrist_coord[0] - right_hand_length / 2), int(right_wrist_coord[1] - right_hand_length / 2))
        left_position = (int(left_wrist_coord[0] - left_hand_length / 2), int(left_wrist_coord[1] - left_hand_length / 2))

        # Overlay the image on the right hand
        alpha_mask = right_overlay_resized[:, :, 3] / 255.0
        overlay_image_alpha(frame, right_overlay_resized[:, :, :3], right_position, alpha_mask)

        # Repeat for the left hand if needed
        alpha_mask = left_overlay_resized[:, :, 3] / 255.0
        overlay_image_alpha(frame, left_overlay_resized[:, :, :3], left_position, alpha_mask)

    return frame

# Load overlay image
overlay_img_path = "tattoo.png"
overlay_img = cv.imread(overlay_img_path, cv.IMREAD_UNCHANGED)
if overlay_img is None:
    raise FileNotFoundError(f"Overlay image not found at path: {overlay_img_path}")

# Load input image
input_img_path = "body1.JPG"
input = cv.imread(input_img_path)
if input is None:
    raise FileNotFoundError(f"Input image not found at path: {input_img_path}")

# Process and display the image
output = poseDetector(input, overlay_img)

# Save the output image
output_img_path = "output_with_overlay1.jpg"
cv.imwrite(output_img_path, output)
print(f"Output saved to {output_img_path}")

# Display the output image
cv.imshow('Pose Detection with Overlay', output)
cv.waitKey(0)  # Wait for a key press to close the image window
cv.destroyAllWindows()
