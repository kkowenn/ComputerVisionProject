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

# Function to color the arm region
def color_arm_region(frame, start_point, end_point, target_color, new_color):
    # Create a mask based on color matching
    mask = cv.inRange(frame, np.array(target_color) - 20, np.array(target_color) + 20)

    # Define the line connecting the wrist and elbow
    arm_line = np.zeros_like(frame)
    cv.line(arm_line, start_point, end_point, (255, 255, 255), thickness=20)

    # Combine the mask with the arm line
    arm_mask = cv.bitwise_and(mask, arm_line[:, :, 0])

    # Apply the new color to the detected arm region
    frame[arm_mask > 0] = new_color

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

        # Color the right arm region
        target_color = [200, 160, 140]  # Example skin tone color to match
        new_color = [0, 255, 0]  # Example color to apply (green)
        color_arm_region(frame, right_wrist_coord, right_elbow_coord, target_color, new_color)

        # Color the left arm region if needed
        color_arm_region(frame, left_wrist_coord, left_elbow_coord, target_color, new_color)

        # Continue with the overlay image if needed
        right_hand_length = int(np.sqrt((right_wrist_coord[0] - right_elbow_coord[0])**2 + (right_wrist_coord[1] - right_elbow_coord[1])**2) * 0.5)
        left_hand_length = int(np.sqrt((left_wrist_coord[0] - left_elbow_coord[0])**2 + (left_wrist_coord[1] - left_elbow_coord[1])**2) * 0.5)

        right_overlay_resized = cv.resize(overlay_img, (right_hand_length, right_hand_length))
        left_overlay_resized = cv.resize(overlay_img, (left_hand_length, left_hand_length))

        right_position = (int(right_wrist_coord[0] - right_hand_length / 2), int(right_wrist_coord[1] - right_hand_length / 2))
        left_position = (int(left_wrist_coord[0] - left_hand_length / 2), int(left_wrist_coord[1] - left_hand_length / 2))

        alpha_mask = right_overlay_resized[:, :, 3] / 255.0
        overlay_image_alpha(frame, right_overlay_resized[:, :, :3], right_position, alpha_mask)

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
output_img_path = "output_with_colored_arm.jpg"
cv.imwrite(output_img_path, output)
print(f"Output saved to {output_img_path}")

# Display the output image
cv.imshow('Pose Detection with Colored Arm', output)
cv.waitKey(0)  # Wait for a key press to close the image window
cv.destroyAllWindows()
