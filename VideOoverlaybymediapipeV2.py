import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to overlay the image with rotation following arm direction
def overlay_image_alpha_rotated(img, img_overlay, pos, alpha_mask, angle):
    x, y = pos

    # Rotate the overlay image
    img_overlay_rotated = rotate_image(img_overlay, angle)
    alpha_mask_rotated = rotate_image(alpha_mask, angle)

    # Calculate the new position after rotation
    h, w = img_overlay_rotated.shape[:2]
    new_pos = (x - w // 2, y - h // 2)

    # Image ranges
    y1, y2 = max(0, new_pos[1]), min(img.shape[0], new_pos[1] + h)
    x1, x2 = max(0, new_pos[0]), min(img.shape[1], new_pos[0] + w)

    # Overlay ranges
    y1o, y2o = max(0, -new_pos[1]), min(h, img.shape[0] - new_pos[1])
    x1o, x2o = max(0, -new_pos[0]), min(w, img.shape[1] - new_pos[0])

    # Exit if nothing to overlay
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay_rotated[y1o:y2o, x1o:x2o]
    alpha = alpha_mask_rotated[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

# Function to rotate an image around its center
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rot_matrix, (w, h), flags=cv.INTER_LINEAR)
    return rotated_image

# Function to detect pose and overlay image following arm direction
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

        # Calculate the length of the hand region (between wrist and elbow)
        right_hand_length = int(np.sqrt((right_wrist_coord[0] - right_elbow_coord[0])**2 + (right_wrist_coord[1] - right_elbow_coord[1])**2) * 0.5)
        left_hand_length = int(np.sqrt((left_wrist_coord[0] - left_elbow_coord[0])**2 + (left_wrist_coord[1] - left_elbow_coord[1])**2) * 0.5)

        # Calculate the angle of the hand for rotation
        right_angle = np.degrees(np.arctan2(right_wrist_coord[1] - right_elbow_coord[1], right_wrist_coord[0] - right_elbow_coord[0]))
        left_angle = np.degrees(np.arctan2(left_wrist_coord[1] - left_elbow_coord[1], left_wrist_coord[0] - left_elbow_coord[0]))

        # Resize overlay image to fit the hand length
        right_overlay_resized = cv.resize(overlay_img, (right_hand_length, right_hand_length))
        left_overlay_resized = cv.resize(overlay_img, (left_hand_length, left_hand_length))

        # Adjust position to overlay the image at the wrist
        right_position = right_wrist_coord
        left_position = left_wrist_coord

        # Overlay the image on the right hand with rotation following the arm direction
        alpha_mask = right_overlay_resized[:, :, 3] / 255.0
        overlay_image_alpha_rotated(frame, right_overlay_resized[:, :, :3], right_position, alpha_mask, right_angle)

        # Repeat for the left hand with rotation following the arm direction
        alpha_mask = left_overlay_resized[:, :, 3] / 255.0
        overlay_image_alpha_rotated(frame, left_overlay_resized[:, :, :3], left_position, alpha_mask, left_angle)

    return frame

# Load overlay image
overlay_img_path = "tattoo.png"
overlay_img = cv.imread(overlay_img_path, cv.IMREAD_UNCHANGED)
if overlay_img is None:
    raise FileNotFoundError(f"Overlay image not found at path: {overlay_img_path}")

# Load video
video_path = "Video1.mp4"
cap = cv.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_video_path = "output_with_overlay_video.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame
    output_frame = poseDetector(frame, overlay_img)

    # Write the processed frame to the output video
    out.write(output_frame)

# Release everything
cap.release()
out.release()

print(f"Processed video saved as {output_video_path}")
