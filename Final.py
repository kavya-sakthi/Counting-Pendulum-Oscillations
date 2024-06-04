import cv2
import numpy as np

# Parameters
threshold_value = 40  # Binary threshold for image segmentation
min_contour_length = 20  # Minimum contour length for pendulum detection
frame_skip = 0  # Number of frames to skip for analysis

# Load video
video_path = r"Video1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video was successfully opened
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Initialize variables
frame_count = 0
oscillation_count = 0
previous_angle = None

# Loop through frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Skip frames
    if frame_count % (frame_skip + 1) != 0:
        continue

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours
    for contour in contours:
        # Filter contours based on length
        if cv2.arcLength(contour, True) < min_contour_length:
            continue

        # Check if contour has enough points for fitting an ellipse
        if len(contour) < 5:
            continue

        # Fit ellipse to contour
        ellipse = cv2.fitEllipse(contour)

        # Calculate angle of major axis
        angle = ellipse[2]

        # Initialize previous angle
        if previous_angle is None:
            previous_angle = angle

        # Detect oscillation by checking sign change
        if np.sign(angle - previous_angle) != np.sign(previous_angle):
            oscillation_count += 1

        # Update previous angle
        previous_angle = angle

    # Display frame with detected pendulum
    cv2.imshow('Pendulum Detection', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print result
print('Number of oscillations:', oscillation_count)
