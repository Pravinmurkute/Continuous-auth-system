import cv2
import numpy as np

# Create a blank image with a white background
placeholder_image = np.ones((500, 500, 3), dtype=np.uint8) * 255

# Draw a simple face-like structure
cv2.circle(placeholder_image, (250, 250), 100, (0, 0, 0), 3)  # Face outline
cv2.circle(placeholder_image, (220, 220), 10, (0, 0, 0), -1)  # Left eye
cv2.circle(placeholder_image, (280, 220), 10, (0, 0, 0), -1)  # Right eye
cv2.ellipse(placeholder_image, (250, 280), (50, 20), 0, 0, 180, (0, 0, 0), 3)  # Mouth

# Save the placeholder image
cv2.imwrite("D:/Continuous_auth_system/stored_face.jpg", placeholder_image)
