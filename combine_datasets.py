import os
from deepface import DeepFace

# Combine original and augmented datasets
dataset_dirs = ["your_face_dataset", "augmented_lfw"]
combined_images = []

for directory in dataset_dirs:
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                combined_images.append(os.path.join(subdir, file))

# Use combined dataset for verification
DeepFace.find(img_path="test_face.jpg", db_path=combined_images)
