import cv2
import numpy as np
import imgaug.augmenters as iaa
import os

# Augmentation function
def augment_image(image_path, output_dir):
    img = cv2.imread(image_path)

    # Define augmentations
    augmenters = iaa.Sequential([
        iaa.Fliplr(0.5),               # Flip horizontally
        iaa.Affine(rotate=(-20, 20)),  # Rotate image
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add noise
        iaa.Multiply((0.8, 1.2))       # Adjust brightness
    ])

    # Apply augmentations
    aug_images = [augmenters(image=img) for _ in range(5)]
    
    # Save augmented images
    base_name = os.path.basename(image_path).split('.')[0]
    for i, aug_img in enumerate(aug_images):
        cv2.imwrite(f"{output_dir}/{base_name}_aug_{i}.jpg", aug_img)

# Augment all images in dataset
dataset_dir = "lfw"
output_dir = "augmented_lfw"
os.makedirs(output_dir, exist_ok=True)

for subdir, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(subdir, file)
            augment_image(img_path, output_dir)
