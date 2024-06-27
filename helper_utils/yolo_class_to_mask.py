import os
import numpy as np
import cv2

def generate_mask(image_shape, keypoints):
    """Generate a mask from keypoints"""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert normalized coordinates to absolute pixel coordinates
    absolute_points = [(int(x * width), int(y * height)) for x, y in keypoints]

    # Draw the polygon on the mask
    absolute_points = np.array(absolute_points, dtype=np.int32)
    cv2.fillPoly(mask, [absolute_points], 1)

    return mask

def save_mask(mask, output_path):
    cv2.imwrite(output_path, mask * 255)  # Convert mask to 0-255 range before saving

def process_directory(input_dir, output_dir, image_shape):
    """Process all label files in the input directory and save the resulting masks to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Assuming the keypoints are in .txt files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.txt', '.png'))

            # Read the keypoints from the file
            keypoints = []
            with open(input_path, 'r') as file:
                line = file.readline().strip()
                values = list(map(float, line.split(' ')))
                
                # dump 1st
                values = values[1:]

                if len(values) % 2 != 0:
                    print(f"Invalid number of values in {filename}: {line}")
                    continue

                keypoints = [(values[i], values[i+1]) for i in range(0, len(values), 2)]

            # Generate and save the mask if valid keypoints are found
            if keypoints:
                mask = generate_mask(image_shape, keypoints)
                save_mask(mask, output_path)
            else:
                print(f"No valid keypoints found in {filename}")

# Example usage
input_dir = 'H:\Cataract Dataset\FSR Dataset\Yolo_Segmentation_640x640\Labels_txt'
output_dir = 'H:\Cataract Dataset\FSR Dataset\Yolo_Segmentation_640x640\Labels'
image_shape = (640, 640)  # Example image shape (height, width)

process_directory(input_dir, output_dir, image_shape)