import os
import cv2
import numpy as np

# Paths
RAW_DATA_PATH = './data/raw/words'
PROCESSED_DATA_PATH = './data/processed_words'

# Create processed data folder if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Parameters
IMAGE_HEIGHT = 32  # Fixed height for word images
IMAGE_WIDTH = 128  # Max width (padded if necessary)

def get_word_images_and_labels():
    """
    Parses the word images directory and extracts labels from filenames.
    Returns:
        images (list): List of preprocessed word images.
        labels (list): Corresponding word labels.
    """
    images = []
    labels = []

    # Walk through the directory structure
    for root, _, files in os.walk(RAW_DATA_PATH):
        for file in files:
            if file.endswith('.png'):
                # Extract the label from the filename or a corresponding metadata file
                # File path example: a01/a01-000u/a01-000u-00.png
                label_path = os.path.join(root, file.replace('.png', '.txt'))
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label = f.readline().strip()
                    image_path = os.path.join(root, file)
                    image = preprocess_image(image_path)
                    images.append(image)
                    labels.append(label)

    return images, labels

def preprocess_image(image_path):
    """
    Reads and preprocesses an image for training.
    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to fixed height, preserving aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(aspect_ratio * IMAGE_HEIGHT)
    image = cv2.resize(image, (new_width, IMAGE_HEIGHT))

    # Pad or truncate to fixed width
    if new_width < IMAGE_WIDTH:
        padded_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32) * 255  # White background
        padded_image[:, :new_width] = image
        image = padded_image
    else:
        image = image[:, :IMAGE_WIDTH]

    # Normalize pixel values to range [0, 1]
    image = image / 255.0

    return image

def process_dataset():
    """
    Processes the IAM dataset and saves it as NumPy arrays.
    """
    images, labels = get_word_images_and_labels()

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Save the processed data
    np.save(os.path.join(PROCESSED_DATA_PATH, 'word_images.npy'), images)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'word_labels.npy'), labels)

    print(f"Processed {len(images)} word images and labels.")
    print(f"Data saved to {PROCESSED_DATA_PATH}.")

if __name__ == '__main__':
    process_dataset()
