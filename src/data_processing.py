import os
import numpy as np
from torchvision import datasets, transforms

# Define paths
RAW_DATA_PATH = './data/raw'
PROCESSED_DATA_PATH = './data/processed'

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def download_and_process_emnist():
    """
    Downloads and preprocesses the EMNIST Letters dataset.
    Saves the processed data as .npy files.
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Download and load EMNIST Letters
    train_dataset = datasets.EMNIST(root=RAW_DATA_PATH, split='letters', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root=RAW_DATA_PATH, split='letters', train=False, download=True, transform=transform)

    # Convert training data to NumPy arrays
    train_images = []
    train_labels = []
    for image, label in train_dataset:
        train_images.append(image.numpy())
        train_labels.append(label)

    train_images = np.array(train_images).squeeze()  # Remove channel dimension
    train_labels = np.array(train_labels)

    # Normalize the images
    train_images = train_images / 255.0

    # Save training data
    np.save(os.path.join(PROCESSED_DATA_PATH, 'train_data.npy'), train_images)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'train_labels.npy'), train_labels)

    # Convert test data to NumPy arrays
    test_images = []
    test_labels = []
    for image, label in test_dataset:
        test_images.append(image.numpy())
        test_labels.append(label)

    test_images = np.array(test_images).squeeze()
    test_labels = np.array(test_labels)

    # Normalize the images
    test_images = test_images / 255.0

    # Save test data
    np.save(os.path.join(PROCESSED_DATA_PATH, 'test_data.npy'), test_images)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'test_labels.npy'), test_labels)

    print("Data processing complete. Files saved in ./data/processed")

if __name__ == '__main__':
    download_and_process_emnist()
