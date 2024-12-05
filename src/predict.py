import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Define paths
MODEL_PATH = './saved_models/emnist_cnn_model.h5'
NEW_IMAGES_PATH = './data/new_images'

def load_model():
    """
    Loads the trained model from file.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

def preprocess_image(image_path):
    """
    Preprocesses a single image for prediction.
    Args:
        image_path (str): Path to the image file.
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)

    # Normalize the image
    image_array = image_array / 255.0

    # Add channel dimension
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array

def predict_image(model, image_path):
    """
    Makes a prediction on a single image.
    Args:
        model: Trained model.
        image_path (str): Path to the image file.
    Returns:
        str: Predicted label.
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return chr(65 + predicted_class)  # Convert numeric label to ASCII letter (A-Z)

def predict_new_images():
    """
    Predicts labels for all images in the new_images folder.
    """
    if not os.path.exists(NEW_IMAGES_PATH):
        raise FileNotFoundError(f"New images folder not found at {NEW_IMAGES_PATH}.")
    
    model = load_model()
    images = [f for f in os.listdir(NEW_IMAGES_PATH) if f.endswith(('png', 'jpg', 'jpeg'))]

    if not images:
        print("No images found in the new_images folder.")
        return

    for image_file in images:
        image_path = os.path.join(NEW_IMAGES_PATH, image_file)
        predicted_label = predict_image(model, image_path)
        print(f"Image: {image_file}, Predicted Label: {predicted_label}")

        # Optionally, display the image with the prediction
        image = Image.open(image_path)
        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    predict_new_images()
