import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2

MODEL_PATH = './saved_models/emnist_cnn_model.h5'
NEW_IMAGES_PATH = './data/new_images_words'

def load_model():
    """
    Loads the trained model from file.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

def segment_characters(image_path):
    """
    Segments a handwritten word image into individual characters.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Sort bounding boxes left-to-right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    segmented_chars = []
    for x, y, w, h in bounding_boxes:
        char_image = image[y:y+h, x:x+w]
        char_image = cv2.resize(char_image, (28, 28))
        segmented_chars.append(char_image)

    return segmented_chars

def predict_word(model, image_path):
    """
    Predicts the word from a handwritten word image.
    """
    segmented_chars = segment_characters(image_path)
    predicted_word = ""

    for char_image in segmented_chars:
        char_image = np.array(char_image) / 255.0
        char_image = char_image.reshape(1, 28, 28, 1)

        prediction = model.predict(char_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_word += chr(65 + predicted_class)

    return predicted_word

def predict_new_images():
    """
    Predicts words for all images in the new_images folder.
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
        predicted_word = predict_word(model, image_path)
        print(f"Image: {image_file}, Predicted Word: {predicted_word}")

        image = Image.open(image_path)
        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {predicted_word}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    predict_new_images()
