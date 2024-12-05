import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define paths
PROCESSED_DATA_PATH = './data/processed'
MODEL_PATH = './saved_models/emnist_cnn_model.h5'

def load_test_data():
    """
    Loads preprocessed test data.
    """
    test_data = np.load(os.path.join(PROCESSED_DATA_PATH, 'test_data.npy'))
    test_labels = np.load(os.path.join(PROCESSED_DATA_PATH, 'test_labels.npy'))

    # Reshape data to add a channel dimension
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    return test_data, test_labels

def load_model():
    """
    Loads the trained model from file.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    return model

def evaluate_model(model, test_data, test_labels):
    """
    Evaluates the model on test data and generates a classification report.
    """
    # Make predictions
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(test_labels, predicted_classes, target_names=[chr(i) for i in range(65, 91)]))

    # Generate confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[chr(i) for i in range(65, 91)], yticklabels=[chr(i) for i in range(65, 91)])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def visualize_predictions(model, test_data, test_labels, num_samples=10):
    """
    Visualizes predictions on random test samples.
    """
    indices = np.random.choice(range(len(test_data)), num_samples, replace=False)
    for i in indices:
        plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
        true_label = chr(65 + test_labels[i])  # Convert numeric label to ASCII letter
        predicted_label = chr(65 + np.argmax(model.predict(test_data[i:i+1])))
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    test_data, test_labels = load_test_data()
    model = load_model()
    evaluate_model(model, test_data, test_labels)
    visualize_predictions(model, test_data, test_labels)
