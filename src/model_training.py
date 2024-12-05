import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define paths
PROCESSED_DATA_PATH = './data/processed'

def load_data():
    """
    Loads preprocessed training data and splits it into train and validation sets.
    """
    train_data = np.load(os.path.join(PROCESSED_DATA_PATH, 'train_data.npy'))
    train_labels = np.load(os.path.join(PROCESSED_DATA_PATH, 'train_labels.npy'))

    # Reshape data to add a channel dimension
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)

    # Convert labels to categorical (one-hot encoding)
    train_labels = to_categorical(train_labels, num_classes=27)  # 26 letters + 1 padding class

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    return x_train, x_val, y_train, y_val

def build_model():
    """
    Builds and compiles a CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(27, activation='softmax')  # 27 classes for output
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_training_history(history):
    """
    Plots the accuracy and loss from the training history.
    """
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def train_model():
    """
    Loads data, builds the model, trains it, saves the model, and visualizes training results.
    """
    x_train, x_val, y_train, y_val = load_data()

    model = build_model()

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64
    )

    # Save the trained model
    os.makedirs('./saved_models', exist_ok=True)
    model.save('./saved_models/emnist_cnn_model.h5')
    print("Model trained and saved at ./saved_models/emnist_cnn_model.h5")

    # Visualize training results
    plot_training_history(history)

if __name__ == '__main__':
    train_model()
