# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from pathlib import Path
import glob
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 50
CODE = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
NAME_TO_CODE = {k.lower(): v for v, k in CODE.items()}

# Paths (update these to your local paths)
TRAIN_DIR = Path('dataset/train')
TEST_DIR = Path('dataset/test')
MODEL_PATH = 'facial_emotion_model.h5'

# Load data
def load_data():
    x_train, y_train, x_test, y_test = [], [], [], []
    
    print(f"Loading training images from {TRAIN_DIR}")
    for folder in os.listdir(TRAIN_DIR):
        folder_key = folder.lower()
        if folder_key not in NAME_TO_CODE:
            print(f"Warning: Folder '{folder}' not recognized, skipping")
            continue
        files = glob.glob(str(TRAIN_DIR / folder / '*.jpg'))
        print(f"Found {len(files)} images in {folder}")
        for image_path in files:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {image_path}")
                continue
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x_train.append(resized_img)
            y_train.append(NAME_TO_CODE[folder_key])

    print(f"Loading test images from {TEST_DIR}")
    for folder in os.listdir(TEST_DIR):
        folder_key = folder.lower()
        if folder_key not in NAME_TO_CODE:
            continue
        files = glob.glob(str(TEST_DIR / folder / '*.jpg'))
        for image_path in files:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x_test.append(resized_img)
            y_test.append(NAME_TO_CODE[folder_key])

    x_train = np.array(x_train).astype('float32') / 255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test).astype('float32') / 255.0
    y_test = np.array(y_test)
    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test, y_train_one_hot, y_test_one_hot

# Compute class weights
def compute_class_weights(y_train):
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (NUM_CLASSES * count) for i, count in class_counts.items()}
    return class_weights

# Define CNN model
def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    # Load data
    x_train, y_train, x_test, y_test, y_train_one_hot, y_test_one_hot = load_data()
    print(f"Total training images: {len(x_train)}")
    print(f"Total test images: {len(x_test)}")

    # Class weights
    class_weights = compute_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # Create and compile model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    # Train model
    history = model.fit(
        datagen.flow(x_train, y_train_one_hot, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test_one_hot),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test_one_hot)
    print(f"Test accuracy: {accuracy:.4f}")

    # Additional metrics
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"{CODE[i]}: {acc:.4f}")
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CODE.values(), yticklabels=CODE.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history.history['loss'], label='Train Loss', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Training Accuracy and Loss Over Epochs')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.savefig('training_history.png')
    plt.close()

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()