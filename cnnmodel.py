import os
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ----------------------------
# Parameters & Configuration
# ----------------------------
IMG_WIDTH = 129
IMG_HEIGHT = 40
NUM_CHANNELS = 1  # using grayscale images
NUM_CLASSES = 10  # digits 0-9
NUM_DIGITS = 6    # each captcha has 6 digits

REAL_DATA_DIR = 'test'
SYNTHETIC_DATA_DIR = 'train'

# ----------------------------
# Synthetic Dataset Loading
# ----------------------------
def load_synthetic_dataset(data_dir):
    X = []
    y = [[] for _ in range(NUM_DIGITS)]
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.png'):
            label_str = os.path.splitext(filename)[0]
            if len(label_str) != NUM_DIGITS:
                continue  # Skip images with an unexpected label length
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)
            X.append(image_array)
            # Create one-hot encoded labels for each digit in the captcha
            for j, char in enumerate(label_str):
                digit = int(char)
                one_hot = to_categorical(digit, num_classes=NUM_CLASSES)
                y[j].append(one_hot)
    X = np.array(X)
    y = [np.array(arr) for arr in y]
    return X, y

print("Loading synthetic dataset...")
X_synth, y_synth = load_synthetic_dataset(SYNTHETIC_DATA_DIR)
print("Synthetic data shape:", X_synth.shape)

# Shuffle the synthetic dataset and labels to ensure even distribution
indices = np.arange(len(X_synth))
np.random.shuffle(indices)
X_synth = X_synth[indices]
y_synth = [arr[indices] for arr in y_synth]

# ----------------------------
# Model Definition
# ----------------------------
def build_captcha_model():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)

    # Create one output per digit
    outputs = [layers.Dense(NUM_CLASSES, activation='softmax', name=f'digit_{i}')(x)
               for i in range(NUM_DIGITS)]

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = build_captcha_model()
model.compile(
    optimizer='adam',
    loss={f'digit_{i}': 'categorical_crossentropy' for i in range(NUM_DIGITS)},
    metrics={f'digit_{i}': 'accuracy' for i in range(NUM_DIGITS)}
)
model.summary()

# ----------------------------
# Pre-training on Synthetic Data
# ----------------------------
print("Pre-training on synthetic data...")
history_synth = model.fit(
    X_synth,
    {f'digit_{i}': y_synth[i] for i in range(NUM_DIGITS)},
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ----------------------------
# Loading the Real Dataset
# ----------------------------
def load_real_dataset(data_dir):
    X = []
    y = [[] for _ in range(NUM_DIGITS)]
    for filename in os.listdir(data_dir):
        if filename.lower().endswith('.png'):
            label_str = os.path.splitext(filename)[0]
            if len(label_str) != NUM_DIGITS:
                continue  # skip if the label length does not match expected digits
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path).convert('L')
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)
            X.append(image_array)
            for j, char in enumerate(label_str):
                digit = int(char)
                one_hot = to_categorical(digit, num_classes=NUM_CLASSES)
                y[j].append(one_hot)
    X = np.array(X)
    y = [np.array(arr) for arr in y]
    return X, y

print("Loading real dataset...")
X_real, y_real = load_real_dataset(REAL_DATA_DIR)
print("Real data shape:", X_real.shape)

# ----------------------------
# Combined Synthetic & Real Data Generator with Augmentation
# ----------------------------
# Create a data augmentation generator for real images
datagen_real = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2]
)

# Custom generator that samples from synthetic or real data based on a synthetic_ratio
class CombinedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_synth, y_synth, X_real, y_real, batch_size, synthetic_ratio, datagen_real=None):
        self.X_synth = X_synth
        self.y_synth = y_synth  # List of arrays, one per digit
        self.X_real = X_real
        self.y_real = y_real    # List of arrays, one per digit
        self.batch_size = batch_size
        self.synthetic_ratio = synthetic_ratio  # initial probability for synthetic sample
        self.datagen_real = datagen_real
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices_synth = np.arange(len(self.X_synth))
        np.random.shuffle(self.indices_synth)
        self.indices_real = np.arange(len(self.X_real))
        np.random.shuffle(self.indices_real)
        self.pos_synth = 0
        self.pos_real = 0

    def __len__(self):
        # Number of steps per epoch
        return 100

    def __getitem__(self, index):
        X_batch = []
        y_batch = [[] for _ in range(NUM_DIGITS)]
        for _ in range(self.batch_size):
            if np.random.rand() < self.synthetic_ratio:
                # Sample a synthetic example
                if self.pos_synth >= len(self.X_synth):
                    self.pos_synth = 0
                    np.random.shuffle(self.indices_synth)
                idx = self.indices_synth[self.pos_synth]
                self.pos_synth += 1
                X_sample = self.X_synth[idx]
                sample_labels = [self.y_synth[d][idx] for d in range(NUM_DIGITS)]
            else:
                # Sample a real example
                if self.pos_real >= len(self.X_real):
                    self.pos_real = 0
                    np.random.shuffle(self.indices_real)
                idx = self.indices_real[self.pos_real]
                self.pos_real += 1
                X_sample = self.X_real[idx]
                if self.datagen_real is not None:
                    X_sample = self.datagen_real.random_transform(X_sample)
                sample_labels = [self.y_real[d][idx] for d in range(NUM_DIGITS)]
            X_batch.append(X_sample)
            for d in range(NUM_DIGITS):
                y_batch[d].append(sample_labels[d])
        X_batch = np.array(X_batch)
        y_batch = [np.array(y_list) for y_list in y_batch]
        return X_batch, {f'digit_{d}': y_batch[d] for d in range(NUM_DIGITS)}

# Callback to gradually reduce the synthetic ratio over epochs
class SyntheticRatioScheduler(tf.keras.callbacks.Callback):
    def __init__(self, generator, initial_ratio, total_epochs):
        super().__init__()
        self.generator = generator
        self.initial_ratio = initial_ratio
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Linearly reduce synthetic ratio from initial_ratio to 0 over total_epochs
        new_ratio = self.initial_ratio * (1 - epoch / self.total_epochs)
        self.generator.synthetic_ratio = new_ratio
        print(f"Epoch {epoch+1}: Setting synthetic ratio to {new_ratio:.2f}")

# ----------------------------
# Combined Training on Synthetic & Real Data
# ----------------------------
BATCH_SIZE = 16
initial_synthetic_ratio = 0.8
total_epochs = 20

combined_generator = CombinedDataGenerator(
    X_synth, y_synth, X_real, y_real,
    batch_size=BATCH_SIZE,
    synthetic_ratio=initial_synthetic_ratio,
    datagen_real=datagen_real
)

scheduler_callback = SyntheticRatioScheduler(combined_generator, initial_synthetic_ratio, total_epochs)

# Use real data (without augmentation) for validation
val_data = (X_real, {f'digit_{d}': y_real[d] for d in range(NUM_DIGITS)})

print("Training on combined synthetic and real data...")
history_combined = model.fit(
    combined_generator,
    epochs=total_epochs,
    validation_data=val_data,
    callbacks=[scheduler_callback]
)

# ----------------------------
# Prediction and Evaluation on Real Data
# ----------------------------
preds = model.predict(X_real)
predicted_labels = []
for sample_idx in range(len(X_real)):
    digits = [str(np.argmax(preds[i][sample_idx])) for i in range(NUM_DIGITS)]
    predicted_labels.append("".join(digits))

# Compute ground truth labels as strings from one-hot arrays in y_real
ground_truth_labels = []
for sample_idx in range(len(X_real)):
    digits = [str(np.argmax(y_real[i][sample_idx])) for i in range(NUM_DIGITS)]
    ground_truth_labels.append("".join(digits))

# Calculate string-level accuracy
correct_count = sum(1 for pred, true in zip(predicted_labels, ground_truth_labels) if pred == true)
string_accuracy = correct_count / len(ground_truth_labels)
print("String-level accuracy on real data:", string_accuracy)

# Calculate digit-level accuracy (per output)
digit_accuracies = []
for i in range(NUM_DIGITS):
    digit_preds = np.argmax(preds[i], axis=1)
    digit_true = np.argmax(y_real[i], axis=1)
    acc = np.mean(digit_preds == digit_true)
    digit_accuracies.append(acc)
    print(f"Digit {i} accuracy: {acc}")

print("Sample predictions on real data:", predicted_labels[:10])
