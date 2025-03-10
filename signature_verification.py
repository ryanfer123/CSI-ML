import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load Images Function
def load_images(folder, label):
    images, labels = [], []
    for file in os.listdir(folder):
        if file.endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Failed to load {img_path}")
                continue
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

# NST Function to Apply Style Transfer
def apply_nst(content_img, style_img):
    hub_module = tf.keras.applications.vgg19.preprocess_input
    stylized_img = hub_module(tf.constant(content_img), tf.constant(style_img))
    return np.array(stylized_img[0])

# Define Paths
genuine_path = "/Users/ryanfernandes/PythonProject1/dataset/signatures/full_org"
forged_path = "/Users/ryanfernandes/PythonProject1/dataset/signatures/full_forg"
style_path = "/Users/ryanfernandes/PythonProject1/dataset/archive (1)/artbench-10-python"

# Load Original Signature Images
genuine_images, genuine_labels = load_images(genuine_path, label=1)
forged_images, forged_labels = load_images(forged_path, label=0)

# Load Style Images
style_images, _ = load_images(style_path, label=None)

# Apply NST to Generate Synthetic Forgeries
synthetic_forgeries = []
for img in genuine_images[:len(style_images)]:  # Match number of style images
    style_img = np.random.choice(style_images)
    synthetic_forgeries.append(apply_nst(img, style_img))

# Prepare Data for Training
X = genuine_images + forged_images + synthetic_forgeries
y = genuine_labels + forged_labels + [0] * len(synthetic_forgeries)
X = np.array(X)
y = np.array(y)

# Feature Extraction using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
X_features = np.array([feature_extractor.predict(np.expand_dims(img, axis=0)).flatten() for img in X])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train SVM Classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

import numpy as np
import matplotlib.pyplot as plt
from random import sample

# Function to plot images
def plot_signatures(genuine, forged, synthetic, num_samples=5):
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    fig.suptitle("Signature Samples: Genuine | Forged | NST-Forged", fontsize=14)

    # Randomly select samples
    genuine_samples = sample(genuine, num_samples)
    forged_samples = sample(forged, num_samples)
    synthetic_samples = synthetic if len(synthetic) < num_samples else sample(synthetic, num_samples)

    for i in range(num_samples):
        axes[0, i].imshow(genuine_samples[i], cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(forged_samples[i], cmap='gray')
        axes[1, i].axis('off')

        axes[2, i].imshow(synthetic_samples[i], cmap='gray')
        axes[2, i].axis('off')

    plt.show()

# Call the function to visualize samples
print(f"📢 Image Counts - Genuine: {len(genuine_images)}, Forged: {len(forged_images)}, Synthetic: {len(synthetic_forgeries)}")
plot_signatures(genuine_images, forged_images, synthetic_forgeries)