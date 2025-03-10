# Signature Forgery Detection using CNN & SVM

## 📌 Project Overview
This project implements a **signature forgery detection system** using **Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs)**. The pipeline involves:

- **Loading genuine and forged signature images**
- **Applying Neural Style Transfer (NST)** to generate synthetic forgeries
- **Extracting features** using **MobileNetV2** (a pre-trained CNN model)
- **Training an SVM classifier** on the extracted features
- **Evaluating the model** for accuracy and performance

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for MobileNetV2 and Neural Style Transfer)
- **OpenCV** (for image processing)
- **Scikit-learn** (for SVM training & evaluation)
- **NumPy** (for numerical operations)

## Dataset
- **Genuine Signatures**: Located in `dataset/signatures/full_org`
- **Forged Signatures**: Located in `dataset/signatures/full_forg`
- **Style Images for NST**: Located in `dataset/archive (1)/artbench-10-python`

## How It Works
### 1️ Load Signature Images
The script loads genuine and forged signatures from the specified dataset directories using OpenCV.
```python
load_images(folder, label)
```

### 2️ Apply Neural Style Transfer (NST)
NST is applied to genuine signatures using style images to generate synthetic forgeries.
```python
apply_nst(content_img, style_img)
```

### 3️ Extract Features using MobileNetV2
The **MobileNetV2** model extracts deep features from the signature images.
```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
```

### 4️ Train SVM Classifier
An **SVM classifier** is trained on the extracted features.
```python
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
```

### 5️ Evaluate Model
The trained model is evaluated using **accuracy score** and a classification report.
```python
accuracy_score(y_test, y_pred)
```

## 📈 Results
- **Achieved Accuracy**: ✅ `{accuracy:.4f}` (as printed in the output)
- **Classification Report**: Printed in the console

## 🔧 Setup & Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ryanfer123/signature-forgery-detection.git
   cd signature-forgery-detection
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow opencv-python numpy scikit-learn
   ```
3. Run the script:
   ```sh
   python signature_detection.py
   ```

## Disclaimer
- Modify dataset paths in the script if needed.

---

🔗 **GitHub Repository**: [ryanfer123/signature-forgery-detection](https://github.com/ryanfer123/CSI-ML)

