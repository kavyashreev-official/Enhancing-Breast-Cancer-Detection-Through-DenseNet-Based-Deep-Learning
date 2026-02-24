# 🧠 Enhancing Breast Cancer Detection Through DenseNet Based Deep Learning

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection significantly improves survival rates. This project proposes a **DenseNet-based deep learning model** to enhance the accuracy and reliability of breast cancer detection from medical imaging data.

The model leverages **transfer learning** and the powerful feature extraction capability of DenseNet to classify images as **benign or insitu or invasive or malignant or normal**, improving diagnostic performance compared to traditional CNN models.

## 🎯 Objectives

* Develop a deep learning model using **DenseNet architecture**
* Improve classification accuracy for breast cancer detection
* Reduce false positives and false negatives
* Compare performance with conventional CNN models
* Provide a reliable AI-assisted diagnostic tool

## 🏗️ Model Architecture

This project uses:

* **DenseNet (Dense Convolutional Network)**
* Transfer Learning (Pre-trained on ImageNet)
* Fully Connected Layers for classification
* Softmax/Sigmoid activation for binary classification

### Why DenseNet?

* Efficient feature reuse
* Reduces vanishing gradient problem
* Fewer parameters than traditional CNNs
* Strong performance on medical imaging tasks

## 📊 Dataset

The model is trained on breast cancer imaging datasets such as:

* Histopathology image datasets
* Mammogram datasets 

Dataset includes:

* Benign tumor images
* Malignant tumor images

Preprocessing steps:

* Image resizing
* Normalization
* Data augmentation
* Train-test split

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* PyTorch (if used)
* NumPy
* Pandas
* Matplotlib
* OpenCV

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kavyashreev-offical/Enhancing-Breast-Cancer-Detection-DenseNet.git
cd Enhancing-Breast-Cancer-Detection-DenseNet
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Model

```bash
python train.py
```

or

```bash
python main.py
```

## 📈 Performance Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Curve

Example Results:

* Accuracy: ~95%+
* Improved sensitivity for malignant detection
* Reduced false negative rate

## 🔬 Methodology

1. Data Collection
2. Data Preprocessing
3. Data Augmentation
4. DenseNet Model Initialization
5. Transfer Learning
6. Model Training
7. Evaluation & Testing
8. Performance Comparison

## 📷 Sample Output

* Classification prediction (Benign / Malignant / Normal / InSitu / Malignant)
* Probability score
* Confusion matrix visualization

## 🚀 Future Enhancements

* Integration with hospital diagnostic systems
* Web-based prediction interface
* Explainable AI (Grad-CAM visualization)
* Multi-class tumor classification
* Real-time detection from uploaded images

## 🏥 Real-World Impact

* Assists radiologists in faster diagnosis
* Reduces human error
* Supports early-stage cancer detection
* Improves survival rates through timely intervention

## 🤝 Contribution

Contributions are welcome!
Feel free to fork this repository and submit pull requests.

## 📄 License

This project is for academic and research purposes.

## 👩‍💻 Author

**Kavyashree V**
Passionate about AI, Deep Learning, and Healthcare Innovation

GitHub: [https://github.com/kavyashreev-official](https://github.com/kavyashreev-official)
