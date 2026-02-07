# 🍌 Banana Leaf Disease Detector

A machine learning web application built with **Python, OpenCV, scikit-learn, and Flask** to detect whether a banana leaf is **Healthy**, **Unhealthy**, or **Not a Leaf** using image processing and KNN classification.

---

## 📖 Project Overview

This project extracts texture and color-based features (using **GLCM**, **LBP**, and **HOG**) from banana leaf images, trains a **K-Nearest Neighbors (KNN)** classifier, and serves the prediction through a simple Flask web app.

### ✨ Key Features

* 🧠 Trained KNN model for 3-class classification
* 🎨 Uses advanced image feature extraction (GLCM, LBP, HOG)
* 🧾 Visualizes feature importance and class balance
* 🌐 Flask web interface for uploading and detecting banana leaves
* 🧩 Dataset augmentation and scaling with `MinMaxScaler`

---

## 📂 Project Structure

```
project/
│
├── app.py                  # Flask app (main entry)
├── extract_features.py     # Feature extraction functions
├── generate_aug.py         # Data augmentation script
├── knn_trainer.py         # Model training script
├── scale.py               # Feature scaling and preprocessing
├── visualization.ipynb    # Data visualization and analysis
│
├── dataset/               # Dataset organization
│   ├── raw_data/         # Original dataset
│   │   ├── Diseased_leaf/
│   │   ├── Healthy_leaf/
│   │   └── Non_leaf/
│   ├── train_data/       # Training dataset
│   └── test_data/        # Testing dataset
│
├── static/               # Static files for web interface
│   └── styles.css        # CSS styling
│
├── templates/            # HTML templates
│   └── index.html       # Main web interface
│
├── uploads/             # Temporary storage for uploaded images
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 How to Run the App

1. **Clone the repository**

   ```bash
   git clone https://github.com/Domincee/Banana-Leaf-Detector.git
   cd Banana-Leaf-Detector
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```
   or
   ```bash
   ./venv/bin/python app.py
   ```

5. **Open in your browser**

   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 Model Performance

| Metric  | Training Accuracy | Test Accuracy |
| ------- | ----------------- | ------------- |
| **KNN** | 0.9991            | 0.90423       |

**Classification Report (Test Set)**


                precision    recall  f1-score   support

  Healthy Leaf       0.91      0.94      0.92       149
     None-leaf       0.93      0.87      0.90       150
Unhealthy leaf       0.88      0.90      0.89       150

      accuracy                           0.90       449
     macro avg       0.90      0.90      0.90       449
  weighted avg       0.90      0.90      0.90       449

---

## 🧬 Dataset Information

The dataset consists of **2,000+ images**, resized to **128×128**, including:

* **Healthy banana leaves** (Augmented images)
* **Diseased banana leaves** (Actual images)
* **Non-banana images** (negative samples,self-collected)

> ⚠️ Raw and training datasets are not included in this repository due to file size limits.
> You can download them from: (https://drive.google.com/drive/folders/1mng06d0Y_U4hC7WM5hnbBNbuC5ohulcq?usp=sharing)

---

## 🧩 Technologies Used

* **Python 3.11**
* **OpenCV**
* **NumPy & Pandas**
* **scikit-learn**
* **scikit-image**
* **Matplotlib / Seaborn**
* **Flask**

---




## 🪪 License

© 2025 Domince Aseberos. All rights reserved.

This project is released under the **MIT License**.

You are free to use, copy, modify, and distribute this software for educational or research purposes, provided that proper credit is given to the author.

> ⚠️ Note: The dataset and sample images are for demonstration and research purposes only.  
> They may contain content collected from public sources and are **not included in this repository** to comply with data-sharing and copyright policies.
