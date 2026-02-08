# 🍌 Banana Leaf Disease Detector & Active Learning System

A smart machine learning web application that detects **Healthy**, **Unhealthy**, and **Non-Leaf** images. It features an **Active Learning** system that improves the model in real-time based on user feedback.

---

## ✨ Key Features

### 1. 🔍 Real-time Disease Detection
*   Analyzes images using advanced feature extraction (**GLCM**, **LBP**, **HOG**, **Color Histograms**).
*   Classifies leaves using a **K-Nearest Neighbors (KNN)** model.
*   Provides detailed confidence scores and visual explanations.

### 2. 🧠 Active Learning (On-the-Fly Training)
*   **Learns from Mistakes:** If the model predicts incorrectly, you can provide the correct label.
*   **Instant Retraining:** The system adds your feedback to the training dataset and immediately **retrains the model** in the background.
*   **Continuous Improvement:** The more you use it, the smarter it gets!

### 3. 📜 Visual Analysis History
*   Keeps a persistent log of your recent scans.
*   **Thumbnails:** Displays the analyzed images stored securely on the server.
*   **Status Indicators:** Clearly shows if a prediction was accurate, incorrect, or corrected by you.
*   **Management:** Includes a "Clear History" option to wipe the log.

---

## 🛠️ Technology Stack

*   **Backend:** Python 3.12, Flask
*   **Database:** Firebase Firestore (Features & Feedback Logging)
*   **Machine Learning:** scikit-learn (KNN), NumPy, Pandas
*   **Computer Vision:** OpenCV, scikit-image
*   **Frontend:** HTML5, CSS3, JavaScript (Vanilla), Chart.js (Analytics)
*   **Deployment:** Vercel / Render (Compatible)

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Domincee/Banana-Leaf-Detector.git
cd Banana-Leaf-Detector
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Firebase
Ensure you have your `firebase_credentials.json` or set the `FIREBASE_CREDENTIALS` environment variable.

### 5. Run the Application
```bash
python app.py
```
The app will start at `http://127.0.0.1:5000`.

---

## 📂 Project Structure

```
project/
│
├── app.py                  # Main Flask application & Active Learning logic
├── firebase_helpers.py     # Firestore interaction (Features/Feedback)
├── extract_features.py     # Feature extraction (Color, Texture, Shape)
├── knn_trainer.py          # Initial model training script
│
├── dataset/                # Training images
├── static/
│   ├── uploads/            # Temporary storage for uploads
│   └── styles.css          # UI Styling
├── templates/
│   └── index.html          # Frontend Interface
│
├── data.csv                # Initial Dataset
├── knn_model.pkl           # Serialized trained model
└── requirements.txt        # Python dependencies
```

---

## 🧠 Model & Feature Details

The system extracts **59 unique features** from each image:
*   **Color (HSV, LAB, Grayscale):** Means, standard deviations, and hue histograms to detect discoloration.
*   **Texture (GLCM, LBP):** Contrast, homogeneity, and local binary patterns to spot fungal textures.
*   **Shape:** Area, perimeter, and circularity to distinguish leaves from other objects.
*   **HOG (Histogram of Oriented Gradients):** Captures edge structures.

**Active Learning Workflow:**
1.  User uploads image -> Model predicts.
2.  User gives "Thumbs Down" -> Selects correct label.
3.  `app.py` saves extracted features + correct label to **Firebase Firestore**.
4.  Feedback is logged for future model retraining.

---

## 🪪 License

© 2025 Domince Aseberos. Released under the **MIT License**.
