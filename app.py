from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv

# Load .env file (silent=True to avoid errors if .env is missing in production)
load_dotenv()
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
from extract_features import extract_features
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from firebase_helpers import init_firebase, save_feature_to_firestore, save_feedback_to_firestore, get_feedback_history, get_analytics_data


app = Flask(__name__)
CORS(app)

# Config
# Config
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 10)) * 1024 * 1024
# Use /tmp for Vercel (or any read-only FS environment)
UPLOAD_FOLDER = '/tmp' if os.environ.get('VERCEL') else 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FEEDBACK_FILE = "feedback.csv"
DATA_CSV = "data.csv"

# Check if running on Vercel
IS_VERCEL = bool(os.environ.get('VERCEL'))

# Global feature cache for active learning
FEATURE_CACHE = {}

MODEL_PATH = os.environ.get("MODEL_PATH", "knn_model.pkl")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Init Firebase
db = init_firebase()


# ============================
# Load trained KNN model
# ============================
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    knn = model_data["model"]                   # trained KNN model
    scaler = model_data["scaler"]               # MinMaxScaler
    label_encoder_classes = model_data["classes"]  # class names

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Initialize dummies to prevent immediate crash, though upload will fail
    knn, scaler, label_encoder = None, None, None

def retrain_model():
    """
    Retrains the KNN model using the updated data.csv.
    This mimics the logic in knn_trainer.py but runs in-process.
    """
    global knn, scaler, label_encoder_classes, label_encoder
    
    print("🔄 Retraining model with new data...")
    
    if IS_VERCEL:
        print("⚠️ Cannot retrain model on Vercel (Read-Only Filesystem). Skipping.")
        return
    
    if not os.path.exists(DATA_CSV):
        print("❌ CSV file not found. Skipping retrain.")
        return

    df = pd.read_csv(DATA_CSV)
    
    # Drop path/hash if they exist, keep only features + label
    # The CSV structure is: path, label, hash, feat1, feat2... 
    # But wait, existing extract_features puts path/label/hash in first 3 cols.
    # We must ensure we align with that.
    
    # Standardize labels
    df['label'] = df['label'].replace('Diseased leaf', 'Unhealthy leaf')
    
    # Balancing logic (Simple version: standardizing 'None-leaf' downsampling)
    df_healthy = df[df['label'] == 'Healthy Leaf']
    df_unhealthy = df[df['label'] == 'Unhealthy leaf']
    df_none = df[df['label'] == 'None-leaf'] # Or 'Non-leaf' depending on standardized name
    # Fix potential label mismatch
    df_none = df[df['label'].str.lower().str.contains('non')] 

    # If we have very few samples, skip complex balancing to avoid crashes
    if len(df_healthy) < 5 or len(df_unhealthy) < 5:
        print("⚠️ Not enough data to balance correctly. Training on raw data.")
        df_balanced = df
    else:
        # Downsample majority (usually non-leaf) to match healthy count (or at least reasonable size)
        target_count = max(len(df_healthy), len(df_unhealthy))
        if len(df_none) > target_count:
            df_none_down = resample(df_none, replace=False, n_samples=target_count, random_state=42)
            df_balanced = pd.concat([df_healthy, df_unhealthy, df_none_down])
        else:
            df_balanced = pd.concat([df_healthy, df_unhealthy, df_none])

    # Prepare X and y
    # Features start from column 2 (indices 0=path, 1=label) in original CSV
    # The structure is: path, label, feat1, feat2...
    
    # Drop non-feature columns. 
    # We know features are numeric.
    X = df_balanced.iloc[:, 2:].values # Columns 2 onwards are features
    y = df_balanced['label'].values
    
    # Re-fit Label Encoder
    le_new = LabelEncoder()
    y_encoded = le_new.fit_transform(y)
    
    # Re-fit Scaler
    scaler_new = MinMaxScaler()
    X_scaled = scaler_new.fit_transform(X)
    
    # Train KNN (Best params from previous grid search: k=5, weights=distance usually)
    # We'll stick to a robust default or what was loaded
    knn_new = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
    knn_new.fit(X_scaled, y_encoded)
    
    # Update Globals
    knn = knn_new
    scaler = scaler_new
    label_encoder = le_new
    label_encoder_classes = le_new.classes_
    label_encoder.classes_ = np.array(label_encoder_classes)
    
    # Save to disk
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'model': knn, 'scaler': scaler, 'classes': label_encoder_classes}, f)
        
    print("✅ Model successfully retrained and saved.")

def generate_explanation(probabilities, prediction):
    """
    Generates a human-readable explanation based on confidence scores.
    probabilities: dict { 'Healthy': 80, 'Unhealthy': 10, ... }
    prediction: str (Key of the highest probability)
    """
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_probs[0]
    second_label, second_score = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
    
    explanation = ""

    # Confidence Logic
    if top_score >= 90:
        explanation = f"The model is highly confident ({top_score}%) that this is {top_label}. The visual features (color density, texture patterns) match the standard profile for this category very closely."
    elif top_score >= 70:
        explanation = f"The model is fairly confident ({top_score}%) in the {top_label} classification. Most features align, though there might be slight variations in lighting or leaf texture."
    elif top_score >= 50:
        explanation = f"The analysis suggests {top_label} ({top_score}%), but there is some ambiguity. The model also detected similarities to {second_label} ({second_score}%). This often happens with early-stage disease or poor lighting."
    else:
        explanation = f"The result is uncertain. While {top_label} was the top match ({top_score}%), the features are very mixed, showing strong similarities to {second_label} ({second_score}%). "

    # Specific Edge Case Explanations
    if "Unhealthy" in top_label:
        explanation += " Distinct discoloration or textural irregularities were detected on the leaf surface."
    elif "Healthy" in top_label and probabilities.get("Unhealthy", 0) > 20:
        explanation += " However, some small irregularities were noted, so keep an eye on the plant."
    elif "Non-Leaf" in top_label:
        explanation += " The image lacks the specific green/yellow color histograms and vein textures typically found in banana leaves."

    return explanation

# ============================
# Routes
# ============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/healthz")
def healthz():
    return {"ok": True}, 200

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower() or ".jpg"
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        # Generate unique filename for persistence
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Extract features and scale
        features = extract_features(file_path)
        
        # Cache features for active learning (feedback loop)
        # Key by the unique filename now
        FEATURE_CACHE[unique_filename] = features
        
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict class
        pred_encoded = knn.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        if "Diseased" in pred_label:
            pred_label = pred_label.replace("Diseased", "Unhealthy")

        # Predict probabilities (use knn.classes_ for correct order)
        prob_array = knn.predict_proba(features_scaled)[0]
        prob_class_labels = label_encoder.inverse_transform(knn.classes_)
        prob_dict = {}
        for cls, prob in zip(prob_class_labels, prob_array):
            cls_name = cls.replace("Diseased", "Unhealthy") if "Diseased" in cls else cls
            prob_dict[cls_name] = int(round(prob * 100))

        # Generate Explanation
        explanation = generate_explanation(prob_dict, pred_label)

        print(f"Image uploaded: {unique_filename} -> {pred_label} | {prob_dict}")

        return jsonify({
            "success": True,
            "prediction": pred_label,
            "probabilities": prob_dict,
            "explanation": explanation,
            "filename": unique_filename # Return the unique name
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    # No finally block -> Image persists

@app.route("/feedback", methods=["POST"])
def save_feedback():
    try:
        data = request.json
        filename = data.get("filename", "unknown")
        prediction = data.get("prediction", "unknown")
        is_correct = data.get("correct", False)
        # Fix: handle explicit None from frontend
        actual_label = data.get("actual_label")
        if not actual_label and is_correct:
            actual_label = prediction
        elif not actual_label:
            actual_label = "Unknown"
        
        # ACTIVE LEARNING LOGIC
        if filename in FEATURE_CACHE:
            features = FEATURE_CACHE[filename]
            
            # Standardize label
            label_map = {
                'Healthy Leaf': 'Healthy Leaf',
                'Unhealthy Leaf': 'Unhealthy leaf',
                'Non-Leaf': 'None-leaf' 
            }
            standardized_label = label_map.get(actual_label, actual_label)
            
            # Save to Firestore (Features)
            success = save_feature_to_firestore(features, standardized_label, filename)
            
            if success:
                print(f"📝 Active Learning: Saved {standardized_label} to Firestore.")
            
            # Clear cache
            del FEATURE_CACHE[filename]

        # Log to Firestore (Feedback)
        save_feedback_to_firestore(filename, prediction, is_correct, actual_label)
            
        return jsonify({"success": True})
    except Exception as e:
        print(e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analytics")
def analytics_api():
    try:
        data = get_analytics_data(limit=1000) # Fetch more for analytics
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history_route():
    # Fetch from Firestore
    history = get_feedback_history()
    return jsonify(history)
@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        # Just open in write mode to truncate
        with open(FEEDBACK_FILE, "w") as f:
            f.write("timestamp,filename,prediction,is_correct,actual_label\n") # Keep header
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)