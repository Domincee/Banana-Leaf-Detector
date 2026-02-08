import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import datetime

# Singleton instance
db = None

def init_firebase():
    """
    Initialize Firebase Admin SDK using environment variable.
    Returns the Firestore client.
    """
    global db
    if db:
        return db

    # Check if app is already initialized
    try:
        app = firebase_admin.get_app()
        db = firestore.client()
        return db
    except ValueError:
        pass # Not initialized

    cred_json = os.environ.get('FIREBASE_CREDENTIALS')
    if not cred_json:
        print("⚠️ FIREBASE_CREDENTIALS env var not found. Firebase features will be disabled.")
        return None

    try:
        # Parse the JSON string (it might be a file path or the actual JSON content)
        if os.path.exists(cred_json):
             cred = credentials.Certificate(cred_json)
        else:
             # Assume it's the JSON string content
             cred_dict = json.loads(cred_json)
             cred = credentials.Certificate(cred_dict)
             
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized successfully.")
        return db
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        return None

def save_feature_to_firestore(features, label, filename, existing_hash=None):
    """
    Save extracted features to 'dataset' collection.
    features: List or Array of 59 float values
    label: String
    filename: String
    existing_hash: String (optional, for deduplication)
    """
    db = init_firebase()
    if not db:
        return False

    try:
        doc_ref = db.collection('dataset').document(filename)
        data = {
            'filename': filename,
            'label': label,
            'features': features.tolist() if hasattr(features, 'tolist') else features,
            'hash': existing_hash,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(data)
        print(f"🔥 Saved features for {filename} to Firestore.")
        return True
    except Exception as e:
        print(f"❌ Firestore Save Error: {e}")
        return False

def save_feedback_to_firestore(filename, prediction, is_correct, actual_label):
    """
    Save user feedback to 'feedback' collection.
    """
    db = init_firebase()
    if not db:
        return False

    try:
        # Use filename as ID? Or auto-generate? 
        # Feedback is unique per event, so maybe auto-id is safer, 
        # but let's use filename if we want one feedback per image upload session.
        # Actually, let's allow multiple feedback entries (though unlikely for same image).
        
        data = {
            'filename': filename,
            'prediction': prediction,
            'is_correct': is_correct,
            'actual_label': actual_label,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        # Add to 'feedback' collection
        db.collection('feedback').add(data)
        print(f"🔥 Saved feedback for {filename} to Firestore.")
        return True
    except Exception as e:
        print(f"❌ Firestore Feedback Error: {e}")
        return False

def get_feedback_history(limit=50):
    """
    Retrieve recent feedback history.
    """
    db = init_firebase()
    if not db:
        return []

    try:
        docs = db.collection('feedback').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
        history = []
        for doc in docs:
            d = doc.to_dict()
            
            # Format status for UI
            is_correct = d.get('is_correct', False)
            actual = d.get('actual_label')
            pred = d.get('prediction')
            
            status = "unknown"
            if is_correct:
                status = "correct"
            elif actual and actual != pred:
                status = "corrected"
            else:
                status = "incorrect"

            history.append({
                "timestamp": d.get('timestamp').isoformat() if d.get('timestamp') else "",
                "filename": d.get('filename'),
                "prediction": pred,
                "actual": actual,
                "status": status
            })
        return history
    except Exception as e:
        print(f"❌ Firestore Read Error: {e}")
        return []

def get_all_dataset_features():
    """
    Retrieve ALL features for retraining.
    Returns: pandas DataFrame or list of dicts suitable for training.
    """
    db = init_firebase()
    if not db:
        return []
        
    try:
        docs = db.collection('dataset').stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            # Flatten structure: features list -> feat_0, feat_1...
            row = {'label': d.get('label'), 'path': d.get('filename'), 'hash': d.get('hash')} # path/hash for compat
            feats = d.get('features', [])
            for i, f in enumerate(feats):
                row[f'feat_{i}'] = f # Assuming order is preserved
            data.append(row)
        return data
    except Exception as e:
        print(f"❌ Firestore Dataset Error: {e}")
        return []
