import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import os
import time

DATA_CSV = "data.csv"

def init_firebase_local():
    """
    Initialize Firebase from local credentials file or env var.
    """
    cred_path = os.environ.get("FIREBASE_CREDENTIALS", "firebase_service_account.json")
    
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
    elif os.environ.get("FIREBASE_CREDENTIALS"):
        # Try parsing as JSON string
        import json
        cred_dict = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
        cred = credentials.Certificate(cred_dict)
    else:
        print("❌ No credentials found. Set FIREBASE_CREDENTIALS env var or place firebase_service_account.json in root.")
        return None

    try:
        app = firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred)
        
    return firestore.client()

def migrate():
    if not os.path.exists(DATA_CSV):
        print(f"❌ {DATA_CSV} not found.")
        return

    db = init_firebase_local()
    if not db:
        return

    print("🚀 Starting migration...")
    df = pd.read_csv(DATA_CSV)
    
    # Batch writes for efficiency
    batch = db.batch()
    count = 0
    total = len(df)
    
    for i, row in df.iterrows():
        # Using path relative filename as ID seems reasonable, or just hash if unique
        # Let's use filename from path
        filename = os.path.basename(row['path'])
        
        # Prepare data
        doc_data = {
            'filename': filename,
            'label': row['label'],
            'hash': row.get('hash'),
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        # Extract features (all columns > index 2 usually)
        # We need to recognize which cols are features.
        # In extract_features.py, cols are: path, label, hash, [features...]
        # So features start at index 3
        features = row.iloc[3:].tolist()
        doc_data['features'] = features
        
        doc_ref = db.collection('dataset').document(filename)
        batch.set(doc_ref, doc_data)
        count += 1
        
        if count % 400 == 0: # Firestore batch limit is 500
            batch.commit()
            print(f"✅ Committed {count}/{total} records...")
            batch = db.batch()
            time.sleep(1) # Rate limit guard
            
    if count % 400 != 0:
        batch.commit()
        
    print(f"🎉 Migration complete! {count} records uploaded.")

if __name__ == "__main__":
    migrate()
