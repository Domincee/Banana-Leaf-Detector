import cv2
import numpy as np
import os
from extract_features import extract_features

# Create dummy image (100x100 green square)
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:] = (0, 255, 0) # Green
cv2.imwrite("dummy_test.png", img)

try:
    print("Testing extraction...")
    config_path = os.path.abspath("dummy_test.png")
    features = extract_features(config_path)
    print(f"Extraction successful! Shape: {features.shape}")
except Exception as e:
    print(f"Extraction failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists("dummy_test.png"):
        os.remove("dummy_test.png")
