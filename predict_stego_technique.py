import sys
import os
import joblib
import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.feature_extraction import extract_image_features

def predict_best_technique(image_path: str, model_type: str = "Model_A") -> dict:
    """
    Loads trained AI model, extracts intrinsic cover-image features from image_path,
    and predicts the optimal steganography technique (LSB, DCT, or DWT).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Determine model file path
    models_dir = os.path.join(os.path.dirname(__file__), "results", "models")
    model_filename = "Adaptive_Stego_Classifier.joblib"
    model_path = os.path.join(models_dir, model_filename)

    if not os.path.exists(model_path):
        model_filename = "Model_A_Image_Only.joblib"
        model_path = os.path.join(models_dir, model_filename)

    # Load trained model package
    package = joblib.load(model_path)
    model = package['model']
    scaler = package['scaler']
    feature_cols = package['feature_cols']

    # Extract intrinsic image features
    raw_feats = extract_image_features(image_path)
    
    # Filter features matching trained model inputs
    feat_df = pd.DataFrame([{col: raw_feats[col] for col in feature_cols if col in raw_feats}])
    
    # Scale features & predict
    feat_scaled = scaler.transform(feat_df)
    
    prediction = model.predict(feat_scaled)[0]
    raw_probs = dict(zip(model.classes_, model.predict_proba(feat_scaled)[0]))
    prob_dict = {cls: float(prob) for cls, prob in raw_probs.items()}

    return {
        'image_path': image_path,
        'model_used': model_filename,
        'predicted_best_technique': prediction,
        'class_probabilities': prob_dict
    }

if __name__ == '__main__':
    # Default test image if no command line argument provided
    default_img = "/Users/kaniishkselvakumar/Downloads/BOSSbase_1.01/1.pgm"
    target_img = sys.argv[1] if len(sys.argv) > 1 else default_img

    print("=" * 60)
    print(" ADAPTIVE STEGANOGRAPHY AI TECHNIQUE PREDICTOR ")
    print("=" * 60)
    
    result = predict_best_technique(target_img, model_type="Model_A")
    
    print(f"  Input Cover Image : {result['image_path']}")
    print(f"  AI Model Used     : {result['model_used']}")
    print(f"\n  >>> PREDICTED OPTIMAL TECHNIQUE: [ {result['predicted_best_technique']} ] <<<")
    print("\n  Class Confidence Probabilities:")
    for tech, prob in result['class_probabilities'].items():
        print(f"    - {tech:5s}: {prob*100:6.2f}%")
    print("=" * 60)
