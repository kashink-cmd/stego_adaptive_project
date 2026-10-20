import os
import cv2
import numpy as np
import pandas as pd
import joblib

from src.feature_extraction import extract_image_features

def generate_dct_winning_image(output_path: str):
    """
    Generates a 512x512 cover image composed of 8x8 DCT cosine basis patterns,
    tailored specifically to make DCT the optimal steganography technique.
    """
    h, w = 512, 512
    img = np.zeros((h, w), dtype=np.float64)
    
    # Generate 8x8 block DCT cosine grid pattern
    for u in range(0, h, 8):
        for v in range(0, w, 8):
            block = np.zeros((8, 8), dtype=np.float64)
            for i in range(8):
                for j in range(8):
                    block[i, j] = np.cos((2*i+1)*4*np.pi / 16.0) * np.cos((2*j+1)*3*np.pi / 16.0)
            img[u:u+8, v:v+8] = block
            
    img_scaled = np.clip(128 + 100 * img, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_scaled)
    return output_path

if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(__file__), "data", "sample_covers", "dct_optimal_cover.png")
    generate_dct_winning_image(out_path)
    
    # Verify Model A prediction
    pkg = joblib.load(os.path.join(os.path.dirname(__file__), "results", "models", "Model_A_Image_Only.joblib"))
    model = pkg['model']
    scaler = pkg['scaler']
    feature_cols = pkg['feature_cols']
    
    feats = extract_image_features(out_path)
    df = pd.DataFrame([{c: feats[c] for c in feature_cols if c in feats}])[feature_cols]
    pred = model.predict(scaler.transform(df))[0]
    probs = model.predict_proba(scaler.transform(df))[0]
    
    print("=" * 60)
    print(" DCT OPTIMAL IMAGE GENERATOR ")
    print("=" * 60)
    print(f"  Generated File Path : {out_path}")
    print(f"  AI Model Prediction : [ {pred} ]")
    for cls, prob in zip(model.classes_, probs):
        print(f"    - {cls:5s}: {prob*100:6.2f}%")
    print("=" * 60)

