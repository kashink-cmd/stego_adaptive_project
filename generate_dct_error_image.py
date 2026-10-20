import os
import cv2
import numpy as np
import pandas as pd

from src.stego_engine import generate_random_bits, embed_dct, extract_dct, simulate_jpeg_attack
from src.evaluator import evaluate_metrics

def create_dct_error_image(output_path: str) -> dict:
    """
    Generates a synthetic high-frequency grid image subjected to heavy JPEG attack (QF=15)
    and Gaussian noise, causing DCT steganography to produce non-zero Bit Error Rate (BER > 0).
    """
    h, w = 512, 512
    # Create high-frequency diagonal checkerboard pattern + Gaussian noise
    x = np.linspace(0, 50 * np.pi, w)
    y = np.linspace(0, 50 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    pattern = 128 + 100 * np.sin(xx) * np.cos(yy)
    
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 25, (h, w))
    
    img = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    
    # Run DCT embedding under heavy JPEG attack (QF=15)
    payload_size = 512
    payload = generate_random_bits(payload_size, seed=42)
    stego_dct = embed_dct(img, payload, delta=20.0)
    
    # Heavy JPEG attack QF=15 + slight Gaussian blur
    attacked_img = simulate_jpeg_attack(stego_dct, quality_factor=15)
    attacked_img = cv2.GaussianBlur(attacked_img, (3, 3), 0)
    
    extracted_payload = extract_dct(attacked_img, payload_size, delta=20.0)
    m = evaluate_metrics(img, attacked_img, payload, extracted_payload)
    
    return {
        'image_path': output_path,
        'psnr': m['psnr'],
        'ssim': m['ssim'],
        'ber': m['ber'],
        'accuracy': m['accuracy']
    }

if __name__ == '__main__':
    out_file = os.path.join(os.path.dirname(__file__), "data", "sample_covers", "dct_error_demo.png")
    res = create_dct_error_image(out_file)
    print("=" * 60)
    print(" DCT BIT ERROR RATE (BER > 0) DEMONSTRATION ")
    print("=" * 60)
    print(f"  Generated Image Path : {res['image_path']}")
    print(f"  PSNR (dB)            : {res['psnr']:.2f}")
    print(f"  SSIM                 : {res['ssim']:.4f}")
    print(f"  Bit Error Rate (BER) : {res['ber']:.4f}  <-- NON-ZERO BER!")
    print(f"  Extraction Accuracy  : {res['accuracy']*100:.2f}%")
    print("=" * 60)

