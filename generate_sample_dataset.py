import os
import numpy as np
import cv2

def generate_sample_images(output_dir: str, num_images: int = 20):
    """
    Generates diverse synthetic grayscale test images (512x512) representing:
    - Smooth gradient images
    - High-frequency Gaussian noise/texture
    - Sharp geometric edge images
    - Sinusoidal wave patterns
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    
    for idx in range(num_images):
        img_type = idx % 4
        h, w = 512, 512
        img = np.zeros((h, w), dtype=np.float64)
        
        if img_type == 0:
            # Smooth gradient
            x = np.linspace(0, 255, w)
            y = np.linspace(0, 255, h)
            xx, yy = np.meshgrid(x, y)
            img = 0.5 * xx + 0.5 * yy
        elif img_type == 1:
            # High-frequency texture (Gaussian noise + local variance)
            img = rng.normal(128, 35, size=(h, w))
        elif img_type == 2:
            # Sharp geometric edges
            img.fill(50)
            cv2.rectangle(img, (100, 100), (400, 400), 200, -1)
            cv2.circle(img, (256, 256), 80, 100, -1)
        elif img_type == 3:
            # Sinusoidal wave pattern
            x = np.linspace(0, 10 * np.pi, w)
            y = np.linspace(0, 10 * np.pi, h)
            xx, yy = np.meshgrid(x, y)
            img = 128 + 60 * np.sin(xx) * np.cos(yy)
            
        img = np.clip(img, 0, 255).astype(np.uint8)
        file_path = os.path.join(output_dir, f"sample_{idx+1:04d}.png")
        cv2.imwrite(file_path, img)

if __name__ == '__main__':
    target_path = os.path.join(os.path.dirname(__file__), "data", "sample_covers")
    generate_sample_images(target_path, num_images=20)
    print(f"Generated sample images in {target_path}")

