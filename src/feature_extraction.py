import numpy as np
import cv2
import pywt
from scipy.stats import entropy, skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

def extract_image_features(image_path: str) -> dict:
    """
    Extracts 22 intrinsic cover-image content features:
    - 1-6: Intensity statistics (mean_intensity, std_intensity, variance, entropy, min_intensity, max_intensity)
    - 7-10: Gradient & Edge features (edge_density, gradient_mean, gradient_std, laplacian_var)
    - 11-14: GLCM Texture features (glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity)
    - 15: DCT high-frequency energy fraction (dct_high_freq_energy)
    - 16-19: DWT sub-band energy ratios (dwt_ll_energy, dwt_lh_energy, dwt_hl_energy, dwt_hh_energy)
    - 20-22: Higher-order spatial features (skewness, kurtosis, corner_density)
    """
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Handle multi-channel images (RGBA, RGB)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_float = img.astype(np.float64)
    h, w = img.shape
    total_pixels = h * w

    features = {}

    # 1. Mean Intensity
    features['mean_intensity'] = float(np.mean(img_float))

    # 2. Standard Deviation Intensity
    features['std_intensity'] = float(np.std(img_float))
    features['std_dev'] = features['std_intensity'] # alias

    # 3. Pixel Variance
    features['variance'] = float(np.var(img_float))

    # 4. Shannon Entropy (256 bins, base 2)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256), density=True)
    features['entropy'] = float(entropy(hist + 1e-12, base=2))

    # 5-6. Min & Max Intensity
    features['min_intensity'] = float(np.min(img))
    features['max_intensity'] = float(np.max(img))

    # 7. Canny Edge Density (thresholds 50/150)
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    features['edge_density'] = float(np.count_nonzero(edges) / total_pixels)

    # 8-9. Sobel Gradient Magnitude Mean & Standard Deviation (ksize=3)
    sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    features['gradient_mean'] = float(np.mean(grad_mag))
    features['gradient_std'] = float(np.std(grad_mag))

    # 10. Laplacian Variance (ksize=1)
    features['laplacian_var'] = float(cv2.Laplacian(img_float, cv2.CV_64F, ksize=1).var())
    features['laplacian_variance'] = features['laplacian_var'] # alias

    # 11-14. GLCM Texture Features (64-level Quantization, 4 angles averaged: 0, pi/4, pi/2, 3*pi/4)
    img_quant_64 = (img // 4).astype(np.uint8)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_quant_64, distances=[1], angles=angles, levels=64, symmetric=True, normed=True)
    
    features['glcm_contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
    features['glcm_correlation'] = float(np.mean(graycoprops(glcm, 'correlation')))
    features['glcm_energy'] = float(np.mean(graycoprops(glcm, 'energy')))
    features['glcm_homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))

    # 15. DCT High-Frequency Energy Fraction (High-High DCT Quadrant)
    # Ensure even dimensions for cv2.dct
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    dct_crop = img_float[:h_even, :w_even]
    dct_img = cv2.dct(dct_crop)
    dct_total = float(np.sum(dct_img ** 2))
    dct_hh_quad = float(np.sum(dct_img[h_even//2:, w_even//2:] ** 2))
    features['dct_high_freq_energy'] = float(dct_hh_quad / (dct_total + 1e-12))
    features['dct_high_freq_energy_ratio'] = features['dct_high_freq_energy'] # alias

    # 16-19. DWT Sub-band Energy Ratios (2D Haar Wavelet, Level 1)
    coeffs = pywt.dwt2(img_float, 'haar')
    LL, (LH, HL, HH) = coeffs

    e_LL = float(np.sum(LL ** 2))
    e_LH = float(np.sum(LH ** 2))
    e_HL = float(np.sum(HL ** 2))
    e_HH = float(np.sum(HH ** 2))
    e_dwt_total = e_LL + e_LH + e_HL + e_HH + 1e-12

    features['dwt_ll_energy'] = float(e_LL / e_dwt_total)
    features['dwt_lh_energy'] = float(e_LH / e_dwt_total)
    features['dwt_hl_energy'] = float(e_HL / e_dwt_total)
    features['dwt_hh_energy'] = float(e_HH / e_dwt_total)

    features['dwt_ll_energy_ratio'] = features['dwt_ll_energy']
    features['dwt_lh_energy_ratio'] = features['dwt_lh_energy']
    features['dwt_hl_energy_ratio'] = features['dwt_hl_energy']
    features['dwt_hh_energy_ratio'] = features['dwt_hh_energy']

    # 20. Skewness
    features['skewness'] = float(skew(img.flatten()))

    # 21. Kurtosis
    features['kurtosis'] = float(kurtosis(img.flatten()))

    # 22. Harris Corner Density (goodFeaturesToTrack, maxCorners=1000, minDistance=10)
    corners = cv2.goodFeaturesToTrack(img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    corner_count = len(corners) if corners is not None else 0
    features['corner_density'] = float(corner_count / total_pixels)

    return features
