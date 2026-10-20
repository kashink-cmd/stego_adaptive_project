import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate_metrics(cover: np.ndarray, stego: np.ndarray, orig_payload: np.ndarray, extracted_payload: np.ndarray) -> dict:
    """
    Computes visual quality (PSNR, SSIM) and extraction error (BER, Accuracy).
    """
    # Visual Quality Metrics
    psnr_val = float(compute_psnr(cover, stego, data_range=255))
    ssim_val = float(compute_ssim(cover, stego, data_range=255))
    
    # Robustness Metrics
    min_len = min(len(orig_payload), len(extracted_payload))
    if min_len == 0:
        ber_val = 1.0
        accuracy_val = 0.0
    else:
        bit_errors = np.count_nonzero(orig_payload[:min_len] != extracted_payload[:min_len])
        ber_val = float(bit_errors / min_len)
        accuracy_val = float(1.0 - ber_val)
        
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'ber': ber_val,
        'accuracy': accuracy_val
    }

