import numpy as np
import cv2
import pywt
import io
from PIL import Image

def generate_random_bits(num_bits: int, seed: int = 42) -> np.ndarray:
    """Generates a reproducible binary secret payload vector of 0s and 1s."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=num_bits, dtype=np.uint8)

def text_to_bits(text: str) -> np.ndarray:
    """Converts a text string into an array of 8-bit binary integers (0 and 1)."""
    byte_arr = bytearray(text.encode('utf-8', errors='replace'))
    bits = []
    for b in byte_arr:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return np.array(bits, dtype=np.uint8)

def bits_to_text(bits: np.ndarray) -> str:
    """Converts binary bit array back to a decoded UTF-8 string."""
    byte_list = []
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | int(bit)
        byte_list.append(byte)
    return bytes(byte_list).decode('utf-8', errors='replace')

# ==========================================
# Error Correction Codes (ECC) - Repetition & Majority Voting
# ==========================================
def encode_repetition_ecc(bits: np.ndarray, reps: int = 3) -> np.ndarray:
    """Encodes bitstream with Repetition Error Correcting Code."""
    return np.repeat(bits, reps)

def decode_repetition_ecc(bits: np.ndarray, reps: int = 3) -> np.ndarray:
    """Decodes bitstream using Majority Voting across repeated symbols."""
    n = len(bits) - (len(bits) % reps)
    trimmed = bits[:n].reshape(-1, reps)
    voted = (np.sum(trimmed, axis=1) > (reps / 2.0)).astype(np.uint8)
    return voted

# ==========================================
# 1. Spatial Least Significant Bit (LSB)
# ==========================================
def embed_lsb(cover: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """Embeds binary payload into the LSB of spatial pixels sequentially."""
    stego = cover.copy()
    flat = stego.flatten()
    if len(payload) > len(flat):
        payload = payload[:len(flat)]
    
    flat[:len(payload)] = (flat[:len(payload)] & 0xFE) | payload
    return flat.reshape(cover.shape)

def extract_lsb(stego: np.ndarray, payload_len: int) -> np.ndarray:
    """Extracts binary payload from spatial LSBs."""
    flat = stego.flatten()
    return flat[:payload_len] & 1


# Generate full 63 zig-zag AC positions per 8x8 block (supports up to 258,048 bits / ~32,256 characters)
def _generate_zigzag_positions():
    pos = []
    for s in range(1, 15):
        if s % 2 == 1:
            for i in range(max(0, s - 7), min(8, s + 1)):
                j = s - i
                pos.append((i, j))
        else:
            for j in range(max(0, s - 7), min(8, s + 1)):
                i = s - j
                pos.append((i, j))
    return pos

DCT_ALL_AC_POSITIONS = _generate_zigzag_positions()

# ==========================================
# 2. Discrete Cosine Transform (DCT)
# ==========================================
def embed_dct(cover: np.ndarray, payload: np.ndarray, delta: float = 40.0) -> np.ndarray:
    """
    Embeds binary payload in 8x8 block DCT coefficients using QIM.
    Dynamically scales from mid-frequency ACs up to full 63 zig-zag AC positions for high-capacity payloads (up to 258,048 bits).
    """
    h, w = cover.shape
    h_8, w_8 = (h // 8) * 8, (w // 8) * 8
    num_blocks = (h_8 // 8) * (w_8 // 8)
    max_capacity = num_blocks * len(DCT_ALL_AC_POSITIONS)
    
    if len(payload) > max_capacity:
        payload = payload[:max_capacity]
    
    coeffs_per_block = int(np.ceil(len(payload) / max(1, num_blocks)))
    active_positions = DCT_ALL_AC_POSITIONS[:max(1, coeffs_per_block)]
    
    stego_float = cover.astype(np.float64)
    bit_idx = 0
    
    for i in range(0, h_8, 8):
        for j in range(0, w_8, 8):
            if bit_idx >= len(payload):
                break
            
            block = cv2.dct(stego_float[i:i+8, j:j+8])
            for (ci, cj) in active_positions:
                if bit_idx >= len(payload):
                    break
                coeff = block[ci, cj]
                bit = payload[bit_idx]
                
                # QIM rule with robust quantization step
                block[ci, cj] = np.round((coeff - bit * (delta / 2.0)) / delta) * delta + bit * (delta / 2.0)
                bit_idx += 1
                
            stego_float[i:i+8, j:j+8] = cv2.idct(block)
        if bit_idx >= len(payload):
            break
            
    return np.clip(np.round(stego_float), 0, 255).astype(np.uint8)

def extract_dct(stego: np.ndarray, payload_len: int, delta: float = 40.0) -> np.ndarray:
    """Extracts payload from 8x8 DCT coefficients using QIM."""
    h, w = stego.shape
    h_8, w_8 = (h // 8) * 8, (w // 8) * 8
    num_blocks = (h_8 // 8) * (w_8 // 8)
    coeffs_per_block = int(np.ceil(payload_len / max(1, num_blocks)))
    active_positions = DCT_ALL_AC_POSITIONS[:max(1, coeffs_per_block)]
    
    extracted = []
    stego_float = stego.astype(np.float64)
    
    for i in range(0, h_8, 8):
        for j in range(0, w_8, 8):
            if len(extracted) >= payload_len:
                break
            block = cv2.dct(stego_float[i:i+8, j:j+8])
            for (ci, cj) in active_positions:
                if len(extracted) >= payload_len:
                    break
                coeff = block[ci, cj]
                extracted.append(int(np.round(coeff / (delta / 2.0))) % 2)
        if len(extracted) >= payload_len:
            break
            
    return np.array(extracted, dtype=np.uint8)


# ==========================================
# 3. Discrete Wavelet Transform (DWT)
# ==========================================
def embed_dwt(cover: np.ndarray, payload: np.ndarray, alpha: float = 48.0) -> np.ndarray:
    """
    Embeds binary payload using 2D Haar Wavelet Multi-Subband QIM modulation (LL -> LH -> HL -> HH).
    Supports payloads up to full 4-subband capacity (262,144 bits for 512x512).
    """
    h, w = cover.shape
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    crop = cover[:h_even, :w_even].astype(np.float64)
    
    LL, (LH, HL, HH) = pywt.dwt2(crop, 'haar')
    subbands = [LL, LH, HL, HH]
    sub_size = LL.size
    max_capacity = sub_size * 4
    
    if len(payload) > max_capacity:
        payload = payload[:max_capacity]
        
    flat_subs = [s.flatten() for s in subbands]
    bit_idx = 0
    for s_idx in range(4):
        for k in range(sub_size):
            if bit_idx >= len(payload):
                break
            c = flat_subs[s_idx][k]
            b = payload[bit_idx]
            flat_subs[s_idx][k] = np.round((c - b * (alpha / 2.0)) / alpha) * alpha + b * (alpha / 2.0)
            bit_idx += 1
        if bit_idx >= len(payload):
            break
            
    mod_LL = flat_subs[0].reshape(LL.shape)
    mod_LH = flat_subs[1].reshape(LH.shape)
    mod_HL = flat_subs[2].reshape(HL.shape)
    mod_HH = flat_subs[3].reshape(HH.shape)
    
    recon = pywt.idwt2((mod_LL, (mod_LH, mod_HL, mod_HH)), 'haar')
    recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
    
    stego = cover.copy()
    stego[:h_even, :w_even] = recon
    return stego

def extract_dwt(stego: np.ndarray, payload_len: int, alpha: float = 48.0) -> np.ndarray:
    """Extracts payload from 2D Haar Wavelet Multi-Subband coefficients."""
    h, w = stego.shape
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    crop = stego[:h_even, :w_even].astype(np.float64)
    
    LL, (LH, HL, HH) = pywt.dwt2(crop, 'haar')
    flat_subs = [LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()]
    
    extracted = []
    bit_idx = 0
    for s_idx in range(4):
        for k in range(LL.size):
            if len(extracted) >= payload_len:
                break
            c = flat_subs[s_idx][k]
            extracted.append(int(np.round(c / (alpha / 2.0))) % 2)
        if len(extracted) >= payload_len:
            break
            
    return np.array(extracted, dtype=np.uint8)


# ==========================================
# 4. Transmission & Robustness Attack Simulations
# ==========================================
def simulate_jpeg_attack(image: np.ndarray, quality_factor: int) -> np.ndarray:
    """1. JPEG Compression Attack: Compresses image at specified Quality Factor (1-100)."""
    if quality_factor >= 100:
        return image.copy()
    
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality_factor)
    buffer.seek(0)
    attacked_pil = Image.open(buffer)
    return np.array(attacked_pil, dtype=np.uint8)

def simulate_noise_attack(image: np.ndarray, noise_type: str = 'Gaussian', intensity: float = 10.0) -> np.ndarray:
    """2. Noise Attack: Adds Gaussian noise or Salt-and-Pepper impulse noise."""
    noisy = image.astype(np.float64)
    if noise_type == 'Gaussian':
        sigma = np.sqrt(max(0.1, intensity))
        gauss = np.random.normal(0, sigma, image.shape)
        return np.clip(noisy + gauss, 0, 255).astype(np.uint8)
    else:  # Salt & Pepper
        p = np.clip(intensity / 100.0, 0.001, 0.5)
        attacked = image.copy()
        num_salt = np.ceil(p * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        attacked[tuple(coords)] = 255
        num_pepper = np.ceil(p * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        attacked[tuple(coords)] = 0
        return attacked

def simulate_resizing_attack(image: np.ndarray, scale_percent: int = 75) -> np.ndarray:
    """3. Resizing Attack: Reduces dimensions and interpolates back, altering pixel values."""
    h, w = image.shape[:2]
    new_w = max(8, int(w * (scale_percent / 100.0)))
    new_h = max(8, int(h * (scale_percent / 100.0)))
    down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up

def simulate_filtering_attack(image: np.ndarray, filter_type: str = 'Median', kernel_size: int = 3) -> np.ndarray:
    """4. Filtering / Blurring Attack: Applies Median filter or Gaussian blur smoothing."""
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    if filter_type == 'Median':
        return cv2.medianBlur(image, k)
    else:  # Gaussian Blur
        return cv2.GaussianBlur(image, (k, k), 1.0)

def simulate_cropping_attack(image: np.ndarray, crop_percent: int = 10) -> np.ndarray:
    """5. Cropping Attack: Removes/masks a portion of the image to simulate packet/region loss."""
    attacked = image.copy()
    h, w = image.shape[:2]
    crop_pixels = int((crop_percent / 100.0) * h * w)
    side = int(np.sqrt(crop_pixels))
    side = min(side, min(h, w) - 1)
    attacked[:side, :side] = 0  # Zero out cropped area
    return attacked
