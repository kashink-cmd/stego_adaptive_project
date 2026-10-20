# Content-Aware Adaptive Image Steganography via Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)

An intelligent, content-aware steganographic framework that automatically analyzes cover image texture characteristics and predicts the optimal embedding domain (**Spatial LSB**, **8×8 Block DCT**, or **2D Haar DWT**) to maximize imperceptibility and channel robustness.

---

## 🌟 Key Features

- **18 Intrinsic Statistical & Texture Features**:
  - First-order statistics: Variance, Shannon Entropy, Kurtosis, Skewness.
  - Structural features: Sobel Edge Density, Gradient Mean/Std, Harris Corner Density.
  - Spatial Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation.
  - Multi-resolution Wavelet Subbands: Energy distributions across LL, LH, HL, HH, and Fourier Spectral Energy.
- **Three Embedding Domains**:
  - **Spatial LSB**: Pixel bit-plane replacement for pristine visual fidelity ($>65\text{ dB PSNR}$) on smooth images.
  - **Block DCT (Discrete Cosine Transform)**: 8×8 block frequency transform with Quantization Index Modulation (QIM) on AC mid-frequencies (survives JPEG compression and sensor noise).
  - **Wavelet DWT (Discrete Wavelet Transform)**: Multi-subband 2D Haar wavelet modulation in LH/HL/HH detail coefficients (superior resistance to spatial resampling and edge filtering).
- **Simulated Real-World Channel Attacks**:
  1. 🗜️ **JPEG Compression Attack** ($\text{QF} = 15 \text{ to } 95$).
  2. ⚡ **Additive Noise Attack** (Gaussian Sensor Noise & Salt-and-Pepper Impulse Noise).
  3. 📐 **Resizing Attack** (Downsampling and cubic interpolation back to full size, simulating social media platforms like WhatsApp/Telegram).
  4. 🧹 **Filtering / Blurring Attack** (Non-linear Median filter anti-steganography scrubbers and Gaussian blur).
  5. ✂️ **Cropping Attack** (Localized border/region masking simulating packet loss).
- **Trained on 10,000 BOSSBase Images**:
  - Uses a single unified Random Forest classifier trained on the standard academic BOSSBase 1.01 benchmark dataset.
  - Objective Utility Function:
    $$\text{Utility} = 0.5 \times \text{Normalized\_Clean\_PSNR} + 0.5 \times \text{Post\_Attack\_Accuracy}$$
- **Full Interactive Web Dashboard**:
  - Real-time prediction and feature importance breakdown.
  - Simultaneous embedding into LSB, DCT, and DWT with real-time PSNR, MSE, SSIM, and BER calculations.
  - Comprehensive 5-attack performance comparison matrix.
  - Interactive attack tabs for instant visual verification of decoded secret messages.
  - Residual difference heatmaps ($|\text{Stego} - \text{Cover}| \times 30$) and carrier pixel coordinate maps.

---

## 📁 Repository Structure

```
├── app.py                                # Full interactive Streamlit dashboard
├── main.py                               # CLI execution pipeline
├── predict_stego_technique.py            # Quick CLI prediction utility
├── requirements.txt                      # Project dependencies
├── src/
│   ├── feature_extraction.py             # 18 texture & statistical feature extractors
│   ├── stego_engine.py                   # LSB, DCT, DWT embedding/extraction & 5 channel attacks
│   ├── evaluator.py                      # PSNR, MSE, SSIM, BER, and Extraction Accuracy metrics
│   ├── train_predict.py                  # Random Forest training and inference pipeline
│   └── dataset_pipeline.py               # Batch feature extraction and labeling
├── data/
│   ├── sample_covers/                    # Benchmark sample test cover images
│   ├── bossbase_image_labels_10k.csv     # Optimal technique ground-truth labels
│   ├── features_10000.csv                # Extracted 18-D feature matrix for 10k images
│   └── final_clean_image_ml_dataset.csv  # Final cleaned training dataset
└── results/
    ├── models/
    │   └── Adaptive_Stego_Classifier.joblib # Trained production Random Forest model
    └── plots/                            # Performance evaluation and confusion matrix plots
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📊 Evaluation & Empirical Attack Survival

| Attack Scenario | Spatial LSB | Block DCT | Wavelet DWT | Optimal Domain |
| :--- | :---: | :---: | :---: | :---: |
| **Clean Channel (Pre-Attack)** | **>65 dB** (Pristine) | ~46 dB | ~48 dB | Spatial LSB |
| **JPEG Compression (QF=75)** | ❌ 47.6% (Fails) | **100.0%** (Survives) | **100.0%** (Survives) | DCT / DWT |
| **Gaussian Noise (10%)** | ❌ 53.7% (Fails) | **100.0%** (Survives) | **100.0%** (Survives) | DCT / DWT |
| **Resizing Attack (75% Scale)** | 🟡 82.8% (Degraded) | 🟢 97.6% (Resilient) | **99.0%** (Top Resilient) | Wavelet DWT |
| **Median Filter (3×3 Scrubber)**| ❌ 56.1% (Fails) | **96.3%** (Survives) | 🟢 94.6% (Resilient) | Block DCT |
| **Cropping Attack (10% Loss)** | 🟡 77.7% (Partial) | **86.8%** (Robust) | 🟢 83.1% (Robust) | DCT / DWT |

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
