import os
import glob
import time
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count

from src.feature_extraction import extract_image_features
from src.stego_engine import (
    generate_random_bits, embed_lsb, extract_lsb,
    embed_dct, extract_dct, embed_dwt, extract_dwt,
    simulate_jpeg_attack
)
from src.evaluator import evaluate_metrics
from src.utility_labeler import compute_image_level_utility, test_alpha_stability
from src.train_predict import train_and_evaluate_model
from src.visualize_results import save_plots

def _process_single_image_features(image_path: str) -> dict:
    img_id = os.path.basename(image_path)
    feats = extract_image_features(image_path)
    feats['image_id'] = img_id
    return feats

def _process_single_image_robustness(args) -> list:
    path, payload_bits_list, jpeg_qf_list = args
    img_id = os.path.basename(path)
    cover = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        return []
    h, w = cover.shape
    rows = []
    
    for payload_bits in payload_bits_list:
        payload_bpp = payload_bits / (h * w)
        payload = generate_random_bits(payload_bits, seed=42)
        
        try:
            st_lsb = embed_lsb(cover, payload)
        except Exception: st_lsb = None
            
        try:
            st_dct = embed_dct(cover, payload, delta=20.0)
        except Exception: st_dct = None
            
        try:
            st_dwt = embed_dwt(cover, payload, alpha=15.0)
        except Exception: st_dwt = None
            
        technique_stegos = {
            'LSB': (st_lsb, extract_lsb),
            'DCT': (st_dct, extract_dct),
            'DWT': (st_dwt, extract_dwt)
        }
        
        for qf in jpeg_qf_list:
            for tech_name, (stego_img, extract_fn) in technique_stegos.items():
                if stego_img is None:
                    continue
                attacked_img = simulate_jpeg_attack(stego_img, quality_factor=qf)
                extracted_payload = extract_fn(attacked_img, payload_bits)
                m = evaluate_metrics(cover, attacked_img, payload, extracted_payload)
                
                rows.append({
                    'image_id': img_id,
                    'payload_bits': payload_bits,
                    'payload_bpp': payload_bpp,
                    'target_qf': qf,
                    'technique': tech_name,
                    'psnr': m['psnr'],
                    'ssim': m['ssim'],
                    'ber': m['ber'],
                    'accuracy': m['accuracy']
                })
    return rows


def run_experiment_pipeline(
    image_dir: str, 
    payload_bits_list: list = [128, 256, 512, 1024, 2048], 
    jpeg_qf_list: list = [15, 30, 45, 60, 75, 85, 95],
    max_images: int = 10000
):
    print("=" * 80)
    print(" COMPREHENSIVE ADAPTIVE STEGANOGRAPHY METHODOLOGY BENCHMARK ")
    print("=" * 80)
    
    data_cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_cache_dir, exist_ok=True)
    
    features_csv = os.path.join(data_cache_dir, f"features_{max_images}.csv")
    robustness_csv = os.path.join(data_cache_dir, f"robustness_{max_images}.csv")
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.pgm")) + glob.glob(os.path.join(image_dir, "*.png")))[:max_images]
    print(f"[*] Target dataset size: {len(image_paths)} cover images.")
    
    # -------------------------------------------------------------
    # Step 1: Intrinsic Cover-Image Feature Extraction (Cached)
    # -------------------------------------------------------------
    if os.path.exists(features_csv):
        print(f"[1/5] Loading pre-extracted features from {features_csv}...")
        features_df = pd.read_csv(features_csv)
    else:
        print(f"[1/5] Extracting intrinsic features from {len(image_paths)} images in parallel...")
        t0 = time.time()
        with Pool(processes=min(cpu_count(), 8)) as pool:
            feats_list = pool.map(_process_single_image_features, image_paths)
        features_df = pd.DataFrame(feats_list)
        features_df.to_csv(features_csv, index=False)
        print(f"    Extracted {features_df.shape[1]-1} features for {len(features_df)} images in {time.time()-t0:.2f}s. Saved to CSV.")

    # -------------------------------------------------------------
    # Step 2: Robustness Experiment (1,050,000 trials for 10k images, Cached)
    # -------------------------------------------------------------
    total_expected_trials = len(image_paths) * len(payload_bits_list) * len(jpeg_qf_list) * 3
    if os.path.exists(robustness_csv):
        print(f"[2/5] Loading pre-evaluated robustness measurements from {robustness_csv}...")
        robustness_df = pd.read_csv(robustness_csv)
    else:
        print(f"[2/5] Running {total_expected_trials:,} robustness trials in parallel...")
        t0 = time.time()
        tasks = [(p, payload_bits_list, jpeg_qf_list) for p in image_paths]
        with Pool(processes=min(cpu_count(), 8)) as pool:
            nested_results = pool.map(_process_single_image_robustness, tasks)
        
        flat_results = [item for sublist in nested_results for item in sublist]
        robustness_df = pd.DataFrame(flat_results)
        robustness_df.to_csv(robustness_csv, index=False)
        print(f"    Evaluated {len(robustness_df)} robustness rows in {time.time()-t0:.2f}s. Saved to CSV.")

    print(f"    Total Robustness Observations: {len(robustness_df)}")

    # -------------------------------------------------------------
    # Step 3: Sensitivity Analysis across Alpha Weights
    # -------------------------------------------------------------
    print("\n[3/5] Testing Target Label Stability across Alpha Weightings (0.9, 0.8, 0.7, 0.6, 0.5)...")
    stability_res = test_alpha_stability(robustness_df, alpha_list=[0.9, 0.8, 0.7, 0.6, 0.5])
    for alpha_key, info in stability_res.items():
        print(f"  Alpha={info['alpha']:.1f} -> Target Distribution: {info['distribution']} | Agreement with Alpha=0.7: {info['agreement_with_alpha_0.7']*100:.1f}%")

    # Select Alpha = 0.5 as balanced default (50% Visual Quality / 50% Robustness Balance)
    print("\n  Using Justified Default Alpha = 0.5 (50% Quality / 50% Robustness Balance).")
    image_target_df = compute_image_level_utility(robustness_df, alpha=0.5, features_df=features_df)
    
    print("  Final Image-Level Target Distribution (1 row per image):")
    print(image_target_df['best_technique'].value_counts())

    # -------------------------------------------------------------
    # Step 4: Construct Clean ML Datasets (NO DATA LEAKAGE)
    # -------------------------------------------------------------
    # 18 Core Intrinsic Image Features
    intrinsic_feature_cols = [
        'mean_intensity', 'std_dev', 'variance', 'entropy', 'edge_density',
        'dwt_ll_energy', 'dwt_lh_energy', 'dwt_hl_energy', 'dwt_hh_energy',
        'dwt_ll_energy_ratio', 'dwt_lh_energy_ratio', 'dwt_hl_energy_ratio', 'dwt_hh_energy_ratio',
        'dct_total_energy', 'dct_low_freq_energy', 'dct_high_freq_energy',
        'dct_low_freq_energy_ratio', 'dct_high_freq_energy_ratio'
    ]
    # Filter to features actually present
    intrinsic_feature_cols = [c for c in intrinsic_feature_cols if c in features_df.columns]

    # MODEL A DATASET: Exactly 1 row per image (10,000 rows total)
    model_a_df = pd.merge(image_target_df[['image_id', 'best_technique']], features_df, on='image_id')
    print(f"\n[4/5] Model A Dataset Shape: {model_a_df.shape} (1 row per image, {len(intrinsic_feature_cols)} intrinsic features)")

    # MODEL B DATASET: Condition-Specific (17,500 rows), split strictly by image_id
    # Calculate condition-specific best technique per (image_id, payload_bits, target_qf)
    cond_group = ['image_id', 'payload_bits', 'payload_bpp', 'target_qf']
    g_rob = robustness_df.groupby(['image_id', 'payload_bits', 'target_qf'])
    p_min, p_max = g_rob['psnr'].transform('min'), g_rob['psnr'].transform('max')
    a_min, a_max = g_rob['accuracy'].transform('min'), g_rob['accuracy'].transform('max')
    
    p_norm = np.where((p_max - p_min) > 1e-6, (robustness_df['psnr'] - p_min) / (p_max - p_min + 1e-12), 1.0)
    a_norm = np.where((a_max - a_min) > 1e-6, (robustness_df['accuracy'] - a_min) / (a_max - a_min + 1e-12), 1.0)
    
    robustness_df['cond_utility'] = 0.5 * p_norm + 0.5 * a_norm
    idx_cond_best = robustness_df.groupby(['image_id', 'payload_bits', 'target_qf'])['cond_utility'].idxmax()
    cond_best_df = robustness_df.loc[idx_cond_best][cond_group + ['technique']].rename(columns={'technique': 'best_technique'})
    
    model_b_df = pd.merge(cond_best_df, features_df, on='image_id')
    model_b_features = intrinsic_feature_cols + ['payload_bpp', 'target_qf']
    print(f"      Model B Dataset Shape: {model_b_df.shape} (Condition-Specific, {len(model_b_features)} features)")

    # Save final clean ML dataset to disk as requested
    clean_ml_dataset_path = os.path.join(data_cache_dir, "final_clean_image_ml_dataset.csv")
    model_a_df[['image_id'] + intrinsic_feature_cols + ['best_technique']].to_csv(clean_ml_dataset_path, index=False)
    print(f"      Saved final clean image-level ML dataset to {clean_ml_dataset_path}")

    # -------------------------------------------------------------
    # Step 5: Machine Learning Evaluations & Baseline Comparisons
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" MODEL A — INTRINSIC COVER-IMAGE ONLY ADAPTIVE MODEL (1 Row Per Image) ")
    print("=" * 80)
    res_a = train_and_evaluate_model(
        model_a_df, intrinsic_feature_cols, target_col='best_technique', group_col='image_id', model_name="Model_A_Image_Only"
    )
    
    print(f"  Random Forest Classifier Accuracy : {res_a['RF']['accuracy_mean']:.4f} | Macro F1: {res_a['RF']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always LSB) Accuracy    : {res_a['Baseline_Always_LSB']['accuracy_mean']:.4f} | Macro F1: {res_a['Baseline_Always_LSB']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always DCT) Accuracy    : {res_a['Baseline_Always_DCT']['accuracy_mean']:.4f} | Macro F1: {res_a['Baseline_Always_DCT']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always DWT) Accuracy    : {res_a['Baseline_Always_DWT']['accuracy_mean']:.4f} | Macro F1: {res_a['Baseline_Always_DWT']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Random Uniform) Acc     : {res_a['Baseline_Random']['accuracy_mean']:.4f} | Macro F1: {res_a['Baseline_Random']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Heuristic Edge) Acc     : {res_a['Baseline_Heuristic_Edge']['accuracy_mean']:.4f} | Macro F1: {res_a['Baseline_Heuristic_Edge']['f1_macro_mean']:.4f}")
    
    print("\n  Classification Report (Model A):")
    rep_a = res_a['classification_report']
    for cls in res_a['labels']:
        if cls in rep_a:
            print(f"    Class {cls:5s} -> Precision: {rep_a[cls]['precision']:.4f} | Recall: {rep_a[cls]['recall']:.4f} | F1-Score: {rep_a[cls]['f1-score']:.4f}")
            
    print("\n  Top 5 Predictor Features (Model A Intrinsic Image Features):")
    for feat, imp in list(res_a['feature_importances'].items())[:5]:
        print(f"    - {feat:30s}: {imp:.4f}")
        
    save_plots(res_a, os.path.join(os.path.dirname(__file__), "results", "plots", "Model_A_Image_Only"))

    print("\n" + "=" * 80)
    print(" MODEL B — IMAGE + OPERATING-CONDITION MODEL (Condition-Specific, GroupKFold) ")
    print("=" * 80)
    res_b = train_and_evaluate_model(
        model_b_df, model_b_features, target_col='best_technique', group_col='image_id', model_name="Model_B_Image_Plus_Condition"
    )
    
    print(f"  Random Forest Classifier Accuracy : {res_b['RF']['accuracy_mean']:.4f} | Macro F1: {res_b['RF']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always LSB) Accuracy    : {res_b['Baseline_Always_LSB']['accuracy_mean']:.4f} | Macro F1: {res_b['Baseline_Always_LSB']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always DCT) Accuracy    : {res_b['Baseline_Always_DCT']['accuracy_mean']:.4f} | Macro F1: {res_b['Baseline_Always_DCT']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Always DWT) Accuracy    : {res_b['Baseline_Always_DWT']['accuracy_mean']:.4f} | Macro F1: {res_b['Baseline_Always_DWT']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Random Uniform) Acc     : {res_b['Baseline_Random']['accuracy_mean']:.4f} | Macro F1: {res_b['Baseline_Random']['f1_macro_mean']:.4f}")
    print(f"  Baseline (Heuristic Edge) Acc     : {res_b['Baseline_Heuristic_Edge']['accuracy_mean']:.4f} | Macro F1: {res_b['Baseline_Heuristic_Edge']['f1_macro_mean']:.4f}")
    
    save_plots(res_b, os.path.join(os.path.dirname(__file__), "results", "plots", "Model_B_Image_Plus_Condition"))
    
    print("=" * 80)
    return {'Model_A': res_a, 'Model_B': res_b}


if __name__ == '__main__':
    data_dir = "/Users/kaniishkselvakumar/Downloads/BOSSbase_1.01"
    run_experiment_pipeline(data_dir, max_images=10000)
