import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_plots(results: dict, output_dir: str):
    """
    Generates and saves publication-ready figures for Model A and Model B:
    1. Feature Importance horizontal bar chart
    2. Confusion Matrix heatmap
    3. Baseline Performance Comparison bar chart
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    model_title = results.get('model_name', 'Model').replace('_', ' ')
    
    # 1. Feature Importance Ranking
    importances = results['feature_importances']
    top_n = min(10, len(importances))
    top_features = list(importances.keys())[:top_n]
    top_scores = [importances[k] for k in top_features]
    
    plt.figure(figsize=(9, 5))
    plt.barh(top_features[::-1], top_scores[::-1], color='#2b5c8f', edgecolor='black')
    plt.xlabel('Gini Importance (Random Forest)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Predictor Features ({model_title})', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'feature_importance_top10.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # 2. Confusion Matrix Heatmap
    cm = np.array(results['confusion_matrix'])
    labels = results['labels']
    
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel('Predicted Technique', fontsize=12, fontweight='bold')
    plt.ylabel('Ground-Truth Best Technique', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix ({model_title})', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    # 3. Model vs Baselines Comparison Bar Chart
    model_names = ['RF (Ours)', 'Always LSB', 'Always DCT', 'Always DWT', 'Random', 'Heuristic Edge']
    accs = [
        results['RF']['accuracy_mean'],
        results['Baseline_Always_LSB']['accuracy_mean'],
        results['Baseline_Always_DCT']['accuracy_mean'],
        results['Baseline_Always_DWT']['accuracy_mean'],
        results['Baseline_Random']['accuracy_mean'],
        results['Baseline_Heuristic_Edge']['accuracy_mean']
    ]
    f1s = [
        results['RF']['f1_macro_mean'],
        results['Baseline_Always_LSB']['f1_macro_mean'],
        results['Baseline_Always_DCT']['f1_macro_mean'],
        results['Baseline_Always_DWT']['f1_macro_mean'],
        results['Baseline_Random']['f1_macro_mean'],
        results['Baseline_Heuristic_Edge']['f1_macro_mean']
    ]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, accs, width, label='Accuracy', color='#1f77b4', edgecolor='black')
    plt.bar(x + width/2, f1s, width, label='Macro F1-Score', color='#ff7f0e', edgecolor='black')
    
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    plt.title(f'Predictive Performance Comparison ({model_title})', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, model_names, rotation=15, ha='right', fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True, facecolor='white', framealpha=1.0)
    
    for i in range(len(model_names)):
        plt.text(x[i] - width/2, accs[i] + 0.02, f"{accs[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
        plt.text(x[i] + width/2, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'model_vs_baselines_comparison.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
