import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

class HeuristicEdgeClassifier:
    """Baseline heuristic classifier: selects DWT for high edge density, else DCT."""
    def __init__(self, median_edge_density: float):
        self.median_edge_density = median_edge_density
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            if row.get('edge_density', 0.0) > self.median_edge_density:
                preds.append('DWT')
            else:
                preds.append('DCT')
        return np.array(preds)


def train_and_evaluate_model(
    df: pd.DataFrame, 
    feature_cols: list, 
    target_col: str = 'best_technique',
    group_col: str = 'image_id',
    model_name: str = "Model_A_Image_Only"
) -> dict:
    """
    Trains and evaluates Random Forest and 5 Baselines using cross-validation.
    - If df has 1 row per image (Model A): Uses StratifiedKFold.
    - If df has multiple rows per image (Model B): Uses GroupKFold grouped strictly by group_col (image_id).
    
    Guarantees ZERO image overlap between Train and Test splits!
    """
    X = df[feature_cols].copy()
    y = df[target_col].values
    groups = df[group_col].values if group_col in df.columns else np.arange(len(df))
    
    # Select cross-validation strategy
    unique_groups = len(np.unique(groups))
    if unique_groups == len(df):
        # 1 row per image (Model A) -> StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = cv.split(X, y)
    else:
        # Multiple rows per image (Model B) -> GroupKFold grouped by image_id
        cv = GroupKFold(n_splits=5)
        split_iter = cv.split(X, y, groups=groups)

    rf_accs, rf_f1s = [], []
    lsb_accs, lsb_f1s = [], []
    dct_accs, dct_f1s = [], []
    dwt_accs, dwt_f1s = [], []
    rnd_accs, rnd_f1s = [], []
    heur_accs, heur_f1s = [], []
    
    all_y_true = []
    all_rf_preds = []
    
    median_edge = float(X['edge_density'].median()) if 'edge_density' in X.columns else 0.05
    
    for fold, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        seed = 42 + fold
        
        # Apply SMOTE oversampling to create equal synthetic training samples for DCT, LSB, and DWT
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=seed, k_neighbors=2)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except Exception:
            X_train_res, y_train_res = X_train, y_train

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest Classifier with balanced sample weighting
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=seed,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train_res)
        rf_preds = rf.predict(X_test_scaled)
        
        all_y_true.extend(y_test)
        all_rf_preds.extend(rf_preds)
        
        rf_accs.append(accuracy_score(y_test, rf_preds))
        rf_f1s.append(f1_score(y_test, rf_preds, average='macro', zero_division=0))
        
        # 2. Baseline: Always LSB
        lsb_preds = np.full_like(y_test, 'LSB')
        lsb_accs.append(accuracy_score(y_test, lsb_preds))
        lsb_f1s.append(f1_score(y_test, lsb_preds, average='macro', zero_division=0))
        
        # 3. Baseline: Always DCT
        dct_preds = np.full_like(y_test, 'DCT')
        dct_accs.append(accuracy_score(y_test, dct_preds))
        dct_f1s.append(f1_score(y_test, dct_preds, average='macro', zero_division=0))
        
        # 4. Baseline: Always DWT
        dwt_preds = np.full_like(y_test, 'DWT')
        dwt_accs.append(accuracy_score(y_test, dwt_preds))
        dwt_f1s.append(f1_score(y_test, dwt_preds, average='macro', zero_division=0))
        
        # 5. Baseline: Random Uniform
        unique_classes = np.unique(y)
        rng = np.random.RandomState(42 + fold)
        rnd_preds = rng.choice(unique_classes, size=len(y_test))
        rnd_accs.append(accuracy_score(y_test, rnd_preds))
        rnd_f1s.append(f1_score(y_test, rnd_preds, average='macro', zero_division=0))
        
        # 6. Baseline: Heuristic Edge Classifier
        heur = HeuristicEdgeClassifier(median_edge_density=median_edge)
        heur_preds = heur.predict(X_test)
        heur_accs.append(accuracy_score(y_test, heur_preds))
        heur_f1s.append(f1_score(y_test, heur_preds, average='macro', zero_division=0))

    labels = sorted(list(np.unique(all_y_true)))
    cm = confusion_matrix(all_y_true, all_rf_preds, labels=labels)
    cls_report = classification_report(all_y_true, all_rf_preds, labels=labels, output_dict=True, zero_division=0)
    
    # Feature Importance & Trained Model Export
    import joblib
    import os
    
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    
    last_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    last_rf.fit(X_scaled, y)
    
    model_save_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump({'model': last_rf, 'feature_cols': feature_cols, 'scaler': final_scaler}, model_save_path)
    
    importances = dict(zip(feature_cols, last_rf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    results = {
        'model_name': model_name,
        'model_saved_path': model_save_path,
        'RF': {'accuracy_mean': float(np.mean(rf_accs)), 'f1_macro_mean': float(np.mean(rf_f1s))},
        'Baseline_Always_LSB': {'accuracy_mean': float(np.mean(lsb_accs)), 'f1_macro_mean': float(np.mean(lsb_f1s))},
        'Baseline_Always_DCT': {'accuracy_mean': float(np.mean(dct_accs)), 'f1_macro_mean': float(np.mean(dct_f1s))},
        'Baseline_Always_DWT': {'accuracy_mean': float(np.mean(dwt_accs)), 'f1_macro_mean': float(np.mean(dwt_f1s))},
        'Baseline_Random': {'accuracy_mean': float(np.mean(rnd_accs)), 'f1_macro_mean': float(np.mean(rnd_f1s))},
        'Baseline_Heuristic_Edge': {'accuracy_mean': float(np.mean(heur_accs)), 'f1_macro_mean': float(np.mean(heur_f1s))},
        'classification_report': cls_report,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'feature_importances': importances
    }
    
    return results
