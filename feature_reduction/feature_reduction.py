

import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_engineered_features(feature_dir):
    
    all_features = []
    all_subjects = []

    subject_dirs = glob.glob(os.path.join(feature_dir, 'subject_*'))
    print(f"Found {len(subject_dirs)} subjects with engineered features")
    
    for subject_dir in tqdm(subject_dirs, desc="Loading data"):
        subject_id = os.path.basename(subject_dir).replace('subject_', '')

        for feature_file in glob.glob(os.path.join(subject_dir, '*_features.csv')):
            df = pd.read_csv(feature_file)

            df['subject_id'] = subject_id

            all_features.append(df)

    if all_features:
        features_df = pd.concat(all_features, ignore_index=True)
        print(f"Combined dataset contains {len(features_df)} samples with {features_df.shape[1]} columns")
        return features_df
    else:
        print("No feature data found")
        return None

def select_features_mi(X, y, k=50):
    
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True), selector.scores_

def select_features_anova(X, y, k=50):
    
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True), selector.scores_

def run_feature_reduction(features_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features_df = load_engineered_features(features_dir)
    if features_df is None:
        return

    exclude_cols = ['epoch', 'sleep_stage', 'subject_id']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"\nSelected {len(feature_cols)} feature columns for analysis:")
    print(", ".join(feature_cols))

    nan_counts = features_df[feature_cols].isna().sum()
    print("\nNaN values in each feature:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"  {col}: {count} NaNs ({count/len(features_df)*100:.2f}%)")

    if features_df['sleep_stage'].isna().any():
        initial_count = len(features_df)
        features_df = features_df.dropna(subset=['sleep_stage'])
        print(f"Dropped {initial_count - len(features_df)} rows with NaN sleep_stage values")

    if 'prev_stage' in feature_cols and features_df['prev_stage'].isna().mean() > 0.5:
        print("Removing prev_stage feature due to excessive NaN values")
        feature_cols.remove('prev_stage')

    valid_cols = []
    for col in feature_cols:
        if features_df[col].isna().all():
            print(f"Removing {col} as it contains only NaN values")
        else:
            valid_cols.append(col)
    
    feature_cols = valid_cols

    X = features_df[feature_cols].values
    y = features_df['sleep_stage'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nSplit data into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets")

    nan_percentage = np.isnan(X_train).mean() * 100
    print(f"Training data contains {nan_percentage:.2f}% NaN values")

    print("Imputing missing values with column means...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print("Applied standard scaling to normalize features")

    k_values = [10, 20, 30, 40, 50]

    print("\nApplying Mutual Information feature selection...")
    mi_results = {}
    
    for k in k_values:

        k_actual = min(k, len(feature_cols))
        
        selected_indices, scores = select_features_mi(X_train_scaled, y_train, k=k_actual)

        selected_features = [feature_cols[i] for i in selected_indices]

        mi_results[k] = {
            'indices': selected_indices,
            'scores': scores,
            'features': selected_features
        }
        
        print(f"MI-{k}: Selected {len(selected_features)} features")

        with open(os.path.join(output_dir, f'mi_{k}_features.txt'), 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

        feature_scores = list(zip(feature_cols, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        with open(os.path.join(output_dir, f'mi_{k}_scores.txt'), 'w') as f:
            f.write("Feature,Score\n")
            for feature, score in feature_scores:
                f.write(f"{feature},{score}\n")

    print("\nApplying ANOVA F-test feature selection...")
    anova_results = {}
    
    for k in k_values:

        k_actual = min(k, len(feature_cols))
        
        selected_indices, scores = select_features_anova(X_train_scaled, y_train, k=k_actual)

        selected_features = [feature_cols[i] for i in selected_indices]

        anova_results[k] = {
            'indices': selected_indices,
            'scores': scores,
            'features': selected_features
        }
        
        print(f"ANOVA-{k}: Selected {len(selected_features)} features")

        with open(os.path.join(output_dir, f'anova_{k}_features.txt'), 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")

        feature_scores = list(zip(feature_cols, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        with open(os.path.join(output_dir, f'anova_{k}_scores.txt'), 'w') as f:
            f.write("Feature,Score\n")
            for feature, score in feature_scores:
                f.write(f"{feature},{score}\n")

    print("\nCreating datasets with selected features...")

    for k in k_values:
        selected_indices = mi_results[k]['indices']

        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        np.save(os.path.join(output_dir, f'X_train_mi_{k}.npy'), X_train_selected)
        np.save(os.path.join(output_dir, f'X_test_mi_{k}.npy'), X_test_selected)

    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    for k in k_values:
        selected_indices = anova_results[k]['indices']

        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]

        np.save(os.path.join(output_dir, f'X_train_anova_{k}.npy'), X_train_selected)
        np.save(os.path.join(output_dir, f'X_test_anova_{k}.npy'), X_test_selected)

    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(output_dir, 'imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)

    plot_feature_importance(feature_cols, mi_results, anova_results, output_dir)
    
    print(f"\nFeature reduction complete. Results saved to {output_dir}")
    print("\nTop features by each method (k=50):")
    print("\nMutual Information (MI-50):")
    for i, feature in enumerate(mi_results[50]['features'][:10]):
        print(f"  {i+1}. {feature}")
    
    print("\nANOVA F-test (ANOVA-50):")
    for i, feature in enumerate(anova_results[50]['features'][:10]):
        print(f"  {i+1}. {feature}")

def plot_feature_importance(feature_cols, mi_results, anova_results, output_dir):

    figures_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt.figure(figsize=(12, 8))

    feature_scores = list(zip(feature_cols, mi_results[50]['scores']))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    top_features = feature_scores[:min(20, len(feature_scores))]

    features = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mutual Information Score')
    plt.title('Top Features by Mutual Information')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'mi_top_features.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))

    feature_scores = list(zip(feature_cols, anova_results[50]['scores']))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    top_features = feature_scores[:min(20, len(feature_scores))]

    features = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('ANOVA F-score')
    plt.title('Top Features by ANOVA F-test')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'anova_top_features.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    
    overlap_counts = []
    for k in [10, 20, 30, 40, 50]:
        if k in mi_results and k in anova_results:
            mi_features = set(mi_results[k]['features'])
            anova_features = set(anova_results[k]['features'])
            overlap = len(mi_features.intersection(anova_features))
            overlap_counts.append(overlap)
        else:
            overlap_counts.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar([10, 20, 30, 40, 50], overlap_counts)
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Number of Common Features')
    plt.title('Overlap Between MI and ANOVA Feature Selection')
    plt.xticks([10, 20, 30, 40, 50])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_selection_overlap.png'), dpi=300)
    plt.close()

    feature_types = {
        'Raw': ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'],
        'Log': ['delta_power_log', 'theta_power_log', 'alpha_power_log', 'beta_power_log', 'gamma_power_log'],
        'Ratio': ['delta_theta_ratio', 'theta_alpha_ratio', 'slow_fast_ratio'],
        'Statistics': ['mean', 'std'],
        'Transition': ['prev_stage']
    }

    valid_types = {}
    for type_name, patterns in feature_types.items():
        for feature in feature_cols:
            if any(pattern in feature for pattern in patterns):
                if type_name not in valid_types:
                    valid_types[type_name] = []
                valid_types[type_name].append(feature)
    
    if len(valid_types) > 1:  # Only create plot if we have multiple feature types
        mi_type_counts = {type_name: 0 for type_name in valid_types}
        anova_type_counts = {type_name: 0 for type_name in valid_types}

        if 50 in mi_results:
            for feature in mi_results[50]['features']:
                for type_name, type_features in valid_types.items():
                    if feature in type_features:
                        mi_type_counts[type_name] += 1
                        break

        if 50 in anova_results:
            for feature in anova_results[50]['features']:
                for type_name, type_features in valid_types.items():
                    if feature in type_features:
                        anova_type_counts[type_name] += 1
                        break

        plt.figure(figsize=(10, 6))
        
        types = list(valid_types.keys())
        mi_counts = [mi_type_counts[t] for t in types]
        anova_counts = [anova_type_counts[t] for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        
        plt.bar(x - width/2, mi_counts, width, label='MI-50')
        plt.bar(x + width/2, anova_counts, width, label='ANOVA-50')
        
        plt.xticks(x, types)
        plt.ylabel('Number of Features')
        plt.title('Feature Types Selected by MI and ANOVA (k=50)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_types.png'), dpi=300)
        plt.close()

if __name__ == "__main__":

    features_dir = os.path.join(os.getcwd(), 'feature_engineering', 'engineered_features')
    output_dir = os.path.join(os.getcwd(), 'feature_engineering', 'reduced_features')
    
    if not os.path.exists(features_dir):
        print(f"Feature directory {features_dir} does not exist.")
        print("Please run the feature engineering script first.")
    else:
        run_feature_reduction(features_dir, output_dir)