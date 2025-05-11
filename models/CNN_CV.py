

import os
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LeakyReLU, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
import time

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth set for GPU: {device}")
else:
    print("No GPU devices found, using CPU")

def create_sleep_1d_cnn(input_shape, num_classes=5):

    num_features = input_shape[0]

    if num_features <= 16:
        model = Sequential()

        model.add(Conv1D(filters=32, kernel_size=1, strides=1, padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv1D(filters=64, kernel_size=1, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        model.add(GlobalAveragePooling1D())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
    else:

        model = Sequential()

        model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.01))

        model.add(GlobalAveragePooling1D())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def reshape_data_for_cnn(X):
    
    return X.reshape(X.shape[0], X.shape[1], 1)

def analyze_feature_importance(model, X_test, y_test, feature_names):
    
    print("\nAnalyzing feature importance...")
    X_test_reshaped = reshape_data_for_cnn(X_test)
    baseline_acc = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]

    importance_scores = []

    for i in range(X_test.shape[1]):
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"

        X_permuted = X_test.copy()

        np.random.shuffle(X_permuted[:, i])

        X_permuted_reshaped = reshape_data_for_cnn(X_permuted)
        permuted_acc = model.evaluate(X_permuted_reshaped, y_test, verbose=0)[1]

        importance = baseline_acc - permuted_acc
        importance_scores.append((feature_name, importance))
        
        print(f"  Testing feature: {feature_name:<20} | Importance: {importance:.4f}")

    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance Ranking:")
    print("Feature | Importance Score (higher = more important)")
    print("-" * 50)
    for feature, score in importance_scores:
        print(f"{feature:<30} | {score:.4f}")
    
    return importance_scores

def get_feature_names(feature_method, k_value, reduced_features_dir):

    feature_file = os.path.join(reduced_features_dir, f'{feature_method}_{k_value}_features.txt')
    if os.path.exists(feature_file):
        with open(feature_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    try:
        pickle_file = os.path.join(reduced_features_dir, 'feature_names.pkl')
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                all_features = pickle.load(f)
                return all_features[:k_value]
    except:
        pass

    return [f"Feature_{i}" for i in range(k_value)]

def run_cross_validation(feature_method, k_value, model_dir, reduced_features_dir, n_folds=5):
    
    start_time = time.time()

    X_train_orig = np.load(os.path.join(reduced_features_dir, f'X_train_{feature_method}_{k_value}.npy'))
    X_test_orig = np.load(os.path.join(reduced_features_dir, f'X_test_{feature_method}_{k_value}.npy'))
    y_train_orig = np.load(os.path.join(reduced_features_dir, 'y_train.npy'))
    y_test_orig = np.load(os.path.join(reduced_features_dir, 'y_test.npy'))

    feature_names = get_feature_names(feature_method, k_value, reduced_features_dir)

    X_all = np.vstack((X_train_orig, X_test_orig))
    y_all = np.concatenate((y_train_orig, y_test_orig))

    min_label = np.min(y_all)
    if min_label > 0:
        y_all = y_all - min_label
        print(f"Labels adjusted by subtracting {min_label}")
    
    print(f"\nRunning {n_folds}-fold Cross-Validation for {feature_method.upper()}-{k_value}")
    print(f"Total samples: {len(X_all)}")
    print(f"Class distribution: {np.bincount(y_all)}")

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = {
        'accuracy': [],
        'f1_score': [],
        'training_time': [],
        'fold_importance_scores': []
    }


    np.random.seed(42)
    y_shuffled = y_all.copy()
    np.random.shuffle(y_shuffled)
    
    shuffled_results = {
        'accuracy': [],
        'f1_score': []
    }

    print("\n=== Cross-Validation with True Labels ===")
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X_all, y_all)):
        print(f"\nFold {fold_idx+1}/{n_folds}")
        
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]

        X_train_reshaped = reshape_data_for_cnn(X_train)
        X_test_reshaped = reshape_data_for_cnn(X_test)

        fold_start_time = time.time()
        model = create_sleep_1d_cnn(
            input_shape=(X_train_reshaped.shape[1], 1), 
            num_classes=len(np.unique(y_all))
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
        ]

        model.fit(
            X_train_reshaped, y_train,
            epochs=30,  # Reduced for CV
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        fold_training_time = time.time() - fold_start_time
        fold_results['training_time'].append(fold_training_time)

        y_pred = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        fold_results['accuracy'].append(accuracy)
        fold_results['f1_score'].append(f1)
        
        print(f"Fold {fold_idx+1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Training Time: {fold_training_time:.2f} seconds")

        importance_scores = analyze_feature_importance(model, X_test, y_test, feature_names)
        fold_results['fold_importance_scores'].append(importance_scores)

    print("\n=== Cross-Validation with Shuffled Labels (Chance Level Check) ===")
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X_all, y_shuffled)):
        print(f"\nShuffled Fold {fold_idx+1}/{n_folds}")
        
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

        X_train_reshaped = reshape_data_for_cnn(X_train)
        X_test_reshaped = reshape_data_for_cnn(X_test)

        model = create_sleep_1d_cnn(
            input_shape=(X_train_reshaped.shape[1], 1), 
            num_classes=len(np.unique(y_all))
        )

        model.fit(
            X_train_reshaped, y_train,
            epochs=20,  # Fewer epochs for shuffled data
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )

        y_pred = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        
        shuffled_results['accuracy'].append(accuracy)
        shuffled_results['f1_score'].append(f1)
        
        print(f"Shuffled Fold {fold_idx+1} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")

    all_features = {}

    for fold_scores in fold_results['fold_importance_scores']:
        for feature, score in fold_scores:
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(score)

    feature_importance = []
    for feature, scores in all_features.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        feature_importance.append((feature, mean_score, std_score))

    feature_importance.sort(key=lambda x: x[1], reverse=True)

    total_time = time.time() - start_time
    
    cv_results = {
        'feature_method': feature_method,
        'k_value': k_value,
        'n_folds': n_folds,
        'accuracy': {
            'mean': np.mean(fold_results['accuracy']),
            'std': np.std(fold_results['accuracy']),
            'values': fold_results['accuracy']
        },
        'f1_score': {
            'mean': np.mean(fold_results['f1_score']),
            'std': np.std(fold_results['f1_score']),
            'values': fold_results['f1_score']
        },
        'training_time': {
            'mean': np.mean(fold_results['training_time']),
            'total': total_time
        },
        'feature_importance': feature_importance,
        'shuffled_results': {
            'accuracy': {
                'mean': np.mean(shuffled_results['accuracy']),
                'std': np.std(shuffled_results['accuracy']),
                'values': shuffled_results['accuracy']
            },
            'f1_score': {
                'mean': np.mean(shuffled_results['f1_score']),
                'std': np.std(shuffled_results['f1_score']),
                'values': shuffled_results['f1_score']
            }
        }
    }

    print("\n=== Cross-Validation Summary ===")
    print(f"Model: CNN with {feature_method.upper()}-{k_value}")
    print(f"Number of folds: {n_folds}")
    print(f"\nTrue Labels Performance:")
    print(f"  Mean Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    print(f"  Mean F1 Score: {cv_results['f1_score']['mean']:.4f} ± {cv_results['f1_score']['std']:.4f}")
    print(f"  Mean Training Time per Fold: {cv_results['training_time']['mean']:.2f} seconds")
    print(f"  Total Time: {cv_results['training_time']['total']:.2f} seconds")
    
    print(f"\nShuffled Labels Performance (Chance Level):")
    print(f"  Mean Accuracy: {cv_results['shuffled_results']['accuracy']['mean']:.4f} ± {cv_results['shuffled_results']['accuracy']['std']:.4f}")
    print(f"  Mean F1 Score: {cv_results['shuffled_results']['f1_score']['mean']:.4f} ± {cv_results['shuffled_results']['f1_score']['std']:.4f}")

    real_acc = np.array(fold_results['accuracy'])
    shuffled_acc = np.array(shuffled_results['accuracy'])
    acc_diff = real_acc - shuffled_acc

    t_stat = np.mean(acc_diff) / (np.std(acc_diff, ddof=1) / np.sqrt(n_folds))
    
    print(f"\nStatistical Analysis:")
    print(f"  Mean accuracy difference (real - shuffled): {np.mean(acc_diff):.4f}")
    print(f"  Approximate t-statistic: {t_stat:.4f}")
    
    if t_stat > 2.0:  # Rule of thumb for significance at p < 0.05 for small samples
        print("  ✓ Results are statistically significant compared to chance level")
    else:
        print("  ✗ Results are NOT statistically significant compared to chance level")

    print("\nTop 10 Most Important Features (Consistent Across Folds):")
    for i, (feature, mean_score, std_score) in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature:<20} | Importance: {mean_score:.4f} ± {std_score:.4f}")

    cv_results_file = os.path.join(model_dir, f'cv_results_{feature_method}_{k_value}.pkl')
    with open(cv_results_file, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f"\nCV results saved to {cv_results_file}")
    
    return cv_results

def train_final_model(feature_method, k_value, model_dir, reduced_features_dir):

    X_train = np.load(os.path.join(reduced_features_dir, f'X_train_{feature_method}_{k_value}.npy'))
    X_test = np.load(os.path.join(reduced_features_dir, f'X_test_{feature_method}_{k_value}.npy'))
    y_train = np.load(os.path.join(reduced_features_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(reduced_features_dir, 'y_test.npy'))

    feature_names = get_feature_names(feature_method, k_value, reduced_features_dir)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    print(f"Unique values in y_train: {unique_train}")
    print(f"Unique values in y_test: {unique_test}")

    min_label = min(np.min(y_train), np.min(y_test))
    if min_label > 0:
        y_train = y_train - min_label
        y_test = y_test - min_label
        print(f"Labels adjusted by subtracting {min_label}")

    X_train_reshaped = reshape_data_for_cnn(X_train)
    X_test_reshaped = reshape_data_for_cnn(X_test)
    
    print(f"\nTraining Final Sleep-1D-CNN with {feature_method.upper()}-{k_value} features")
    print(f"Training data shape: {X_train_reshaped.shape}")
    print(f"Testing data shape: {X_test_reshaped.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    model = create_sleep_1d_cnn(
        input_shape=(X_train_reshaped.shape[1], 1), 
        num_classes=len(np.unique(y_train))
    )

    model.summary()

    model_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"Sleep_1D_CNN_Final_{feature_method}_{k_value}_{model_timestamp}"
    model_path = os.path.join(model_dir, model_name + '.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

    y_pred = model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    importance_scores = analyze_feature_importance(model, X_test, y_test, feature_names)

    results = {
        'feature_method': feature_method,
        'k_value': k_value,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'history': history.history,
        'model_path': model_path,
        'feature_importance': importance_scores
    }

    model.save(model_path)

    results_file = os.path.join(model_dir, f'final_results_{feature_method}_{k_value}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nFinal Results for {feature_method.upper()}-{k_value}:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nTop 10 Most Important Features:")
    for i, (feature, score) in enumerate(importance_scores[:10]):
        print(f"{i+1}. {feature:<20} | Importance: {score:.4f}")
    
    return results

def run_experiments(model_dir, reduced_features_dir):
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cv_results = {}
    final_results = {}

    priority_experiments = [('mi', 50), ('anova', 20)]

    for feature_method, k_value in priority_experiments:
        if os.path.exists(os.path.join(reduced_features_dir, f'X_train_{feature_method}_{k_value}.npy')):
            result_key = f"{feature_method}_{k_value}"
            print(f"\n=== CROSS-VALIDATION FOR {feature_method.upper()}-{k_value} ===")
            cv_results[result_key] = run_cross_validation(
                feature_method, k_value, model_dir, reduced_features_dir, n_folds=5
            )

            print(f"\n=== TRAINING FINAL MODEL FOR {feature_method.upper()}-{k_value} ===")
            final_results[result_key] = train_final_model(
                feature_method, k_value, model_dir, reduced_features_dir
            )
        else:
            print(f"Warning: Data files for {feature_method.upper()}-{k_value} not found at {reduced_features_dir}")

    if len(cv_results) > 1:
        print("\n===== COMPARISON OF CROSS-VALIDATION RESULTS =====")
        print("Feature Method | K Value | CV Accuracy | CV F1-Score | Shuffled Accuracy | Final Accuracy")
        print("-" * 95)
        
        for result_key in cv_results:
            feature_method, k_value = result_key.split('_')
            cv_acc_mean = cv_results[result_key]['accuracy']['mean']
            cv_acc_std = cv_results[result_key]['accuracy']['std']
            cv_f1_mean = cv_results[result_key]['f1_score']['mean']
            cv_f1_std = cv_results[result_key]['f1_score']['std']
            
            shuffled_acc_mean = cv_results[result_key]['shuffled_results']['accuracy']['mean']
            final_acc = final_results[result_key]['test_accuracy']
            
            print(f"{feature_method.upper():<13} | {k_value:<7} | {cv_acc_mean:.4f}±{cv_acc_std:.4f} | "
                  f"{cv_f1_mean:.4f}±{cv_f1_std:.4f} | {shuffled_acc_mean:.4f} | {final_acc:.4f}")

    summary = {
        'cv_results': cv_results,
        'final_results': final_results
    }
    
    summary_file = os.path.join(model_dir, 'summary_with_cv.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nAll results saved to {summary_file}")
    
    return summary

if __name__ == "__main__":

    base_dir = os.getcwd()
    reduced_features_dir = os.path.join(base_dir, 'feature_engineering', 'reduced_features')
    model_dir = os.path.join(base_dir, 'models', 'sleep_1d_cnn_with_cv')
    
    print("=== SLEEP-1D-CNN WITH CROSS-VALIDATION ===")
    print("This script runs models with cross-validation to verify robustness\n")
    
    if not os.path.exists(reduced_features_dir):
        print(f"Reduced features directory {reduced_features_dir} does not exist.")
        print("Please run the feature reduction script first.")
    else:

        all_results = run_experiments(model_dir, reduced_features_dir)