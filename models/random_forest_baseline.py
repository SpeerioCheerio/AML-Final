

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.getcwd()
reduced_features_dir = os.path.join(base_dir, 'feature_engineering', 'reduced_features')

experiments = [('mi', 50), ('anova', 20)]

print("=== SIMPLE RANDOM FOREST BASELINE ===")

for feature_method, k_value in experiments:

    X_train = np.load(os.path.join(reduced_features_dir, f'X_train_{feature_method}_{k_value}.npy'))
    X_test = np.load(os.path.join(reduced_features_dir, f'X_test_{feature_method}_{k_value}.npy'))
    y_train = np.load(os.path.join(reduced_features_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(reduced_features_dir, 'y_test.npy'))

    min_label = min(np.min(y_train), np.min(y_test))
    if min_label > 0:
        y_train = y_train - min_label
        y_test = y_test - min_label

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest with {feature_method.upper()}-{k_value}:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))