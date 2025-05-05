from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
import os

def plot(gt_data_path:str, pred_data_path:str, save_dir: str) -> None:
    """Plot the confusion matrix."""
    # Absolute Path Conversion
    save_dir = os.path.abspath(save_dir)
    gt_dir = os.path.abspath(gt_dir)
    pred_dir = os.path.abspath(pred_dir)

    ground_truth_data = pd.read_csv(gt_data_path)

    # Determine ground truth labels based on 'Object Count'
    ground_truth_data['Ground_Truth_Label'] = ground_truth_data['Object Count'].apply(lambda x: 1 if x > 0 else 0)

    report_df = pd.read_csv(pred_data_path)

    report_df['Predicted_Label'] = report_df['Object Count'].apply(lambda x: 1 if x > 0 else 0)

    merged_data = pd.merge(report_df, ground_truth_data[['Building ID', 'Ground_Truth_Label']], on='Building ID', how='left')

    y_true = merged_data['Ground_Truth_Label']
    y_pred = merged_data['Predicted_Label']

    conf_matrix = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    TN, FP, FN, TP = conf_matrix.ravel()

    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Positives (TP): {TP}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Object', 'Object'], yticklabels=['No Object', 'Object'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    # plt.show()
