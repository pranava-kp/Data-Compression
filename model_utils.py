# model_utils.py
import joblib
import pandas as pd
import os
import math
import numpy as np
from collections import Counter

def load_models():
    dt_bundle = joblib.load('DecisionTree.joblib')
    rf_bundle = joblib.load('RandomForest.joblib')
    return dt_bundle, rf_bundle  # Return full bundles including encoders

def compute_entropy(byte_data):
    byte_counts = Counter(byte_data)
    total = len(byte_data)
    return -sum((count / total) * math.log2(count / total) for count in byte_counts.values())

def preprocess_data(file_path):
    try:
        with open(file_path, "rb") as f:
            byte_data = f.read()

        file_size = os.path.getsize(file_path)
        entropy = compute_entropy(byte_data)

        try:
            text = byte_data.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            avg_line_length = np.mean([len(line) for line in lines]) if lines else 0
            column_count = np.mean([line.count(",") + 1 for line in lines if "," in line]) if lines else 0
        except:
            avg_line_length = 0
            column_count = 0

        bit_depth = 8

        byte_counts = np.zeros(256)
        for byte in byte_data:
            byte_counts[byte] += 1
        byte_distribution = byte_counts / len(byte_data)

        ext = os.path.splitext(file_path)[1][1:].lower()

        features = [ext, file_size, entropy, avg_line_length, column_count, bit_depth]
        features.extend(byte_distribution)

        columns = ['file_extension', 'file_size', 'entropy', 'avg_line_length', 'column_count', 'bit_depth'] + \
                 [f'byte_{i}' for i in range(256)]

        return pd.DataFrame([features], columns=columns)

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None