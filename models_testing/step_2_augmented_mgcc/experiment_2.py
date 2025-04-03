"""
Usage:

python cli.py --models random_forest svm
python cli.py --models random_forest --input-dir /path/to/custom/data
python cli.py --models random_forest --input-dir /path/to/custom/data --output-dir /path/to/custom/results
"""

import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import re
from testing_models_augmented import ModelEvaluator

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(current_file), 'results')

def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]), librosa.get_duration(y=y, sr=sr)

def build_dataset(root_dir):
    data = []

    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path):
            continue

        label = 0 if subdir == 'real' else 1
        source_model = subdir

        for file in tqdm(os.listdir(sub_path), desc=f"Processing {subdir}"):
            if file.endswith('.wav'):
                path = os.path.join(sub_path, file)

                try:
                    features, duration = extract_features(path)

                    data.append({
                        "file": path,
                        "label": label,
                        "source_model": source_model,
                        "features": features,
                        "duration": duration,
                        "group_id": file
                    })

                except Exception as e:
                    print(f"Failed to process {file}: {e}")
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=True)
    return pd.DataFrame(data)

DEFAULT_MODELS = {
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'svm': SVC(kernel='rbf', C=1.0, random_state=42),
    'mlp': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
    'logistic': LogisticRegression(max_iter=1000, random_state=42)
}

def get_selected_models(selected_models):
    if not selected_models:
        return DEFAULT_MODELS
    return {name: DEFAULT_MODELS[name] for name in selected_models if name in DEFAULT_MODELS}

def main():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument('--models', nargs='+', choices=DEFAULT_MODELS.keys(),
                        help='Optional: specific models to evaluate. If not specified, all models will be evaluated.')
    parser.add_argument('--input-dir', default=DEFAULT_DATA_DIR,
                        help='Optional: custom input directory')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Optional: custom output directory')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected_models = get_selected_models(args.models)

    print(f"\nEvaluating models: {', '.join(selected_models.keys())}")
    print(f"Input directory: {args.input_dir}")
    print(f"Results will be saved in: {args.output_dir}")
    labeled_data = build_dataset(args.input_dir)
    
    evaluator = ModelEvaluator(selected_models, labeled_data, args.output_dir)
    evaluator.evaluate()

    print("\nEvaluation complete!") 
    
    


if __name__ == '__main__':
    main()
