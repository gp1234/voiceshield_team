"""
Usage: 

python experiment_1.py default

python experiment_1.py overfit

python experiment_1.py underfit


"""


import argparse
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from testing_models import ModelEvaluator
import librosa

CODE_DIR = code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(code_dir)



transcripts_path = os.path.join(CODE_DIR, 'assets', 'audio', 'transcripts', 'transcripts.csv')
original_audio_path = os.path.join(CODE_DIR, 'assets', 'audio', 'processed_audio', 'real')
processed_audio_path = os.path.join(CODE_DIR, 'assets', 'audio', 'processed_audio', 'fake_1')
output_path = os.path.join(os.getcwd(), 'results')
os.makedirs(output_path, exist_ok=True)

DEFAULT_MODELS = {
    'random_forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    'svm': SVC(
        kernel='rbf',
        C=1.0,
        random_state=42
    ),
    'mlp': MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42
    ),
    'logistic': LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

OVERFIT_MODELS = {
    'random_forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    ),
    'svm': SVC(
        kernel='rbf',
        C=100.0,
        gamma='scale',
        random_state=42
    ),
    'mlp': MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=1000,
        learning_rate_init=0.01,
        random_state=42
    ),
    'logistic': LogisticRegression(
        C=100.0,
        max_iter=1000,
        random_state=42
    )
}

UNDERFIT_MODELS = {
    'random_forest': RandomForestClassifier(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=10,
        random_state=42
    ),
    'svm': SVC(
        kernel='linear',
        C=0.001,
        random_state=42
    ),
    'mlp': MLPClassifier(
        hidden_layer_sizes=(3,),
        max_iter=50,
        learning_rate_init=0.0001,
        random_state=42
    ),
    'logistic': LogisticRegression(
        C=0.001,
        max_iter=50,
        random_state=42
    )
}

MODEL_CONFIGS = {
    'default': DEFAULT_MODELS,
    'overfit': OVERFIT_MODELS,
    'underfit': UNDERFIT_MODELS
}

def extract_mfcc(path, n_mfcc=13):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def prepare_data():    
    data = []
    for fname in os.listdir(original_audio_path):
            data.append({"file": os.path.join(original_audio_path, fname), "label": "real"})

    for fname in os.listdir(processed_audio_path):
        if fname.endswith('.wav'):
            data.append({"file": os.path.join(processed_audio_path, fname), "label": "fake"})

    labeled_data = pd.DataFrame(data)
    return labeled_data

def get_selected_models(selected_models, mode):
    available_models = MODEL_CONFIGS[mode]
    if not selected_models:
        return available_models
    
    return {name: model for name, model in available_models.items() 
            if name in selected_models}

def main():
    parser = argparse.ArgumentParser(description='Evaluate audio classification models')
    parser.add_argument('mode', choices=['default', 'overfit', 'underfit'],
                      help='Model configuration mode to use')
    parser.add_argument('--models', nargs='+', 
                      choices=['random_forest', 'svm', 'mlp', 'logistic'],
                      help='Optional: specific models to evaluate. If not specified, all models will be evaluated.')
    parser.add_argument('--input-dir', default=original_audio_path,
                      help='Optional: custom input directory')
    parser.add_argument('--output-dir', default=output_path,
                      help='Optional: custom output directory')
    
    args = parser.parse_args()
    

    selected_models = get_selected_models(args.models, args.mode)
    
    print(f"\nRunning {args.mode} configuration")
    print(f"Input directory: {args.input_dir}")
    print(f"Results will be saved in: {output_path}")
    
    for model_name, model in selected_models.items():
        model_dir = os.path.join(output_path, args.mode, f"{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nEvaluating {model_name}...")
        print(f"Results will be saved in: {model_dir}")
        
        labeled_data = prepare_data()
        labeled_data["features"] = labeled_data["file"].apply(extract_mfcc)
        
        X = np.array(labeled_data['features'].tolist())
        y = labeled_data['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2, 
            random_state=42,
            stratify=y 
        )
        
        evaluator = ModelEvaluator({model_name: model}, model_dir)
        results = evaluator.evaluate(args.input_dir)
    
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main() 
    
    