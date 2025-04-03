"""
Usage: 

python cli.py --models random_forest svm

python cli.py --models random_forest --input-dir /path/to/custom/data

python cli.py --models random_forest --input-dir /path/to/custom/data --output-dir /path/to/custom/results


"""


import argparse
import sys
import os

# Set paths
current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(current_file), 'results')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from testing_models import ModelEvaluator

# Define paths
transcripts_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'transcripts', 'transcripts.csv')
original_audio_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'original_audio', 'real')
processed_audio_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')

# Define model configurations
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

def get_selected_models(selected_models, mode):
    """Filter selected models from available models based on mode."""
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
    parser.add_argument('--input-dir', default=DEFAULT_DATA_DIR,
                      help='Optional: custom input directory')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                      help='Optional: custom output directory')
    
    args = parser.parse_args()
    
    mode_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    selected_models = get_selected_models(args.models, args.mode)
    
    print(f"\nRunning {args.mode} configuration")
    print(f"Input directory: {args.input_dir}")
    print(f"Results will be saved in: {mode_dir}")
    
    for model_name, model in selected_models.items():
        model_dir = os.path.join(mode_dir, f"{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nEvaluating {model_name}...")
        print(f"Results will be saved in: {model_dir}")
        
        evaluator = ModelEvaluator({model_name: model}, model_dir)
        results = evaluator.evaluate(args.input_dir)
    
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main() 