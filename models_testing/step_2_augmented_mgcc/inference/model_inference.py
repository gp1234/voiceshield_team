import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

class AudioPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
    
    def load_model(self, model_name='mlp'):
        """Load a trained model and its scaler"""
        model_path = os.path.join(self.model_dir, f"{model_name}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        print(f"Loaded model from: {model_path}")
        if self.scaler:
            print(f"Loaded scaler from: {scaler_path}")
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        return np.concatenate([mfccs_mean, mfccs_std])
    
    def predict(self, audio_path):
        """Predict if audio is real or fake"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # Extract features
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)  # Reshape for single sample
        
        # Scale features if scaler exists
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Try to get probability if the model supports it
        confidence = None
        try:
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(features)[0]
                confidence = probability.max() * 100
            elif hasattr(self.model, 'decision_function'):
                # For SVC, we can use decision_function
                decision = self.model.decision_function(features)[0]
                # Convert decision function to a confidence score (0-100)
                confidence = (1 / (1 + np.exp(-abs(decision)))) * 100
        except:
            # If both methods fail, we'll just show the prediction without confidence
            pass
        
        result = {
            'prediction': 'fake' if prediction == 1 else 'real'
        }
        
        if confidence is not None:
            result['confidence'] = confidence
            
        return result

def predict_files_in_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(current_dir, 'files')
    trained_models_dir = os.path.join(current_dir, 'trained_models')
    
    os.makedirs(files_dir, exist_ok=True)

    if not os.path.exists(trained_models_dir):
        print(f"\nError: trained_models directory not found at {trained_models_dir}")
        print("Please ensure you have copied the trained models to this location.")
        return
    
    predictor = AudioPredictor(trained_models_dir)
    
    try:
        predictor.load_model('mlp')
    except FileNotFoundError:
        print("\nError: Could not find mlp model.")
        print("Available models in trained_models directory:")
        models = [f.replace('_model.joblib', '') for f in os.listdir(trained_models_dir) 
                 if f.endswith('_model.joblib')]
        if models:
            print("\n".join(f"- {model}" for model in models))
        else:
            print("No models found.")
        return
    

    audio_files = [f for f in os.listdir(files_dir) if f.endswith(('.mp3', '.wav'))]
    
    if not audio_files:
        print(f"\nNo audio files found in {files_dir}")
        print("Please add .mp3 or .wav files to this directory")
        return
    
    print("\nPrediction Results:")
    print("=" * 50)
    
    # Process each file
    for audio_file in audio_files:
        audio_path = os.path.join(files_dir, audio_file)
        try:
            result = predictor.predict(audio_path)
            print(f"\nAudio file: {audio_file}")
            print(f"Prediction: {result['prediction']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.2f}%")
            else:
                print("Confidence: Not available for this model type")
            print("-" * 30)
        except Exception as e:
            print(f"\nError processing {audio_file}: {str(e)}")
            print("-" * 30)

if __name__ == '__main__':
    predict_files_in_directory() 