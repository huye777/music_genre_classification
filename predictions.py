import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('saved_model/my_genre_classification_model.h5')


le = LabelEncoder()

le.fit(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])


def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Error details: {e}")
        return None 
    return mfccs_processed


def predict_genre(file_path):
    features = extract_features(file_path)
    
    
    if features is None:
        return "Error processing file - could not extract features."
    
    
    features = np.array(features).reshape(1, -1)
    
    
    prediction = model.predict(features)
    
    
    predicted_index = np.argmax(prediction, axis=1)
    
    
    predicted_genre = le.inverse_transform(predicted_index)
    return predicted_genre[0]


file_path = 'new_audio/q2.wav'  
predicted_genre = predict_genre(file_path)
print(f"The genre of the sing is: {predicted_genre}")

