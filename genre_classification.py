import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


dataset_path = 'GTZAN'


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


def get_all_file_paths_and_labels(dataset_path):
    features = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith('.wav'):
                file_path = os.path.join(root, name)
                genre = os.path.basename(root)
                features.append(file_path)
                labels.append(genre)
    return features, labels


features, labels = get_all_file_paths_and_labels(dataset_path)


le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded)


X = np.array([extract_features(file) for file in features])
y = np.array(y_categorical)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer with one unit per genre


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")


model.save('my_genre_classification_model.h5')
