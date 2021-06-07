pip install soundfile

pip install -U scikit-learn

import soundfile  
import glob
import os
import pickle 
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import librosa.display

def spec(file_name,em):
  data, sr = librosa.load(file_name)
  stft = librosa.core.stft(data)
  ddb = librosa.core.amplitude_to_db(abs(stft))
  # plt.figure(figsize = (20, 4))
  # librosa.display.specshow(ddb, sr=sr, x_axis='time', y_axis='hz')
  # plt.savefig('/content/1.jpeg')
  plt.figure(figsize = (20, 4))
  librosa.display.specshow(ddb, sr=sr, x_axis='time', y_axis='log')
  plt.savefig('/content/speclog/'+em+'/'+file_name[72:-4]+'.jpeg')
  plt.close('all')

def extract_feature(file_name, **kwargs):
    """
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

def load_data():
    X, y = [], []
    i=0
    for file in glob.glob("/content/drive/My Drive/speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        
        basename = os.path.basename(file)
        
        emotion = int2emotion[basename.split("-")[2]]
        
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # spec(file,emotion)
        # i=i+1
        # print(i)
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        X.append(features)
        y.append(emotion)
   
    return X,y

# load_data()

# import pandas as pd
# df = pd.DataFrame(np.vstack(X))
# df['emotion'] = np.vstack(y)

# df.to_csv('/content/sample_data/data.csv', index=False, header=True)

# df.head()

X, y = load_data()
# load RAVDESS dataset, 75% training 25% testing
X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=0.25, random_state=7)

# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
print("[+] Number of features:", X_train.shape[1])

model_params = {
    'alpha': 0.01,
    'batch_size': 1,
    'epsilon': 1e-10, 
    'hidden_layer_sizes': (500,), 
    'learning_rate': 'adaptive', 
    'max_iter': 1000, 
}

# initialize MLP
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data 
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# predict 25% of data 
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# pip install pydub

# import librosa
# import numpy as np

# def extract_features(file_name, **kwargs):
#     """
#     Extract feature from audio file :
#             - MFCC (mfcc)
#             - Chroma (chroma)
#             - MEL Spectrogram Frequency (mel)
#             - Contrast (contrast)
#             - Tonnetz (tonnetz)
#     """
#     mfcc = kwargs.get("mfcc")
#     chroma = kwargs.get("chroma")
#     mel = kwargs.get("mel")
#     contrast = kwargs.get("contrast")
#     tonnetz = kwargs.get("tonnetz")
#     X, sample_rate = librosa.load(file_name)
#     #print(sample_rate)
#     if chroma or contrast:
#         stft = np.abs(librosa.stft(X))
#     result = np.array([])
#     if mfcc:
#         mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#         result = np.hstack((result, mfccs))
#     if chroma:
#         chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, chroma))
#     if mel:
#         mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, mel))
#     if contrast:
#         contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, contrast))
#     if tonnetz:
#         tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
#         result = np.hstack((result, tonnetz))
#     return result

# from pydub import AudioSegment

# def predict_emotion():
#     audio_file = '/content/Zinda.mp3'
#     audio_file = AudioSegment.from_file(audio_file, format='mp3')
#     audio_file.export('/content/test.wav', format='wav')
#     audio_file = 'test.wav'

#     features = extract_features(audio_file, mel=True, chroma=True, mfcc=True, contrast=True, tonnetz=True).reshape(1,-1)
#     result = model.predict(features)
#     return result[0]

# print(predict_emotion())

# if not os.path.isdir("result"):
#     os.mkdir("result")

# pickle.dump(model, open("result/mlp_classifier.model", "wb"))
