import pickle
# from pydub import AudioSegment
from speech_sentiment_analysis.extractfeature import extract_feature

def predict_emotion():
    # audio_file = 'demo.mp3'
    # audio_file = AudioSegment.from_file(audio_file, format='mp3')
    # audio_file.export('test.wav', format='wav')
    audio_file = 'test.wav'

    model = pickle.load(open("speech_sentiment_analysis\\mlp.model", "rb"))
    kwargs = {
        'mel' : True,
        'chroma' : True,
        'mfcc' : True,
        'n_mfccs' : 40,
        # 'contrast' : True,
        # 'tonnetz' : True
    }
    features = extract_feature(audio_file,**kwargs).reshape(1,-1)
    return model.predict(features)[0]

