import sounddevice as sd
from scipy.io.wavfile import write

def record():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2) #Record
    sd.wait()
    write('test.wav', fs, myrecording)
