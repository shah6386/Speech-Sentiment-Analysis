# Speech-Sentiment-Analysis
The aim of this project is to evaluate the sentiment of a speech. Analyzing sentiment of the customer over various parts of the call can help in understanding the transition of customer's emotion.

#Dataset
Data for source separation was taken from KAGGLE which consists of 300 audio files. Each of these files contains audio files.

RAVDESS dataset was used for classification purpose, it consists of 24 professional actors (12 female, 12 male). It can be found here or here. Dataset consists of different emotions like - neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Process
We approach this problem in three stages:

Stage 1 - We perform source separation on the audio, by performing VAD detection on the conversation and dividing the audio conversation into different chunks, on each chunk we apply GMM and a global GMM (UBM) on the whole conversation, using BIC and through spectral clustering we cluster every chunk into different speakers.

Stage 2 – We then apply sentiment analysis on supervised speech emotion dataset (RAVDESS) using  Neural Networks.

Stage 3 - We used the trained model from Stage 2 to classify the sentiment of the speaker chunks.

File description
convert.py helps convert .mp3 audio files into .wav

extract.py is used to extract files from the RAVDESS dataset and store it together on the basis of emotions.

predict.py is used to predicct the emotion.



