from array import array

import librosa
import pyaudio
import wave
import pickle
import audioop
import matplotlib.pyplot as plt
import numpy as np
import util

path= "/home/raunak/Desktop/output.wav"


count=0
CHUNK =1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
classes = ['siren','gunshot','horn']
clusters=30
RATE = 44100
RECORD_SECONDS = 0.5
#WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

with open('/home/raunak/PycharmProjects/beproject/ZCA.pk1', 'rb') as pickle_file:
    XCA_Matrix = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/dictionary.pk1', 'rb') as pickle_file:
    dictionary = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/classifier.pk1', 'rb') as pickle_file:
    rf = pickle.load(pickle_file)



stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer = 4000)
while True:

    threshold = 8000
    max_value = 0
    data = stream.read(4000)
    rms=audioop.rms(data,2)
    print(rms)
    if rms > threshold:
        count +=1
        print("* recording")
        frames = []
        frames.append(data)
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)


        print("* done recording")
        f = wave.open(path, 'wb')
        f.setnchannels(1)

        ####### this is not matching trained audios
        f.setsampwidth(2)

        f.setframerate(RATE)

        f.writeframes(b''.join(frames))
        f.close()

        data_lib = librosa.load('/home/raunak/Desktop/output.wav', sr=44100)
        inp = librosa.stft(data_lib[0])
        inp = librosa.feature.melspectrogram(S=inp, n_mels=60)
        inp = np.dot(XCA_Matrix, inp)
        print(np.shape(inp))
        c=util.encoding(inp,dictionary)
        print("encoding")
        print(np.shape(c))


        c = np.max(c,axis=1)
        #print
        c = c.reshape((1,c.shape[0]))
        print(np.shape(c))
        predict=int(rf.predict(c))
        print(predict)
        print(classes[predict]+" detected")

        break
    else:
        print("no audio")

    