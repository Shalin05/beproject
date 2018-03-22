from array import array

import librosa
import pyaudio
import wave
import audioop
import matplotlib.pyplot as plt
import numpy as np

import util

path= "/home/raunak/Desktop/output.wav"


count=0
CHUNK =1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
#WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

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
        f.setsampwidth(2)
        f.setframerate(RATE)

        f.writeframes(b''.join(frames))
        f.close()
        inp=b''.join(frames)
        D = librosa.stft(inp)
        Spect = librosa.feature.melspectrogram(S=D, n_mels=60)

        c=util.encoding(Spect)
        predict=rf.predict(c)

        break
    else:
        print("no audio")

    