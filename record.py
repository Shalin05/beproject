import librosa
import pyaudio
import wave
import math
import pickle
import time
import audioop
import sounddevice as sd
import soundfile as sf
import numpy as np
import util


path= "/home/raunak/Desktop/output.wav"


count = 0
CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 1
classes = ['siren','gunshot','horn']
clusters = 30
RATE = 32000
#WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

with open('/home/raunak/PycharmProjects/beproject/ZCA.pk1', 'rb') as pickle_file:
    XCA_Matrix = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/dictionary.pk1', 'rb') as pickle_file:
    dictionary = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/classifier.pk1', 'rb') as pickle_file:
    rf = pickle.load(pickle_file)



stream16 = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer = CHUNK)

while True:
    threshold = 10000
    data = stream16.read(CHUNK)

    rms = audioop.rms(data,2)
    print(rms)
    if rms > threshold:

        stream = p.open(format=pyaudio.paFloat32,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")
        frames = []
        #frames.append(data)
        count = 0
        out_data=np.zeros((0,1))
        while True:
            #data = stream.read(CHUNK)

            output = stream.read(CHUNK)
            output = np.frombuffer(output,dtype=np.float32)
            #print(np.shape(out_data))
            output = np.reshape(output,(np.shape(output)[0],1))
            #print(np.shape(output))
            data16 = stream16.read(CHUNK)
            rms = audioop.rms(data16, 2)
            print(rms)
            if rms < threshold:
                count+=1
            if count > 5:
                break

            out_data = np.row_stack((out_data, output))


        '''
        data1 = b''.join(frames)
        out_data = np.fromstring(data1,dtype=np.float32)
        
        print(np.max(out_data))
        print(np.min(out_data))
                '''

        #time.sleep(2)
        #sd.play(out_data)
        #sf.write(path, data = out_data,samplerate=RATE,subtype= 'FLOAT')
        print("* done recording")
        out_data = out_data.ravel()

        '''
        f = wave.open(path, 'wb')
        f.setnchannels(1)

        sample_width = p.get_sample_size(4)

        f.setsampwidth(sample_width)

        f.setframerate(RATE)
        f.writeframes(out_data)
        f.close()

        data_lib = librosa.load('/home/raunak/Desktop/output.wav', sr=32000)'''
        inp = librosa.stft(out_data)
        inp = librosa.feature.melspectrogram(S=inp, n_mels=60)
        inp = np.dot(XCA_Matrix, inp)
        print(np.shape(inp))
        c=util.encoding(inp,dictionary)
        print("encoding")
        print(np.shape(c))

        c = np.max(c,axis=1)
        c = c.reshape((1,c.shape[0]))
        print(np.shape(c))
        predict=int(rf.predict(c))
        print(predict)
        print(classes[predict]+" detected")
        #time.sleep(2)
        break
    else:
        print("no audio")