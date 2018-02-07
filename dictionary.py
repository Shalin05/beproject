import wave
import numpy
from scipy import signal
from librosa import feature
import soundfile
from spherical_dictionary_learning import spherical_kmeans
import os
import matplotlib.pyplot as plt
import pickle

less = 0
greater = 0
X=numpy.zeros([60, 0])
print(X)

#### change path #######
g='/home/raunak/PycharmProjects/beproject/6'
directory=os.fsencode("/home/raunak/PycharmProjects/beproject/6")
for file in os.listdir(directory):
    i=os.fsdecode(file)
    a=soundfile.read(g+"/"+i)
    data=a[0]
    fs=a[1]

    data=numpy.array(data)
    data_flatten=data.flatten()

    f,t,sgn=signal.spectrogram(data_flatten,fs=fs,nperseg=1024,noverlap=512)
    x,y = sgn.shape

    for i in range(5):
        for j in range(y):
            sgn[i][j]=0

    Spect=feature.melspectrogram(S=sgn,n_mels=60)
    X = numpy.column_stack((X, Spect))


X,D,c=spherical_kmeans(X,150,20)





#pickle.dump(D,)
#pickle.dump(c,)