import wave
import numpy
from scipy import signal
from librosa import feature
import os
import scipy.io.wavfile
from tinytag import TinyTag
j=0
k=0
bitrate=[]
g='C:/Users/Shalin/PycharmProjects/beproject/6'
directory=os.fsencode("C:/Users/Shalin/PycharmProjects/beproject/6")
for file in os.listdir(directory):
    i=os.fsdecode(file)
    tag=TinyTag.get(g+"/"+i)
    if tag.bitrate>2400:
        k=k+1
        print(k)

    if tag.bitrate<=1600:
        print(i)
        print(tag.bitrate)
        j=j+1
        print(j)
        #print(tag.bitrate)

        fs,sign=scipy.io.wavfile.read(g+"/"+i)
        a=numpy.array(sign)
        b=a.flatten()
        print("shape of flattened array")
        print(b.shape)



        #sign = car.readframes(car.getnframes())
        #sign = np.fromstring(sign, 'Int16')

        print("flattened array=")
        print (b)

        print("fs="+str(fs))
        f,t,sgn=signal.spectrogram(b,fs=fs,nperseg=1024,noverlap=512)

        print("shape of f=")
        print(f.shape)
        x,y = sgn.shape

        for i in range(5):
            for j in range(y):
                sgn[i][j]=0


        print("shape of sgn=")
        print(sgn.shape)
        Spect=feature.melspectrogram(S=sgn,n_mels=60)
print("greater than1600, less than 1600")
print(j,k)