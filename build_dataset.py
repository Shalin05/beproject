import xml.etree.ElementTree
from  xml.dom import minidom
import xmltodict

import subprocess
import sys
from pydub import AudioSegment
from pydub.utils import which

import pydub
from pydub.playback import play






def slices(track_list,start,end,name):
    song = pydub.AudioSegment.from_wav(track_list)
    start=start * 1000
    end=end*1000
    sliced_audio=song[start:end]
    print(sliced_audio)
    sliced_audio.export(name, format="wav")
    #print("ho gaya")
    # play(sliced_audio)

dir='/home/raunak/Downloads/MIVIA_DB4_dist/testing/'
for i in range(1,30):
    print('\n')
    print(i)
    if(i<10):
        path=dir+"0000"+str(i)+".xml"
        save_path='/home/raunak/Downloads/MIVIA_DB4_dist/extracted_data/testing/8/'
        audio_path=dir+"sounds/0000"+str(i)+"_6.wav"
    else:
        path=dir+"000"+str(i)+".xml"
        audio_path=dir+"sounds/000"+str(i)+"_6.wav"

    print(audio_path)
    xmldoc = minidom.parse(path)
    a=xmldoc.childNodes
    b=a[0].childNodes
    c=b[1].childNodes
    length=len(c)
    for i in range(1,length,2):
        element=c[i]
        start = element.getElementsByTagName("STARTSECOND")[0].firstChild.nodeValue
        end=element.getElementsByTagName("ENDSECOND")[0].firstChild.nodeValue
        start=float(start)
        end=float(end)
        end_path = save_path+element.getElementsByTagName("PATHNAME")[0].firstChild.nodeValue
        print(audio_path,start,end,end_path)

        slices(audio_path,start,end,end_path)
    #child=a[0].childNodes




