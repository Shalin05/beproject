import numpy
from scipy import signal
import seaborn as sn
from librosa import feature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import soundfile
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from spherical_dictionary_learning import spherical_kmeans
from scipy.sparse import csr_matrix
import os
import pandas as pd

X=numpy.zeros([60, 0])

classes=['glass','gunshots','screams']

######no of audio files in the above classes##########
no_of_audio=[208,208,208]


dimensions=[]
#####global dictionary #####
dictionary=numpy.zeros([60,0])

######global input######
inp=numpy.zeros([60,0])

#####no of clusters for dictionary learning#######
clusters =200
print(str(clusters)+"clusters")

#### change path #######
project_path="/home/raunak/Downloads/MIVIA_DB4_dist/final data/training/"
for number in classes:

    g=project_path+number
    directory=os.fsencode(g)
    for file in os.listdir(directory):
        i=os.fsdecode(file)
        a=soundfile.read(g+"/"+i)
        data=a[0]
        fs=a[1]

        data=numpy.array(data)
        data_flatten=data.flatten()

        f,t,sgn=signal.spectrogram(data_flatten,fs=fs,nperseg=1024,noverlap=512)
        x,y = sgn.shape

        #####high pass filter######
        for i in range(5):
            for j in range(y):
                sgn[i][j]=0


        Spect=feature.melspectrogram(S=sgn,n_mels=60)
        dimensions.append(numpy.shape(Spect)[1])
        X = numpy.column_stack((X, Spect))

    X, D, c = spherical_kmeans(X, clusters, 10)
    print(str(number)+" hua")
    dictionary = numpy.column_stack((dictionary, D))
    inp= numpy.column_stack((inp, X))


#pkl_file = open('/home/raunak/PycharmProjects/beproject/dictionary.p', 'rb')
#D = pickle.load(pkl_file)

######learn codebook from the global dictionary#####


sparse_dict=csr_matrix(dictionary)
print(numpy.shape(sparse_dict))
print("dot")
print(numpy.shape(inp))
c = sparse_dict.T.dot(inp)
c = c * (c >= numpy.max(c, axis=0))
print("c mil gaya")


test_dimensions=[]
test_inp=numpy.zeros([60,0])

###read test data#############
project_path="/home/raunak/Downloads/MIVIA_DB4_dist/extracted_data/testing/8/"
for number in classes:
    g = project_path+number
    directory = os.fsencode(g)
    for file in os.listdir(directory):
        i=os.fsdecode(file)
        a=soundfile.read(g+"/"+i)
        data=a[0]
        fs=a[1]

        data=numpy.array(data)
        data_flatten=data.flatten()

        f,t,sgn=signal.spectrogram(data_flatten,fs=fs,nperseg=1024,noverlap=512)
        x,y = sgn.shape

        #####high pass filter######
        for i in range(5):
            for j in range(y):
                sgn[i][j]=0

        Spect = feature.melspectrogram(S=sgn, n_mels=60)
        test_dimensions.append(numpy.shape(Spect)[1])
        X = numpy.column_stack((X, Spect))

print(str(number) + " hua")
test_inp = numpy.column_stack((inp, X))


c_test = sparse_dict.T.dot(test_inp)
c_test = c_test * (c_test >= numpy.max(c_test, axis=0))
print("c test mil gaya")


global_dict_para=clusters*len(classes)
print("global para"+str(global_dict_para))
new_array=numpy.zeros([global_dict_para,0])
#print(numpy.shape(new_array))
print(numpy.shape(c))


print("max pooling")
a=0
for i in dimensions:
    l=numpy.max(c[:, a:a + i],axis=1)
    b=numpy.shape(l)[0]
    l.shape=[b,1]
    #print(numpy.shape(l))

    new_array=numpy.column_stack((new_array,l))
    a=a+i

print("max pooling ho gaya")
output=[]

a=0
new_array_test=numpy.zeros([global_dict_para,0])

for i in test_dimensions:
    l=numpy.max(c_test[:, a:a + i],axis=1)
    b=numpy.shape(l)[0]
    l.shape=[b,1]
    #print(numpy.shape(l))

    new_array_test=numpy.column_stack((new_array_test,l))
    a=a+i



no_of_classes=len(classes)

for i in range(no_of_classes):
    for j in range(no_of_audio[i]):
        output.append(i)

no_of_audio_test=[144,376,112]
no_of_audio_test=[18,47,14]
test_output=[]
for i in range(no_of_classes):
    for j in range(no_of_audio_test[i]):
        test_output.append(i)

print(numpy.shape(test_output))
feat_variables = pd.DataFrame(new_array.T)
expected_output=pd.DataFrame(output)



feat_variables_test = pd.DataFrame(new_array_test.T)
expected_output_test=pd.DataFrame(test_output)

print("Training .......")
rf=RandomForestClassifier(criterion="entropy",n_estimators=2000,max_features=0.33)
#class_weight={0:1,1:1,2:1}
rf.fit(feat_variables,expected_output)

prediction_output=rf.predict(feat_variables)
print("Training")
print (accuracy_score(expected_output, prediction_output))
confusion=confusion_matrix(expected_output.as_matrix(), prediction_output)

df_cm = pd.DataFrame(confusion, range(no_of_classes),
                  range(no_of_classes))
print(df_cm)

print("Expected output test:"+str(numpy.shape(expected_output_test)))
print("Testing")
prediction=rf.predict(feat_variables_test)
print(numpy.shape(expected_output_test))
print(numpy.shape(prediction))
print (accuracy_score(expected_output_test, prediction))
confusion=confusion_matrix(expected_output_test.as_matrix(), prediction)

df_cm = pd.DataFrame(confusion, range(no_of_classes),
                  range(no_of_classes))
print(df_cm)

'''
output = open('/home/raunak/PycharmProjects/beproject/classifier.p', 'wb')
pickle.dump(rf,output)
output.close()
'''

'''
output = open('/home/raunak/PycharmProjects/beproject/classifier.p', 'wb')
pickle.dump(on,output)
output.close()

output = open('/home/raunak/PycharmProjects/beproject/dictionary.p', 'wb')
pickle.dump(D,output)
output.close()
'''
