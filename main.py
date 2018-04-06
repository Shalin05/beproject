import numpy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
import pickle
from util import read_data, encoding,pooling, create_output,spherical_kmeans,ZCA_whitening

classes = ['siren','gunshot','scream']
######no of audio files in the above classes##########
no_of_audio = [250,233,250]

#no of clusters for dictionary learning#######
clusters = 30
no_of_iterations = 10
print(str(clusters)+" clusters")




with open('/home/raunak/PycharmProjects/beproject/ZCA.pk1', 'rb') as pickle_file:
    XCA_Matrix = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/dictionary.pk1', 'rb') as pickle_file:
    dictionary = pickle.load(pickle_file)

with open('/home/raunak/PycharmProjects/beproject/classifier.pk1', 'rb') as pickle_file:
    rf = pickle.load(pickle_file)



project_path = "/home/raunak/PycharmProjects/beproject/dataset/train/"
inp,dimensions,dim = read_data(project_path,classes)
XCA_Matrix =ZCA_whitening(inp)
inp=numpy.dot(XCA_Matrix, inp)

#creating global dictionary ##########
print("creating global dictionary")
a = 0
dictionary = numpy.zeros([60,0])
for i in dim.values():
    X = inp[:,a:a+i]
    a = a+i
    X, D, c = spherical_kmeans(X, clusters, no_of_iterations)
    dictionary = numpy.column_stack((dictionary, D))

c = encoding(inp,dictionary)
print("c mil gaya")


project_path = "/home/raunak/PycharmProjects/beproject/dataset/test/"
test_inp,test_dimensions,_ = read_data(project_path,classes)
test_inp=numpy.dot(XCA_Matrix, test_inp)


c_test = encoding(test_inp,dictionary)
print("c test mil gaya")

new_array = pooling(c,dimensions,clusters,classes)
print("train max pooling ho gaya")

new_array_test = pooling(c_test,test_dimensions,clusters,classes)
print("test max pooling ho gaya")

no_of_classes = len(classes)
output = create_output(no_of_audio,no_of_classes)

no_of_audio_test = [50,50,50]
test_output = create_output(no_of_audio_test,no_of_classes)

feat_variables = pd.DataFrame(new_array.T)
expected_output = pd.DataFrame(output)

feat_variables_test = pd.DataFrame(new_array_test.T)
expected_output_test = pd.DataFrame(test_output)

print("Training .......")
rf = RandomForestClassifier(criterion = "entropy",n_estimators = 10000 ,max_depth=10)
rf.fit(feat_variables,expected_output)

prediction_output = rf.predict(feat_variables)
print("Training")
print(accuracy_score(expected_output, prediction_output))
confusion = confusion_matrix(expected_output.as_matrix(), prediction_output)
df_cm = pd.DataFrame(confusion, range(no_of_classes),range(no_of_classes))
print(df_cm)



print("Testing")
prediction = rf.predict(feat_variables_test)
print(accuracy_score(expected_output_test, prediction))
confusion = confusion_matrix(expected_output_test.as_matrix(), prediction)
df_cm = pd.DataFrame(confusion, range(no_of_classes),range(no_of_classes))
print(df_cm)



output = open('/home/raunak/PycharmProjects/beproject/classifier.pk1', 'wb')
pickle.dump(rf,output)
output.close()

output = open('/home/raunak/PycharmProjects/beproject/ZCA.pk1', 'wb')
pickle.dump(XCA_Matrix,output)
output.close()

output = open('/home/raunak/PycharmProjects/beproject/dictionary.pk1', 'wb')
pickle.dump(dictionary,output)
output.close()
