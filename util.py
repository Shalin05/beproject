import numpy as np
import os
import librosa
from librosa import feature
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import dict_learning
import numpy.linalg as LA


def read_data(project_path,classes):
    dimensions = []
    inp = np.zeros([60, 0])
    dim = {}
    for number in classes:
        X = np.zeros([60, 0])
        dim[number]=0
        g=project_path+number
        directory=os.fsencode(g)
        for file in os.listdir(directory):
            i=os.fsdecode(file)
            data_lib=librosa.load(g+"/"+i, sr=32000)
            print("max = "+str(np.max(data_lib[0])))
            print("min = " + str(np.min(data_lib[0])))
            D = librosa.stft(data_lib[0])
            Spect=feature.melspectrogram(S=D,n_mels=60)
            dimensions.append(np.shape(Spect)[1])
            dim[number]+=np.shape(Spect)[1]
            X = np.column_stack((X, Spect))

        inp = np.column_stack((inp, X))

    return inp,dimensions,dim

def encoding(inp,dictionary):
    sparse_dict = csr_matrix(dictionary)
    print(np.shape(sparse_dict))
    print("dot")
    print(np.shape(inp))
    c = sparse_dict.T.dot(inp)
    c = c * (c >= np.max(c, axis=0))
    return c

def pooling(c,dimensions,clusters,classes):
    a = 0
    global_dict_para = clusters * len(classes)
    new_array = np.zeros([global_dict_para, 0])

    for i in dimensions:
        l = np.max(c[:, a:a + i], axis=1)
        b = np.shape(l)[0]
        l.shape = [b, 1]
        # print(numpy.shape(l))

        new_array = np.column_stack((new_array, l))
        a = a + i
    return new_array

def create_output(no_of_audio,no_of_classes):
    output=[]
    for i in range(no_of_classes):
        for j in range(no_of_audio[i]):
            output.append(i)
    return output

def spherical_kmeans(X,num_centroids, num_iter):

    D = np.random.normal(size=(60,num_centroids))
    D = D / np.sqrt(np.sum(D ** 2, axis=0))
    #### Spherical k_means
    for i in range(num_iter):
        S = D.T.dot(X)
        S = S * (S >= np.max(S, axis=0))
        D = X.dot(S.T) + D
        D = D / np.sqrt(np.sum(D ** 2, axis=0))

    #d,c,e=dict_learning(X=X,n_components=num_centroids,alpha=1,max_iter=num_iter)
    return X,D,S



def ZCA_whitening(X):

    ##### normalize
    #X=normalize(X,axis=1)
    X_mean = np.mean(X, axis=0)
    X_var = np.var(X, axis=0) + 10
    X = (X - X_mean) / np.sqrt(X_var)

    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # [M x M]
    return ZCAMatrix