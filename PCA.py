"""
Created on Sat Nov 26 09:36:45 2022

@author: lukeb
"""

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
import plot_dendrogram as dendro

data = np.load("Playlist_Data.npy", allow_pickle=True)


kPCA = False
Dendro = False;

print(data[1])

features = data[:,1:]
titles = data[:,0]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)


if(not(kPCA)):
    pca = PCA()
    pca.fit(X_scaled)
    plt.figure;
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(np.linspace(1,len(exp_var_cumul),len(exp_var_cumul)),exp_var_cumul,linestyle='--',marker='o')
    plt.xlabel("Number of Principal Components", fontsize=16)
    plt.ylabel("Explained Variance", fontsize=16)

else:
    pca = KernelPCA(kernel = 'rbf',gamma=0.3)
    pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(X_pca.shape)
if(Dendro):
    dendro.plot_dendrogram(X_pca)
    

#col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature'];
#df = pd.DataFrame(features, columns = col)
#df=df.convert_dtypes()
#fig1 = plt.figure;
