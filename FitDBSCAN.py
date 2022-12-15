"""
Created on Sat Nov 12 21:23:00 2022

@author: lukeb
"""

from sklearn.cluster import DBSCAN
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns


data = np.load("Playlist_Data.npy", allow_pickle=True)

print(data[1])

features = data[:,1:]
titles = data[:,0]

Save = False

col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature'];
df = pd.DataFrame(features, columns = col)
df=df.convert_dtypes()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

model_1 = DBSCAN(eps=2.0, min_samples=2)
predictions = model_1.fit_predict(X_scaled);

a, c = np.unique(predictions, return_counts=True)
print(a)
print(c)
sns.set_theme(style="ticks")
col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature','class'];


classes = np.zeros((len(titles),1))
classes[:,0] = predictions 
print(np.shape(features))
print(np.shape(classes))
outputs = np.append(features, classes, 1)

df = pd.DataFrame(outputs, columns = col)


if(Save):
    #titles
    i = np.append(titles.reshape(((len(titles),1))),classes,1)
    of = pd.DataFrame(i, columns = ['Title','Class'])
    of.to_csv('Titles_Luke.csv', sep = ',')