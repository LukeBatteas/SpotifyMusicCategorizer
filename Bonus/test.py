# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 18:06:06 2022

@author: lukeb
"""

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns


data = np.load("Playlist_Data.npy", allow_pickle=True)

print(data[1])

features = data[:,1:]
titles = data[:,0]

#We scale the data to ensure that
#feature units don't impact distances


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature'];
df = pd.DataFrame(features, columns = col)
df=df.convert_dtypes()
fig1 = plt.figure;

#sns.set_theme(style="ticks")
#sns.pairplot(df)#, hue="species")


#scaler = MinMaxScaler()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X_scaled)

model_1 = AgglomerativeClustering(n_clusters=9)
model_1 = model_1.fit(X_scaled)

predictions = model_1.fit_predict(X_scaled);
sns.set_theme(style="ticks")
col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature','class'];

classes = np.zeros((len(titles),1))
classes[:,0] = predictions 
print(np.shape(features))
print(np.shape(classes))
outputs = np.append(features, classes, 1)

df = pd.DataFrame(outputs, columns = col)

#sns.pairplot(df, hue="class")
fig2 =plt.figure;
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()