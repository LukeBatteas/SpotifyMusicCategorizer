"""
FitHAC.py

This program fits a HAC model to imported playlist data

A dendrogram can be generated, a scatter plot can be generated, and data can be saved to a csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import plot_dendrogram as dendro

#Do you want to generate a dendrogram?
Dendro = False;

#Do you want to generate a scatter plot?
Scatter = True;

#Do you want to save the results to a csv?
Save = False;


#Load data
data = np.load("Playlist_Data.npy", allow_pickle=True)


print(data[1])

#Extract features 
features = data[:,1:]
titles = data[:,0]

col = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature'];
df = pd.DataFrame(features, columns = col)
df=df.convert_dtypes()


#scaler = MinMaxScaler()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)



if(Dendro):
    # setting distance_threshold=0 ensures we compute the full tree.
    dendro.plot_dendrogram(X_scaled);


model_1 = AgglomerativeClustering(n_clusters=8)
#model_1 = model_1.fit(X_scaled)
predictions = model_1.fit_predict(X_scaled);
classes = np.zeros((len(titles),1))
classes[:,0] = predictions.astype(int)
print(np.shape(features))
print(np.shape(classes))

if(Scatter):
        col_2 = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration','time signature','class'];    
        sns.set_theme(style="ticks")
        
        #Yes, I do need to convert it twice
        outputs = np.append(features, classes.astype(int).astype(str), 1)
        print(outputs[0,:])
        df = pd.DataFrame(outputs, columns = col_2)
        sns.pairplot(df, hue='class')


if(Save):
    #titles
    i = np.append(titles.reshape(((len(titles),1))),classes,1)
    of = pd.DataFrame(i, columns = ['Title','Class'])
    of.to_csv('Titles_Luke.csv', sep = ',')