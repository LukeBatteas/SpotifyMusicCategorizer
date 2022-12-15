# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:44:37 2022

@author: lukeb
"""


import pandas as pd 
from getTracks import get_playlist_tracks_more_than_100_songs
import spotipy 
import numpy as np
sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 


#You will need a valid login to run this 
cid ="[REMOVED]" 
secret = "[REMOVED]" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 

 
uri = "6qeSBRzXBFqyGnOreDveCg"

df = get_playlist_tracks_more_than_100_songs("geomcintire", uri) 
temp_features=df.to_numpy();
np.save("Playlist_Data", temp_features, allow_pickle=True, fix_imports=True)
#Just confirm that it read correctly 
#print(temp_features[1][4]) 
#0 -> title
#1 -> danceability
#2 -> energy
#3 -> key
#4 -> loudness
#5 -> mode
#6 -> speechiness
#7 -> acousticness
#8 -> instrumentalness
#9 -> liveness
#10 -> valence
#11 -> tempo
#12 -> duration_ms
#13 -> time_signature
