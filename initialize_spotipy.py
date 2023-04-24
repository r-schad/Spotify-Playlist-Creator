from __future__ import print_function
import sys
import spotipy
import spotipy.util as util

def initialize_spotipy(username):
    scope = 'user-library-read playlist-modify-public'
    client_id = 'MY ID'
    client_secret = 'MY SECRET'
    redirect_uri = 'https://localhost/'
    token = util.prompt_for_user_token(username, 
                                       scope=scope,
                                       client_id=client_id,
                                       client_secret=client_secret,
                                       redirect_uri=redirect_uri)

    if token:
        sp = spotipy.Spotify(auth=token)
        return sp
    else:
        print("Can't get token for", username)
        return 0
