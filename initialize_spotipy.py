from __future__ import print_function
import sys
import spotipy
import spotipy.util as util

def initialize_spotipy(username):
    scope = 'user-library-read playlist-modify-public'
    client_id = '57f5d68d3b274171ac2906a882cf14d5'
    client_secret = 'f835aba032f54395a57ef03b43364694'
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
