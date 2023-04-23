from initialize_spotipy import initialize_spotipy
import numpy as np

def get_all_saved_tracks(user, num_songs=500, limit_step=50):
    tracks = []
    for offset in range(0, num_songs, limit_step):
        response = user.current_user_saved_tracks(
            limit=limit_step,
            offset=offset,
        )
        if len(response) == 0:
            break
        tracks_to_add = [item['track'] for item in response['items']]
        tracks += tracks_to_add

    return tracks

def get_songs(num_songs):

    # initialize spotipy
    USERNAME = "Robbie Schad"
    sp = initialize_spotipy(USERNAME)
    id = sp.me()['id']

    tracks = get_all_saved_tracks(sp, num_songs=num_songs, limit_step=50)
    songs = []
    i = 0
    while len(songs) < num_songs:
        # get song info
        track = tracks[i]

        # get song info and audio features in a dictionary
        audio_features = sp.audio_features(track['id'])[0]
        if audio_features:
            index = len(songs)
            id = track['id']
            name = track['name']
            duration = track['duration_ms'] // 1000

            acousticness = audio_features['acousticness']
            danceability = audio_features['danceability']
            energy = audio_features['energy']
            instrumentalness = audio_features['instrumentalness']
            valence = audio_features['valence']

            song = Song(index, id, name, duration, acousticness, danceability, energy, instrumentalness, valence)

            songs += [song]

        else:
            print('skipping: ', track['name'])
        
        i += 1

    return songs


def get_relevant_feature(songs, feature):
    if feature == 'acousticness':
        return np.asarray([song.acousticness for song in songs])
    elif feature == 'danceability':
        return np.asarray([song.danceability for song in songs])
    elif feature == 'energy':
        return np.asarray([song.energy for song in songs])
    elif feature == 'instrumentalness':
        return np.asarray([song.instrumentalness for song in songs])
    elif feature == 'valence':
        return np.asarray([song.valence for song in songs])
    else:
        print('Invalid feature given')
        return None


class Song():
    def __init__(self, index, id, name, duration, acousticness, danceability, energy, instrumentalness, valence):
        self.index = index
        self.id = id
        self.name = name
        self.duration = duration
        self.acousticness = acousticness
        self.danceability = danceability
        self.energy = energy
        self.instrumentalness = instrumentalness
        self.valence = valence
