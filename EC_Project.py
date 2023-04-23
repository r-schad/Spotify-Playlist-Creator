import numpy as np
import pandas as pd
import os
import datetime
from Song import get_songs, get_relevant_feature
from SGA import SGA

def playlist_fitness(population, durations, features, desired_duration):
    '''
    Computes the fitness of all candidate playlists through the scalarization between 
    how far off from the intended duration it is, in addition to how varying the 
    audio-feature is throughout the playlist.

    NOTE: scalarization was chosen to combine the two criteria (duration and variance of audio-feature)
    because 1. we need to combine both criteria into 1 fitness score, and 2. this is easier than dealing with
    Pareto fronts.
    '''
    fitnesses = []
    max_duration = np.sum(durations)
    if len(population) == 1:
        population = [population]
        
    for genome in population:
        # sum of duration of all selected songs in the candidate genome
        duration = np.sum(durations[np.where(genome)])
        # duration score is distance from desired duration, normalized by maximum possible duration (if all songs included).
        # so, range of duration_score is [0.0, 1.0]. 
        if duration == 0:
            duration_score = 1.0
            feature_score = 0.0

        else: 
            # trying to minimize duration_score
            duration_score = (np.abs((duration/60) - (desired_duration/60)) / (max_duration/60)) ** 0.1
            # mean of feature values [0.0, 1.0] of all selected songs in the candidate genome. 
            # trying to maximize feature_score.
            feature_score = np.mean(features[np.where(genome)])

        # then compute fitness as the difference between feature score and duration score, so the actual fitness will be MAXIMIZED.
        # fitness scores are [-1.0, 1.0], so normalize it to [0.0, 1.0]
        fitness = (1 + feature_score - duration_score) / 2
        fitnesses += [fitness]

    if np.isnan(fitnesses).any():
        print('nan found')
        pass

    return fitnesses  


if __name__ == '__main__':

    AUDIO_FEATURE = 'acousticness' # ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence']
    DESIRED_MINUTES = 60
    desired_duration = DESIRED_MINUTES * 60

    GENOME_LENGTH = 200 
    POPULATION_SIZE = 400
    NUM_GENERATIONS = 300

    mutation_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    crossover_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    best_individuals = []
    all_results = []

    dt_string = str(datetime.datetime.today()).replace(':', '_')
    dt_string = str(dt_string).replace('.', '_')
    dt_string = str(dt_string).replace(' ', '_')

    if not os.path.exists('results'):
        os.mkdir('results')

    run_path = f"results\\{'Playlist_Creator' + '_' + dt_string}"

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    songs = get_songs(num_songs=GENOME_LENGTH)

    durations = np.asarray([song.duration for song in songs])
    
    features = get_relevant_feature(songs, AUDIO_FEATURE)

    for mutation_rate in mutation_rates:
        for crossover_rate in crossover_rates:
            if mutation_rate == 0.0 and crossover_rate == 0.0:
                continue
            for attempt in range(1):
                trial_name = f'mr={mutation_rate}_cr={crossover_rate}_trial={attempt}'.replace('.', 'p')
                # trial_path = run_path + '\\' + trial_name
                sga = SGA(problem_name=trial_name, 
                            fitness_function=playlist_fitness, 
                            desired_duration=desired_duration,
                            durations=durations,
                            features=features,
                            population_size=POPULATION_SIZE,
                            genome_length=GENOME_LENGTH,
                            mutation_rate=mutation_rate,
                            crossover_rate=crossover_rate,
                            directory=run_path)
                
                best_individual, results_dict = sga.evolution_loop(NUM_GENERATIONS, convergence_bound=100)

                best_individuals.append(best_individual)
                all_results.append(results_dict)


    sga_results = pd.DataFrame(all_results)

    sga_results.to_excel(run_path + '\\combined_results.xlsx')


    best_playlist = np.asarray(songs)[np.where(best_individual)] # TODO: implement this


    pass
    

