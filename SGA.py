import numpy as np
import matplotlib.pyplot as plt
import datetime
import os, math
from sklearn.metrics.pairwise import euclidean_distances

class SGA():
    def __init__(self, problem_name, fitness_function, desired_duration, durations, features, population_size, genome_length, mutation_rate, crossover_rate, directory):
        '''
        SGA CLASS:

        Serves to (hopefully) allow for extremely reusable code for future homeworks, where the only thing that will 
        need to change is the fitness_function provided into this constructor function, and the hyperparameters of the
        evolution loop to be run.

        But that might be wishful thinking... 
        '''
        self.directory = directory
        # here, each of the options for the SGA receives the *function as an object*
        # (this means we can just call the normal SGA option name, and it will call the correct function)
        # e.g. calling sga.mutation() will automatically call sga.fixed_bitwise_mutation()

        self.initialization = self.init_uniform_rand_pop
        self.parent_selection = self.random_pairing
        self.crossover = self.crossover_n_swap
        self.mutation = self.fixed_bitwise_mutation
        self.survivor_selection = self.tournament_selection
        self.check_termination = self.population_converged

        # here, we initialize the hyperparameters chosen for this run of the SGA

        # REPRESENTATION TYPE: numpy array of binary values [0/1] of length <genome_length>
        # (using a numpy array instead of an actual string, since it's just easier to work with)
        self.representation_type = np.ndarray(genome_length) 
        # population size should be an even number for the sake of SGA crossover 
        # (since each pair of parents must create a pair of kids, the number of kids to produce should be even)
        self.population_size = population_size
        assert self.population_size % 2 == 0, 'population size must be an even number'
        # initialize the population genomes based on the initialization method specified
        self.population = self.initialization()
        # the parameters specific to the actual evolution_loop() to be run
        self.problem_name = problem_name
        self.fitness_function = fitness_function
        self.desired_duration = desired_duration
        self.durations = durations
        self.features = features
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate


    def init_uniform_rand_pop(self):
        '''
        INITIALIZATION: Uniform Random
        
        Returns a uniform, random 2-d, binary [0/1] array of shape (n, b)
            where n is the population size,
            and b is the number of bits in the genome. 
        '''
        num_genome_bits = self.representation_type.shape[0]
        # low is inclusive, high is exclusive, so this gives us binary [0/1] values
        return np.asarray(np.random.choice([0, 1], size=(self.population_size, num_genome_bits)), dtype=int)


    def fit_prop_roulette_parent_selection(self):
        '''
        PARENT SELECTION: Fitness Proportional Roulette Wheel

        Generates two arrays of parents, as selected by the fitness-proportional roulette wheel. 

        Returns a tuple of the array of parent_1 indexes and the array of parent_2 indexes.
        '''
        # get the raw fitness of the entire population
        # (using absolute value to account for minimizing or maximizing fitness functions)
        raw_fitnesses = self.fitness_function(self.population, self.durations, self.features, self.desired_duration)
        min_fitness = np.min(raw_fitnesses)

        # scale fitnesses so minimum is 0.0
        relative_fitnesses = raw_fitnesses - min_fitness
        # rescale fitnesses so minimum is 1.0
        relative_fitnesses = np.add(relative_fitnesses, 1)

        # normalize relative fitnesses by the sum of all fitness scores
        fitnesses_sum = np.sum(relative_fitnesses)
        relative_fitnesses = np.divide(relative_fitnesses, fitnesses_sum)
        # now relative_fitnesses is a *NON-CUMULATIVE* probability density function (PDF)

        # create cumulative distribution function (CDF) of fitnesses (i.e. the roulette wheel) from the PDF
        roulette_wheel = np.zeros(shape=relative_fitnesses.shape) # roulette wheel should be same shape as relative_fitnesses

        # first value in CDF is the first relative_fitness value
        # (since the CDF is cumulative, the first CDF value is just the first fitness value, and all following values accumulate from that)
        roulette_wheel[0] = relative_fitnesses[0]

        # iterate through each individual relative fitness score and assign the slot size in the roulette wheel
        for i in range(1, roulette_wheel.shape[0]):
            # slot sizes are the (previous CDF value) + (individual's relative fitness)
            roulette_wheel[i] = roulette_wheel[i-1] + relative_fitnesses[i]

        # get 2 random lists of values [0,1) for parent selection - each list contains (1/2 of the population size) elements
        # (this is because parent selection is done by pairs, and each pair will produce 2 children)
        random_1 = np.random.rand(self.population_size // 2)
        random_2 = np.random.rand(self.population_size // 2)

        # now convert random lists of floating point values to parent indexes based on roulette wheel
        parent_1s = []
        parent_2s = []

        # get the two parent indexes for each pair of random numbers
        for pair_count in range(random_1.shape[0]):
            # select the first index of the values where the random value < roulette wheel value to get parent index
            parent_1s.append(np.argwhere(random_1[pair_count] < roulette_wheel)[0][0])
            parent_2s.append(np.argwhere(random_2[pair_count] < roulette_wheel)[0][0])

        # convert parent arrays to numpy arrays for consistency with rest of code
        parent_1s = np.array(parent_1s)
        parent_2s = np.array(parent_2s)

        # return a tuple of our 2 arrays of selected parent indexes
        return (parent_1s, parent_2s)


    def random_pairing(self):
        p1s, p2s = np.random.choice(np.asarray(range(self.population_size)), size=(2, self.population_size//2), replace=False)
        return p1s, p2s


    def tournament_selection(self, fitnesses):
        all_parents = []
        idxs = list(range(len(fitnesses)))
        for i in range(len(fitnesses) // 2):
            # grab 2 candidates *without* replacement, and store the winners index in all_parents
            idx1, idx2 = np.random.choice(idxs, size=2, replace=False)
            if fitnesses[idx1] >= fitnesses[idx2]:
                all_parents += [idx1]
            
            elif fitnesses[idx1] < fitnesses[idx2]:
                all_parents += [idx2]

        return np.asarray(all_parents)


    def crossover_n_swap(self, p1s, p2s, max_n):
        '''
        CROSSOVER: n-swap Crossover

        For approximately <crossover_rate> of the population, performs an n-swap crossover of the two parent genomes, 
        which produces two resulting, pre-mutation, crossover'd genomes. 

        n-swap crossover is defined as:
        for a random number (n) crossover locations, the genome bit at that point is swapped between parent_1 and parent_2.

        For the other <1-crossover_rate> of the population, appends the two provided genomes into the array of resultant genomes.

        Returns an array of the pre-mutation, crossover population genomes.
        ''' 

        crossover_pop = np.ndarray(self.population.shape)
        # n_probs_cdf = np.flip(np.linspace(0.0, 1.0, self.representation_type.shape[0]))
        # n_probs_cdf_offset = np.roll(np.flip(np.linspace(0.0, 1.0, self.representation_type.shape[0])), shift=-1)
        # n_probs_cdf_offset[-1] = 0
        # n_probs = probs_cdf - n_probs_cdf_offset # literally just a uniform distribution bruh

        # iterate through each pair of parents
        for i in range(p1s.shape[0]):
            # roll a random number [0,1) to determine if we should crossover this pair of parents
            if np.random.rand() < self.crossover_rate: # do crossover
                # get number of crossover points
                n = np.random.randint(low=0, high=max_n)

                # get the n genes to swap
                xover_points = np.random.randint(0, self.representation_type.shape[0], size=n)
                where_to_swap = np.zeros(self.representation_type.shape[0])
                where_to_swap[xover_points] = 1

                # now store new (crossover'd) genome of both children in crossover_pop
                crossover_pop[2 * i] = np.where(where_to_swap, self.population[p2s[i]], self.population[p1s[i]])
                crossover_pop[(2 * i) + 1] = np.where(where_to_swap, self.population[p1s[i]], self.population[p2s[i]])

            else: # don't crossover, make children identical to parents
                crossover_pop[2 * i] = self.population[p1s[i]]
                crossover_pop[(2 * i) + 1] = self.population[p2s[i]]

        return crossover_pop



    def crossover_1p(self, p1s, p2s):
        '''
        CROSSOVER: Single-Point Crossover

        For approximately <crossover_rate> of the population, performs a 1-point crossover of the two parent genomes, 
        which produces two resulting, pre-mutation, crossover'd genomes. 

        For the other <1-crossover_rate> of the population, appends the two provided genomes into the array of resultatnt genomes.

        Returns an array of the pre-mutation, crossover population genomes.
        ''' 

        crossover_pop = np.ndarray(self.population.shape)
        # iterate through each pair of parents
        for i in range(p1s.shape[0]):
            # roll a random number [0,1) to determine if we should crossover this pair of parents
            if np.random.rand() < self.crossover_rate: # do crossover
                # get random crossover point between [0, b-1] where b is number of bits in representation
                xover_pt = np.random.randint(0, self.representation_type.shape[0])

                # now store new (crossover'd) genome of both children in crossover_pop

                # first chunk of p1's genome with second chunk of p2's genome
                crossover_pop[2 * i] = np.concatenate((self.population[p1s[i]][0:xover_pt], self.population[p2s[i]][xover_pt:]))
                # first chunk of p2's genome with second chunk of p1's genome
                crossover_pop[(2 * i) + 1] = np.concatenate((self.population[p2s[i]][0:xover_pt], self.population[p1s[i]][xover_pt:]))

            else: # don't crossover, make children identical to parents
                crossover_pop[2 * i] = self.population[p1s[i]]
                crossover_pop[(2 * i) + 1] = self.population[p2s[i]]

        return crossover_pop


    def fixed_bitwise_mutation(self, temp_pop, mutation_rate):
        '''
        MUTATION: Fixed-Rate Bitwise Mutation

        Given a temporary, post-crossover population of genomes,
        selects approximately <mutation_rate> of the bits and flips them.

        Returns the array of post-mutation, population genomes. 
        '''
        # get array of random numbers of same shape as our population, (n, b), 
        # where n is number of individuals and b is number of bits in representation
        # and compare each value with the mutation_rate.
        # values that are < mutation rate will be mutated (so, they will be flipped, by storing 1-value)
        # and values that are >= mutation rate will be left alone
        mutated_pop = np.where(np.random.rand(temp_pop.shape[0], temp_pop.shape[1]) < mutation_rate, 1 - temp_pop, temp_pop)
        return mutated_pop


    def calc_mutation_rate(self, gen):
        return self.mutation_rate * np.exp(-1 * gen / 2)


    def mu_plus_lambda(self, lambda_pop):
        all_individuals = np.concatenate((self.population, lambda_pop))
        fitnesses = self.fitness_function(all_individuals)


    def full_generational_replacement(self, new_pop):
        '''
        SURVIVOR SELECTION: Full Replacement
        REPLACEMENT: Generational

        Since all children will just replace their parents in this SGA implementation,
        this function just swaps out the old population array with the new, array of post-mutation, population genomes.

        Returns None.
        '''
        assert self.population.shape == new_pop.shape, "New population isn't the same size"
        self.population = new_pop


    def percentage_nonunique_genomes(self):
        '''
        Computes the percentage of nonunique genomes by determining how many genomes are repeated 
        in the population.

        Returns the percentage (not the ratio) of nonunique genomes.
        '''
        # get the number of occurrences of each present genome
        _, counts = np.unique(self.population, axis=0, return_counts=True)
        # and count how many of those only occurred once
        num_unique = np.count_nonzero(counts == 1)
        # return the percentage of nonunique genomes
        return 100 * (self.population.shape[0] - num_unique) / self.population.shape[0]


    def population_converged(self, convergence_bound):
        '''
        TERMINATION: Convergence with Upper Generational Bound 
        
        Checks if the percentage of nonunique genomes is > upperbound.

        Returns a boolean of whether evolution should terminate. 
        '''
        percent_nonunique = self.percentage_nonunique_genomes()
        if percent_nonunique >= convergence_bound:
            print(f'Converged with {percent_nonunique}% Nonunique Genomes.')
            return True
        else:
            return False


    def evolution_loop(self, num_generations, convergence_bound):
        '''
        Performs the actual SGA evolution algorithm from initialization to termination
        by looping through a maximum of <num_generations> generations,
        or halts evolution when the (percentage of nonunique genomes) is greater than <convergence_bound>. 

        Creates a 'results' directory (if it doesn't exist) and a directory for the current run within the results directory.
        Creates and writes the logging info into an output.txt file that exists in the current run directory.
        Also creates a plot of the best fitness score and the avg fitness score throughout the evolution.

        Returns None.
        '''
        dt_string = str(datetime.datetime.today()).replace(':', '_')
        dt_string = str(dt_string).replace('.', '_')
        dt_string = str(dt_string).replace(' ', '_')

        pathname = f"{self.directory}\\{self.problem_name}"

        if not os.path.exists(pathname):
            os.mkdir(pathname)
        
        output_file = pathname + '\\output.txt'
        best_fitness_plot = pathname + '\\best_fitness_plot'
        avg_fitness_plot = pathname + '\\avg_fitness_plot'
        diversity_plot = pathname + '\\diversity_plot'

        best_fitnesses = []
        avg_fitnesses = []
        max_dists = []
        num_candidates = []

        with open(output_file, 'w+') as f:

            print(f'Problem: {self.problem_name} | ', end='')
            f.write(f'Problem: {self.problem_name} | ')
            print(f'Population Size: {self.population_size} | ', end='')
            f.write(f'Population Size: {self.population_size} | ')
            print(f'Mutation Rate: {self.mutation_rate} | ', end='')
            f.write(f'Mutation Rate: {self.mutation_rate} | ')
            print(f'Crossover Rate: {self.crossover_rate}')
            f.write(f'Crossover Rate: {self.crossover_rate}\n\n')
            
            # initialize population
            self.initialization()
            
            # start evolution
            for gen in range(num_generations):
                # compute all fitness values
                fitnesses = self.fitness_function(self.population, self.durations, self.features, self.desired_duration)
                # print generation info
                print(f'Gen: {gen} | Number of Candidate Evals: {self.population_size*gen} | Best Fitness: {round(np.max(fitnesses), 3)} | Avg Fitness: {round(np.mean(fitnesses), 3)} | Diversity Score: {round(np.max(euclidean_distances(self.population, self.population)), 3)}')
                f.write(f'Gen: {gen} | Number of Candidate Evals: {self.population_size*gen} | Best Fitness: {round(np.max(fitnesses), 3)} | Avg Fitness: {round(np.mean(fitnesses), 3)} | Diversity Score: {round(np.max(euclidean_distances(self.population, self.population)), 3)}\n')

                best_fitnesses.append(round(np.max(fitnesses), 3))
                avg_fitnesses.append(round(np.mean(fitnesses), 3))
                max_dists.append(round(np.max(euclidean_distances(self.population, self.population)), 3))
                num_candidates.append(self.population_size * gen) # lambda child candidates are considered each generation

                # get a list of parent1's and parent2's using the parent_selection strategy
                p1s, p2s = self.parent_selection()

                # get a temporary crossover population of genomes (pre-mutation)
                max_n = math.ceil((self.representation_type.shape[0] // 2) * ((num_generations - gen) / num_generations))
                xover_pop = self.crossover(p1s, p2s, max_n)

                # now apply mutation strategy to the temporary population to get our new population
                mutation_rate = self.calc_mutation_rate(gen)
                lambda_pop = self.mutation(xover_pop, mutation_rate)

                # and replace our population with our new population based on the replacement strategy
                full_pop = np.vstack((self.population, lambda_pop))
                fitnesses = self.fitness_function(full_pop, self.durations, self.features, self.desired_duration)
                survivors = self.survivor_selection(fitnesses)
                self.population = full_pop[survivors]

                # now check if we've converged based on our convergence_bound
                if self.check_termination(convergence_bound):
                    gen += 1
                    print(f'Converged after {gen} generations.')
                    f.write(f'Converged after {gen} generations.\n')
                    break   

            # compute all final fitness values
            fitnesses = self.fitness_function(self.population, self.durations, self.features, self.desired_duration)
            # print final generation info
            print('-----------------------------------')
            f.write('-----------------------------------')
            print('Final Results')
            f.write('Final Results')
            print('-----------------------------------')
            f.write('-----------------------------------\n')
            print(f'Problem: {self.problem_name} | ', end='')
            f.write(f'Problem: {self.problem_name} | ')
            print(f'Population Size: {self.population_size} | ', end='')
            f.write(f'Population Size: {self.population_size} | ')
            print(f'Mutation Rate: {self.mutation_rate} | ', end='')
            f.write(f'Mutation Rate: {self.mutation_rate} | ')
            print(f'Crossover Rate: {self.crossover_rate}')
            f.write(f'Crossover Rate: {self.crossover_rate}\n\n')

            print(f'Gen: {gen+1} | Number of Candidate Evals: {self.population_size*gen} | Best Fitness: {round(np.max(fitnesses), 3)} | Avg Fitness: {round(np.mean(fitnesses), 3)} | Diversity Score: {round(np.max(euclidean_distances(self.population, self.population)), 3)}')
            f.write(f'Gen: {gen+1} | Number of Candidate Evals: {self.population_size*gen} | Best Fitness: {round(np.max(fitnesses), 3)} | Avg Fitness: {round(np.mean(fitnesses), 3)} | Diversity Score: {round(np.max(euclidean_distances(self.population, self.population)), 3)}\n')

            # create best fitness plot
            plt.plot(num_candidates, best_fitnesses, label='Best Fitness')
            plt.legend()
            plt.title('Best Fitness Score vs Candidate Count')
            plt.xlabel('Total Number of Candidates Considered')
            plt.ylabel('(Negative) Fitness')
            plt.savefig(best_fitness_plot)
            plt.close()

            # create avg fitness plot
            plt.plot(num_candidates, avg_fitnesses, label='Average Fitness')
            plt.legend()
            plt.title('Average Fitness Score vs Candidate Count')
            plt.xlabel('Total Number of Candidates Considered')
            plt.ylabel('(Negative) Fitness')
            plt.savefig(avg_fitness_plot)
            plt.close()

            # create diversity score plot
            plt.plot(num_candidates, max_dists, label='Diversity Score')
            plt.legend()
            plt.title('Diversity Score vs Candidate Count')
            plt.xlabel('Total Number of Candidates Considered')
            plt.ylabel('Max Distance Between Genomes')
            plt.savefig(diversity_plot)
            plt.close()

        results_dict = {'Problem': self.problem_name,
                        'Population Size': self.population_size,
                        'Mutation Rate': self.mutation_rate,
                        'Crossover Rate': self.crossover_rate,
                        'Generations': gen+1,
                        'Number of Candidate Evals': self.population_size*gen, 
                        'Best Fitness': round(np.max(fitnesses), 3), 
                        'Avg Fitness': round(np.mean(fitnesses), 3), 
                        'Diversity Score': round(np.max(euclidean_distances(self.population, self.population)), 3)}

        best_individual = np.argmax(fitnesses)

        return self.population[best_individual], results_dict