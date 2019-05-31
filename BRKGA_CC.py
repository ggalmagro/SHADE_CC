import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import copy as cp
import time

class BRKGA:

    def __init__(self, data, ml_const, cl_const, population_size, prt_elite, pbt_mutation, pbt_inherit, nb_clust, mu):

        self._data = data
        self._ml = ml_const
        self._cl = cl_const
        self._dim = data.shape[0]
        self._population_size = population_size
        self._prt_elite = prt_elite
        self._pbt_mutation = pbt_mutation
        self._pbt_inherit = pbt_inherit
        self._result_nb_clust = nb_clust
        self._mu = mu
        self._evals_done = 0

    #Funcionalidad para inicializar la poblacion de manera aleatoria
    def init_population(self):

        self._population = np.random.rand(self._population_size, self._dim)

    #Funcionalidad para decodificar la poblacion de individuos
    def decode_random_keys(self):

        return self.decode_single_random_key(self._population)
        #return np.ceil(self._population * self._result_nb_clust) - 1

    # Funcionalidad para decodificar un unico individuo
    def decode_single_random_key(self, cromosome):

        decoded = np.ceil(cromosome * self._result_nb_clust)
        decoded[decoded == 0] = 1
        return decoded - 1

    #Funcionalidad para evaluar la poblacion actual
    def get_fitness(self):

        fitness = np.array([])
        distances = np.array([])
        penaltys = np.array([])

        for i in range(self._population_size):

            aux_fitness, aux_dist, aux_penalty = self.get_single_fitness(
                self.decode_single_random_key(self._population[i, :]))

            fitness = np.append(fitness, aux_fitness)
            distances = np.append(distances, aux_dist)
            penaltys = np.append(penaltys, aux_penalty)

        return fitness, distances, penaltys


    # Funcionalidad para evaluar un unico individuo decodificado
    def get_single_fitness(self, cromosome):

        # Decodificamos el cromosoma
        #current_clustering = self.decode_random_key(cromosome)
        current_clustering = cromosome
        # Inicializamos la distancia media de las instancias de los clusters
        total_mean_distance = 0
        #Obtenemos el numero de clusters del clustering actual
        nb_clusters = len(set(current_clustering))

        # Para cada cluster en el clustering actual
        for j in set(current_clustering):
            # Obtener las instancias asociadas al cluster
            clust = self._data[current_clustering == j, :]

            if clust.shape[0] > 1:

                #Obtenemos la distancia media intra-cluster
                tot = 0.
                for k in range(clust.shape[0] - 1):
                    tot += ((((clust[k + 1:] - clust[k]) ** 2).sum(1)) ** .5).sum()

                avg = tot / ((clust.shape[0] - 1) * (clust.shape[0]) / 2.)

                # Acumular la distancia media
                total_mean_distance += avg

        # Inicializamos el numero de restricciones que no se satisfacen
        infeasability = 0

        # Calculamos el numero de restricciones must-link que no se satisfacen
        for c in range(np.shape(self._ml)[0]):

            if current_clustering[self._ml[c][0]] != current_clustering[self._ml[c][1]]:
                infeasability += 1

        # Calculamos el numero de restricciones cannot-link que no se satisfacen
        for c in range(np.shape(self._cl)[0]):

            if current_clustering[self._cl[c][0]] == current_clustering[self._cl[c][1]]:
                infeasability += 1

        # Calcular el valor de la funcion fitness
        distance = total_mean_distance / nb_clusters
        penalty = self._mu * self._dim * infeasability
        #penalty = distance * infeasability
        fitness = distance + penalty

        # Aumentar en uno el contador de evaluacions de la funcion objetivo
        self._evals_done += 1

        return fitness, distance, penalty

    #Operador de cruce aleatorio
    def uniform_crossover_operator(self, parent1, parent2):

        #Obtenemos el vector de probabilidades de herdar de parent1 y resolvemos las probabilidades
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]

        #Creamos el nuevo cromosoma como una copia de parent1
        new_cromosome = cp.deepcopy(parent1)

        #Copiamos los genes de parent2 indicados por las probabilidades obtenidas
        new_cromosome[v] = parent2[v]

        return new_cromosome

    def matching_crossover_operator(self, parent1, parent2):

        #Decodificamos los padres
        decoded_p1 = self.decode_single_random_key(parent1)
        decoded_p2 = self.decode_single_random_key(parent2)

        #Obtenemos las posiciones conicidesntes y no coincidentes de ambos padres
        matches = np.where(decoded_p1 == decoded_p2)
        non_matches = np.where(decoded_p1 != decoded_p2)

        #El nuevo individuo hereda las posiciones coincidentes (calculando la media)
        new_cromosome = self.uniform_crossover_operator(parent1, parent2)
        new_cromosome[matches] = (parent1[matches] + parent2[matches])/2

        return new_cromosome

    def get_offspring(self, elite, non_elite, offspring_size):

        #Obtenemos listas de indices aleatorios asociados a cromosomas elite y no-elite
        elite_cromosomes_index = np.random.randint(elite.shape[0], size=offspring_size)
        non_elite_cromosomes_index = np.random.randint(non_elite.shape[0], size=offspring_size)

        #Inicializamos la descendencia vacia
        offspring = np.empty((offspring_size, self._dim))

        #Generamos los nuevos inidividuos
        for i in range(offspring_size):

            #Obtenemos cada nuevo inidividuo como un cruce entre un cromosoma elitista y
            #uno no elitista
            new_cromosome = self.uniform_crossover_operator(elite[elite_cromosomes_index[i], :],
                                                    non_elite[non_elite_cromosomes_index[i], :])

            #Almacenamos el nuevo individuo
            offspring[i, :] = new_cromosome

        return offspring

    #Cuerpo principal del AG
    def run(self, max_eval, ls = False):

        #Inicializamos la poblacion y los parametros necesarios
        self._evals_done = 0
        self.init_population()
        fitness = self.get_fitness()[0]
        sorted_fitness = np.argsort(fitness)
        self._population = self._population[sorted_fitness, :]
        self._best = cp.deepcopy(self.decode_single_random_key(self._population[0, :]))
        self._best_fitness = fitness[sorted_fitness[0]]

        num_elite = int(self._population_size * self._prt_elite)
        num_mutants = int(self._population_size * self._pbt_mutation)
        offspring_size = self._population_size - num_elite - num_mutants
        generations = 0
        gen_times = []

        #Mientras no se haya alcanzado el numero maximo de evaluaciones
        while self._evals_done < max_eval:

            start_gen = time.time()
            #Guardar la elite de la generacion actual
            elite = self._population[:num_elite, :]
            non_elite = self._population[num_elite:, :]

            #Generar los mutantes de la nueva generacion
            mutants = np.random.rand(num_mutants, self._dim)

            #Generar los descendientes de la nueva generacion cruzando los miembros de la elite
            #con el resto de individuos
            offspring = self.get_offspring(elite, non_elite, offspring_size)

            #Introducimos los nuevos individuos en la poblacion conservando la elite
            self._population[num_elite:, :] = np.vstack((offspring, mutants))

            #Se evalua y reordena la poblacion
            fitness, distances, penaltys = self.get_fitness()
            sorted_fitness = np.argsort(fitness)
            self._population = self._population[sorted_fitness, :]

            #Actualizamos el mejor individuo si es necesario
            if fitness[sorted_fitness[0]] < self._best_fitness:

                self._best = cp.deepcopy(self.decode_single_random_key(self._population[0, :]))
                self._best_fitness = fitness[sorted_fitness[0]]

            #Ejecutamos busqueda local sobre la porblacion si se especifica
            if ls:

                self.local_search()

            generations += 1
            gen_times.append(time.time() - start_gen)

        return self.decode_single_random_key(self._population[sorted_fitness[0]]), self._best, np.median(gen_times) * generations

    #Busqueda local por trayectorias simples
    def local_search(self):

        for clust in range(self._population_size):

            current_clustering = cp.deepcopy(self.decode_single_random_key(self._population[clust, :]))
            current_fitness = self.get_single_fitness(current_clustering)[0]

            improvement = True
            object_index = 0

            while improvement and object_index < current_clustering.shape[0]:

                improvement = False
                original_label = current_clustering[object_index]

                for label in range(self._result_nb_clust):

                    if label != original_label:

                        current_clustering[object_index] = label
                        new_fitness = self.get_single_fitness(current_clustering)[0]

                        if new_fitness < current_fitness:

                            current_fitness = new_fitness
                            improvement = True

                        else:

                            current_clustering[object_index] = original_label

                object_index += 1

            if current_fitness < self._best_fitness:

                self._best = cp.deepcopy(current_clustering)
                self._best_fitness = current_fitness






