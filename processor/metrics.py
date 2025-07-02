import numpy as np
from scipy.spatial import distance

def hypervolume(population, reference_point):
    dominated_points = np.maximum(0, reference_point - population)
    return np.sum(np.prod(dominated_points, axis=1))

def diversity(population):
    population = np.array(population)
    if len(population) < 2:
        return 0

    distances = [distance.euclidean(population[i], population[i + 1])
                 for i in range(len(population) - 1)]
    return np.mean(distances)