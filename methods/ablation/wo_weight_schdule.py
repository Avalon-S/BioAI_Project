import time, os, random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

from methods.base_utils import (
    SimpleSolution, 
    standard_generator, 
    heuristic_generator, 
    chaotic_generator,
    standard_evaluator, 
    local_search, 
    crossover_one_point, 
    dominates,
    binary_tournament_selection,
    non_dominated_sorting,
    calculate_knn_score, 
    MUTATION_PROB, 
    CROSSOVER_PROB, 
    MUTATION_DECAY
)


def adaptive_crossover(p1, p2, rnd, gen_ratio, p_times):
    if gen_ratio < 0.3:
        return crossover_one_point(p1, p2, rnd)
    elif gen_ratio < 0.7:
        child = []
        for ops1, ops2 in zip(p1, p2):
            child_ops = [op1 if rnd.random() < 0.5 else op2 for op1, op2 in zip(ops1, ops2)]
            child.append(child_ops)
        return child
    else:
        child = []
        for job_idx, (ops1, ops2) in enumerate(zip(p1, p2)):
            child_ops = []
            for op_idx, (m1, m2) in enumerate(zip(ops1, ops2)):
                time1 = p_times[job_idx, op_idx, m1]
                time2 = p_times[job_idx, op_idx, m2]
                if np.isfinite(time1) and np.isfinite(time2):
                    child_ops.append(m1 if time1 < time2 else m2)
                elif np.isfinite(time1):
                    child_ops.append(m1)
                elif np.isfinite(time2):
                    child_ops.append(m2)
                else:
                    child_ops.append(rnd.choice([m1, m2]))
            child.append(child_ops)
        return child

def calculate_knn_score_k(front_indices, fitness_np, k=5):
    front_fitness = fitness_np[front_indices]
    if len(front_fitness) == 1:
        return [1.0]
    k = min(k, len(front_fitness) - 1)
    tree = KDTree(front_fitness)
    distances, _ = tree.query(front_fitness, k=k+1)
    k_nearest = distances[:, 1:]
    scores = np.sum(1 / (k_nearest + 1e-6), axis=1)
    return scores.tolist()

def calculate_crowding_distance(fitness):
    n = len(fitness)
    num_objs = len(fitness[0])
    crowding = np.zeros(n)
    if n <= 2:
        return [float('inf')] * n
    for m in range(num_objs):
        sorted_indices = np.argsort([f[m] for f in fitness])
        crowding[sorted_indices[0]] = float('inf')
        crowding[sorted_indices[-1]] = float('inf')
        fmin, fmax = fitness[sorted_indices[0]][m], fitness[sorted_indices[-1]][m]
        norm = fmax - fmin
        if norm < 1e-6:
            continue
        for i in range(1, n-1):
            idx = sorted_indices[i]
            next_idx = sorted_indices[i+1]
            prev_idx = sorted_indices[i-1]
            crowding[idx] += (fitness[next_idx][m] - fitness[prev_idx][m]) / norm
    return crowding.tolist()

def advanced_nsga2_wo_weight_schdule(p_times, pop_size, n_gen, seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd = random.Random(seed)
    args = {"processing_times": p_times}
    mutation_prob = MUTATION_PROB

    MAX_ARCHIVE_SIZE = 2 * pop_size
    OBJ_WEIGHTS = [1/3, 1/3, 1/3]

    start_time = time.time()

    population = []
    for _ in range(pop_size):
        r = rnd.random()
        if r < 0.3:
            population.append(heuristic_generator(p_times))
        elif r < 0.6:
            population.append(chaotic_generator(rnd, args))
        else:
            population.append(standard_generator(rnd, args))

    fitness = standard_evaluator(population, args)
    archive = []
    pbar = tqdm(total=n_gen, desc="w/o Weight-Schedule")

    for gen in range(n_gen):
        gen_ratio = gen / n_gen
        children = []

        while len(children) < pop_size:
            p1 = binary_tournament_selection(population, fitness, rnd)
            p2 = binary_tournament_selection(population, fitness, rnd)
            child = adaptive_crossover(p1, p2, rnd, gen_ratio, p_times)
            if rnd.random() < mutation_prob:
                child = local_search(child, p_times)
            children.append(child)

        child_fitness = standard_evaluator(children, args)
        combined_population = population + children
        combined_fitness = fitness + child_fitness
        fitness_np = np.array(combined_fitness)
        fronts, _ = non_dominated_sorting(combined_fitness)

        current_pareto = []
        for idx in fronts[0]:
            candidate = combined_population[idx]
            fit = tuple(combined_fitness[idx])
            current_pareto.append(SimpleSolution(candidate, fit))

        for sol in current_pareto:
            dominated = any(dominates(s.fitness, sol.fitness) for s in archive)
            if not dominated:
                archive = [s for s in archive if not dominates(sol.fitness, s.fitness)]
                archive.append(sol)

        if len(archive) > MAX_ARCHIVE_SIZE:
            archive_fitness = [s.fitness for s in archive]
            crowding = calculate_crowding_distance(archive_fitness)
            sorted_archive = sorted(zip(archive, crowding), key=lambda x: x[1], reverse=True)
            archive = [sol for sol, _ in sorted_archive[:MAX_ARCHIVE_SIZE]]

        new_population, new_fitness = [], []
        selected_count = 0

        for front in fronts:
            front_size = len(front)
            if selected_count + front_size <= pop_size:
                new_population.extend([combined_population[idx] for idx in front])
                new_fitness.extend([combined_fitness[idx] for idx in front])
                selected_count += front_size
            else:
                remaining = pop_size - selected_count
                k_val = max(3, min(10, front_size // 5))

                if gen_ratio < 0.3:
                    scores = calculate_knn_score_k(front, fitness_np, k=k_val)
                    sorted_front = sorted(zip(front, scores), key=lambda x: x[1], reverse=True)
                elif gen_ratio < 0.7:
                    knn_scores = calculate_knn_score_k(front, fitness_np, k=k_val)
                    convergence_scores = [np.dot(fitness_np[idx], OBJ_WEIGHTS) for idx in front]
                    norm_knn = (np.array(knn_scores) - np.min(knn_scores))
                    norm_knn /= (np.max(knn_scores) - np.min(knn_scores) + 1e-6)
                    norm_conv = (np.array(convergence_scores) - np.min(convergence_scores))
                    norm_conv /= (np.max(convergence_scores) - np.min(convergence_scores) + 1e-6)
                    
                    # 固定权重，不再使用 gen_ratio
                    diversity_weight = 0.5
                    convergence_weight = 0.5
                    
                    weights = [diversity_weight * n + convergence_weight * c
                               for n, c in zip(norm_knn, norm_conv)]
                    sorted_front = sorted(zip(front, weights), key=lambda x: x[1], reverse=True)
                else:
                    convergence_scores = [np.dot(fitness_np[idx], OBJ_WEIGHTS) for idx in front]
                    sorted_front = sorted(zip(front, convergence_scores), key=lambda x: x[1])

                selected = [idx for idx, _ in sorted_front[:remaining]]
                new_population.extend([combined_population[idx] for idx in selected])
                new_fitness.extend([combined_fitness[idx] for idx in selected])
                break

        population = new_population
        fitness = new_fitness
        mutation_prob *= MUTATION_DECAY
        pbar.update(1)

    pbar.close()

    if archive:
        unique_archive = {}
        for s in archive:
            key = tuple(map(tuple, s.candidate))
            unique_archive[key] = s
        archive = list(unique_archive.values())
        pareto_fitness = np.array([s.fitness for s in archive])
    else:
        fronts = non_dominated_sorting(fitness)[0]
        if fronts:
            pareto_front = fronts[0]
            archive = [SimpleSolution(population[i], tuple(fitness[i])) for i in pareto_front]
            unique_archive = {tuple(map(tuple, s.candidate)): s for s in archive}
            archive = list(unique_archive.values())
            pareto_fitness = np.array([s.fitness for s in archive])
        else:
            pareto_fitness = np.array([])
            archive = []

    return pareto_fitness, archive, time.time() - start_time
