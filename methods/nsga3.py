import numpy as np
import time, os, random
from tqdm import tqdm
from .base_utils import (
    SimpleSolution, 
    standard_generator, 
    standard_evaluator,
    local_search, 
    crossover_one_point, 
    non_dominated_sorting,
    MUTATION_PROB, 
    CROSSOVER_PROB, 
    MUTATION_DECAY
)

def generate_reference_points(num_objs, divisions):
    ref_points = []
    for i in range(num_objs):
        point = np.zeros(num_objs)
        point[i] = 1.0
        ref_points.append(point)
    for d in range(1, divisions):
        for i in range(num_objs):
            point = np.zeros(num_objs)
            point[i] = d / divisions
            ref_points.append(point)
    center = np.ones(num_objs) / num_objs
    ref_points.append(center)
    return np.array(ref_points)

def find_extreme_points(fitness_np):
    num_objs = fitness_np.shape[1]
    extreme_points = []
    for i in range(num_objs):
        idx = np.argmin(fitness_np[:, i])
        extreme_points.append(fitness_np[idx])
    return np.array(extreme_points)

def calculate_intercepts(extreme_points, num_objs):
    """计算截距（用于标准化）"""
    A = extreme_points
    b = np.ones(num_objs)
    try:
        intercepts = np.linalg.solve(A, b)
        # 如果解中有 inf 或 nan，说明不可用
        if not np.all(np.isfinite(intercepts)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = np.max(A, axis=0)  # 退化为 max 值作为截距
        intercepts = np.maximum(intercepts, 1e-6)
    return intercepts


def normalize_fitness(fitness_np, ideal_point, intercepts):
    """标准化适应度值"""
    normalized = fitness_np - ideal_point
    normalized /= intercepts
    normalized = np.clip(normalized, 0, 1e6)  # 限制最大值
    normalized[~np.isfinite(normalized)] = 1e6  # 将 inf 或 nan 替换为大值
    return normalized


def associate_to_reference_points(normalized_fitness, ref_points):
    associations = []
    perpendicular_distances = []
    for f in normalized_fitness:
        dists = []
        for r in ref_points:
            w = r / np.linalg.norm(r)
            d = np.linalg.norm(f - np.dot(f, w) * w)
            dists.append(d)
        min_idx = np.argmin(dists)
        associations.append(min_idx)
        perpendicular_distances.append(dists[min_idx])
    return np.array(associations), np.array(perpendicular_distances)

def niching_selection(front, fitness_np, ref_points, ideal_point, intercepts, remaining):
    normalized_fitness = normalize_fitness(fitness_np[front], ideal_point, intercepts)
    associations, distances = associate_to_reference_points(normalized_fitness, ref_points)
    niche_counts = np.zeros(len(ref_points), dtype=int)
    for a in associations:
        niche_counts[a] += 1
    niches = {i: [] for i in range(len(ref_points))}
    for i, (idx, ref_idx, dist) in enumerate(zip(front, associations, distances)):
        niches[ref_idx].append((idx, dist))
    selected = []
    while len(selected) < remaining:
        min_niche = np.argmin(niche_counts)
        if niches[min_niche]:
            niches[min_niche].sort(key=lambda x: x[1])
            chosen_idx, _ = niches[min_niche].pop(0)
            selected.append(chosen_idx)
            niche_counts[min_niche] += 1
        else:
            niche_counts[min_niche] = np.iinfo(np.int32).max  # ✅ 使用最大合法整数
    return selected


def evolve_nsga3(generator_fn, evaluator_fn, p_times, pop_size, n_gen, seed, div=12):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd = random.Random(seed)
    args = {"processing_times": p_times}
    mutation_prob = MUTATION_PROB
    start_time = time.time()

    population = [generator_fn(rnd, args) for _ in range(pop_size)]
    fitness = evaluator_fn(population, args)
    fitness = [tuple(np.nan_to_num(f, nan=1e6, posinf=1e6, neginf=-1e6)) for f in fitness]
    fitness_np = np.array(fitness)

    ideal_point = np.min(fitness_np, axis=0)
    num_objs = len(fitness[0])
    ref_points = generate_reference_points(num_objs, div)
    pbar = tqdm(total=n_gen, desc="NSGA-III")

    for gen in range(n_gen):
        children = []
        while len(children) < pop_size:
            p1, p2 = rnd.sample(population, 2)
            child = crossover_one_point(p1, p2, rnd) if rnd.random() < CROSSOVER_PROB else [op.copy() for op in p1]
            if rnd.random() < mutation_prob:
                child = local_search(child, p_times)
            children.append(child)

        child_fitness = evaluator_fn(children, args)
        child_fitness = [tuple(np.nan_to_num(f, nan=1e6, posinf=1e6, neginf=-1e6)) for f in child_fitness]

        combined_population = population + children
        combined_fitness = fitness + child_fitness
        combined_fitness_np = np.array(combined_fitness)

        current_ideal = np.min(combined_fitness_np, axis=0)
        ideal_point = np.minimum(ideal_point, current_ideal)

        fronts, _ = non_dominated_sorting(combined_fitness)

        new_population = []
        new_fitness = []
        selected_count = 0

        for i, front in enumerate(fronts):
            if selected_count + len(front) <= pop_size:
                new_population.extend([combined_population[idx] for idx in front])
                new_fitness.extend([combined_fitness[idx] for idx in front])
                selected_count += len(front)
            else:
                remaining = pop_size - selected_count
                extreme_points = find_extreme_points(combined_fitness_np[front])
                intercepts = calculate_intercepts(extreme_points, num_objs)
                selected = niching_selection(front, combined_fitness_np, ref_points, ideal_point, intercepts, remaining)
                new_population.extend([combined_population[idx] for idx in selected])
                new_fitness.extend([combined_fitness[idx] for idx in selected])
                break

        population = new_population
        fitness = new_fitness
        fitness_np = np.array(fitness)
        mutation_prob *= MUTATION_DECAY
        pbar.update(1)

    pbar.close()
    final_fitness = np.array(fitness)
    fronts = non_dominated_sorting(fitness)[0]
    pareto_indices = fronts[0] if fronts else []
    archive = [SimpleSolution(population[i], tuple(final_fitness[i])) for i in pareto_indices]
    unique_archive = {tuple(map(tuple, s.candidate)): s for s in archive}
    archive = list(unique_archive.values())
    pareto_fitness = np.array([s.fitness for s in archive])
    return pareto_fitness, archive, time.time() - start_time

def standard_nsga3(p_times, pop_size, n_gen, seed):
    return evolve_nsga3(standard_generator, standard_evaluator, p_times, pop_size, n_gen, seed)
