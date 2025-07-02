import numpy as np
import random

# Global constants
MUTATION_PROB = 0.5
CROSSOVER_PROB = 0.9
MUTATION_DECAY = 0.99

class SimpleSolution:
    def __init__(self, candidate, fitness):
        self.candidate = candidate
        self.fitness = fitness

def standard_generator(rnd, args):
    p_times = args["processing_times"]
    num_jobs, num_ops, _ = p_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_ops):
            available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
            if len(available) > 0:
                machine = rnd.choice(available)
                job_schedule.append(machine)
        solution.append(job_schedule)
    return solution

def heuristic_generator(p_times):
    num_jobs, num_ops, _ = p_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_ops):
            available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
            if len(available) > 0:
                best_machine = available[np.argmin(p_times[job_idx, op_idx, available])]
                job_schedule.append(best_machine)
        solution.append(job_schedule)
    return solution

def generate_chaotic_sequence(length, seed=0.7, r=4.0):
    x = seed
    sequence = []
    for _ in range(length):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence

def chaotic_generator(rnd, args):
    p_times = args["processing_times"]
    num_jobs, num_ops, _ = p_times.shape
    total_ops = num_jobs * num_ops
    chaos = generate_chaotic_sequence(total_ops)
    solution = []
    idx = 0
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_ops):
            available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
            if len(available) > 0:
                chaos_value = chaos[idx]
                chosen_machine = available[int(chaos_value * len(available)) % len(available)]
                job_schedule.append(chosen_machine)
            idx += 1
        solution.append(job_schedule)
    return solution

def standard_evaluator(population, args):
    p_times = args["processing_times"]
    fits = []
    for sol in population:
        machine_times = np.zeros(p_times.shape[2])
        machine_usage = [[] for _ in range(p_times.shape[2])]
        job_end = np.zeros(p_times.shape[0])
        for job_idx, job_schedule in enumerate(sol):
            start_time = 0
            for op_idx, machine in enumerate(job_schedule):
                duration = p_times[job_idx, op_idx, machine]
                if np.isfinite(duration):
                    start_time = max(start_time, machine_times[machine])
                    machine_usage[machine].append((machine_times[machine], machine_times[machine] + duration))
                    machine_times[machine] = start_time + duration
                    start_time += duration
            job_end[job_idx] = start_time
        makespan = max(job_end)
        load_balance = np.std(machine_times)
        idle_time = sum(
            sum(b - a for a, b in zip(sorted([0]+[end for _, end in slots]),
                                      sorted([start for start, _ in slots]+[makespan])))
            for slots in machine_usage if slots
        )
        fits.append((makespan, load_balance, idle_time))
    return fits


def local_search(solution, p_times):
    new_solution = [row.copy() for row in solution]
    job_idx = np.random.randint(0, len(solution))
    op_idx = np.random.randint(0, len(solution[job_idx]))
    current_machine = new_solution[job_idx][op_idx]
    available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
    if len(available) > 1:
        choices = available[available != current_machine]
        new_solution[job_idx][op_idx] = np.random.choice(choices)
    return new_solution

def crossover_one_point(p1, p2, rnd):
    child = []
    for j_ops1, j_ops2 in zip(p1, p2):
        point = rnd.randint(1, len(j_ops1) - 1)
        child.append(j_ops1[:point] + j_ops2[point:])
    return child

def dominates(fitness1, fitness2):
    f1 = np.array(fitness1)
    f2 = np.array(fitness2)
    return np.all(f1 <= f2) and np.any(f1 < f2)

def binary_tournament_selection(population, fitness, rnd):
    idx1, idx2 = rnd.sample(range(len(population)), 2)
    f1, f2 = fitness[idx1], fitness[idx2]
    if dominates(f1, f2):
        return population[idx1]
    elif dominates(f2, f1):
        return population[idx2]
    return population[idx1] if rnd.random() < 0.5 else population[idx2]

def non_dominated_sorting(fitness):
    pop_size = len(fitness)
    fronts = [[]]
    domination_count = np.zeros(pop_size)
    dominates_set = [[] for _ in range(pop_size)]

    for p in range(pop_size):
        for q in range(pop_size):
            if p == q: continue
            if dominates(fitness[p], fitness[q]):
                dominates_set[p].append(q)
            elif dominates(fitness[q], fitness[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], None

def calculate_crowding_distance(front, fitness):
    d = np.zeros(len(front))
    for m in range(fitness.shape[1]):
        idx = np.argsort(fitness[front, m])
        d[idx[0]] = d[idx[-1]] = np.inf
        min_v, max_v = fitness[front, m][idx[0]], fitness[front, m][idx[-1]]
        if max_v == min_v: continue
        for i in range(1, len(front) - 1):
            d[idx[i]] += (fitness[front, m][idx[i + 1]] - fitness[front, m][idx[i - 1]]) / (max_v - min_v)
    return d

def calculate_knn_score(front_indices, fitness_np, k=5):
    """改进的knn分数计算"""
    scores = []
    n = len(front_indices)
    
    # 计算距离矩阵
    dist_matrix = cdist(fitness_np[front_indices], fitness_np[front_indices])
    np.fill_diagonal(dist_matrix, np.inf)
    
    for i in range(n):
        # 计算到k个最近邻的距离
        sorted_dist = np.sort(dist_matrix[i])
        k_nearest = sorted_dist[:k]
        
        # 避免距离为0
        score = np.sum(1 / (k_nearest + 1e-6))
        scores.append(score)
    
    return scores
