import time
import numpy as np
import random
import os
from tqdm import tqdm

# ========== Global Parameters ==========
DIVERSIFIED_INIT_PROB = 0.1     # Initial solution heuristic generation probability
LOCAL_SEARCH_PROB = 0.05        # Local search probability

MUTATION_PROB = 0.5             # Mutation probability
CROSSOVER_PROB = 0.9            # Crossover probability
MUTATION_DECAY = 0.99           # Reduce the probability of mutation each generation

# ========== Solution Representation ==========
class SimpleSolution:
    def __init__(self, candidate, fitness):
        self.candidate = candidate  # Scheduling plan (2D list)
        self.fitness = fitness      # [makespan, load balance]

# ========== Generator & Evaluator ==========
def standard_generator(rnd, args):
    """
    Standard NSGA-II generator: Randomly assign a machine to each operation of each job.
    """
    p_times = args["processing_times"]
    num_jobs, num_ops, num_machines = p_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_ops):
            available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0] # Available machines
            if len(available) > 0:
                machine = rnd.choice(available)  # Randomly select an available machine
                job_schedule.append(machine)
        solution.append(job_schedule)
    return solution

def heuristic_generator(p_times):
    """
    Heuristic generator: Assign the machine with the shortest processing time for each operation.
    """
    num_jobs, num_ops, _ = p_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_ops):
            available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
            if len(available) > 0:
                best_machine = available[np.argmin(p_times[job_idx, op_idx, available])] # Choose the shortest time in availible machines.
                job_schedule.append(best_machine)
        solution.append(job_schedule)
    return solution

def local_search(solution, p_times):  # Used as both random perturbation and a mutation operator.
    """
    Local search: Randomly select an operation and reassign a machine.
    """
    # Randomly select a process --> If the process can be processed on multiple machines --> Randomly choose a new availible machine.
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
        new_job = j_ops1[:point] + j_ops2[point:]
        child.append(new_job)
    return child

def standard_evaluator(population, args):
    """
    Standard NSGA-II evaluator: Calculate the objectives (Makespan and Load Balance) for each candidate solution.
    """
    p_times = args["processing_times"]
    fits = []
    for sol in population:
        machine_times = np.zeros(p_times.shape[2])
        job_end = np.zeros(p_times.shape[0])
        for job_idx, job_schedule in enumerate(sol):
            start_time = 0
            for op_idx, machine in enumerate(job_schedule):
                duration = p_times[job_idx, op_idx, machine]
                if np.isfinite(duration):
                    start_time = max(start_time, machine_times[machine])
                    machine_times[machine] = start_time + duration
                    start_time += duration
            job_end[job_idx] = start_time
        makespan = max(job_end)
        load_balance = np.std(machine_times)
        fits.append((makespan, load_balance))
    return fits

# ========== NSGA-II Core ==========
def dominates(ind1, ind2):
    ind1, ind2 = np.array(ind1), np.array(ind2)
    return np.all(ind1 <= ind2) and np.any(ind1 < ind2)

def non_dominated_sorting(fitness):
    pop_size = len(fitness)
    fronts = [[]]
    domination_count = np.zeros(pop_size)
    dominates_set = [[] for _ in range(pop_size)]
    rank = np.zeros(pop_size)

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
    return fronts[:-1], rank

def calculate_crowding_distance(front, fitness):
    d = np.zeros(len(front))
    for m in range(fitness.shape[1]):
        idx = np.argsort(fitness[front, m])
        d[idx[0]] = d[idx[-1]] = np.inf  # Boundary solution priority
        min_v, max_v = fitness[front, m][idx[0]], fitness[front, m][idx[-1]]
        if max_v == min_v: continue
        for i in range(1, len(front) - 1):
            d[idx[i]] += (fitness[front, m][idx[i + 1]] - fitness[front, m][idx[i - 1]]) / (max_v - min_v)
    return d

# ========== Main Evolution ==========
def evolve_nsga2(generator_fn, evaluator_fn, p_times, pop_size, n_gen, seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd = random.Random(seed)
    args = {"processing_times": p_times}
    mutation_prob = MUTATION_PROB
    
    start_time = time.time()
    
    population = [generator_fn(rnd, args) for _ in range(pop_size)] # Generate parents
    fitness = evaluator_fn(population, args)
    
    for _ in tqdm(range(n_gen), desc="NSGA-II Evolving"):
        # ========== Step 1: Create children ==========
        children = []
        while len(children) < pop_size:
            p1, p2 = rnd.sample(population, 2)
            if rnd.random() < CROSSOVER_PROB:
                child = crossover_one_point(p1, p2, rnd)
            else:
                child = [op.copy() for op in p1]
            if rnd.random() < mutation_prob:
                child = local_search(child, p_times)
            children.append(child)

        # ========== Step 2: Merge parents + children ==========
        combined_population = population + children
        combined_fitness = evaluator_fn(combined_population, args)
        fitness_np = np.array(combined_fitness)

        # ========== Step 3: Non-dominated sorting ==========
        fronts, _ = non_dominated_sorting(combined_fitness)

        # ========== Step 4: Elitism selection ==========
        new_population = []
        new_fitness = []

        for front in fronts:
            if len(new_population) + len(front) > pop_size:
                distances = calculate_crowding_distance(front, fitness_np)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                selected = sorted_front[:pop_size - len(new_population)]
                new_population.extend([combined_population[idx] for idx, _ in selected])
                new_fitness.extend([combined_fitness[idx] for idx, _ in selected])
                break
            else:
                new_population.extend([combined_population[idx] for idx in front])
                new_fitness.extend([combined_fitness[idx] for idx in front])

        population = new_population
        fitness = new_fitness
        mutation_prob *= MUTATION_DECAY  # The probability of mutation decreases with each generation

    # ========== Step 5: Extract final Pareto front ==========
    final_fitness = np.array(fitness)
    pareto_front = non_dominated_sorting(final_fitness)[0][0]
    archive = [SimpleSolution(population[i], tuple(final_fitness[i])) for i in pareto_front]

    # Eliminate duplicate scheduling plans (candidate)
    unique_archive = {}
    for sol in archive:
        key = tuple(map(tuple, sol.candidate))
        if key not in unique_archive:
            unique_archive[key] = sol
    archive = list(unique_archive.values())
    pareto_fitness = np.array([s.fitness for s in archive])

    return pareto_fitness, archive, time.time() - start_time

# ========== Wrapper API ==========
def run_standard_nsga2(p_times, pop_size, n_gen, seed):
    return evolve_nsga2(standard_generator, standard_evaluator, p_times, pop_size, n_gen, seed)

def advanced_nsga2(p_times, pop_size, n_gen, seed):
    def mixed_generator(rnd, args):
        if rnd.random() < DIVERSIFIED_INIT_PROB:
            return heuristic_generator(args["processing_times"])
        return standard_generator(rnd, args)

    def generator_with_ls(rnd, args):
        sol = mixed_generator(rnd, args)
        if rnd.random() < LOCAL_SEARCH_PROB:
            return local_search(sol, args["processing_times"])
        return sol

    return evolve_nsga2(generator_with_ls, standard_evaluator, p_times, pop_size, n_gen, seed)
