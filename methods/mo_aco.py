import time
import os
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from methods.base_utils import (
    SimpleSolution,
    standard_evaluator,
    dominates,
    MUTATION_PROB,
    MUTATION_DECAY
)

def multiobjective_aco(p_times, pop_size, n_gen, seed,evaluator_fn=standard_evaluator,
                       alpha=1.0, beta=2.0, rho=0.1, q0=0.9):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    start_time = time.time()
    num_jobs, num_ops, num_machines = p_times.shape
    pheromone = np.ones((num_jobs, num_ops, num_machines))

    heuristic = np.zeros((num_jobs, num_ops, num_machines))
    for i in range(num_jobs):
        for j in range(num_ops):
            for k in range(num_machines):
                if np.isfinite(p_times[i, j, k]):
                    heuristic[i, j, k] = 1.0 / p_times[i, j, k]

    archive = []
    pbar = tqdm(total=n_gen, desc="MultiObjective-ACO")

    for _ in range(n_gen):
        solutions = []
        for ant in range(pop_size):
            solution = []
            for job_idx in range(num_jobs):
                job_schedule = []
                for op_idx in range(num_ops):
                    available = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]

                    if len(available) == 0:
                        machine = -1
                    elif len(available) == 1:
                        machine = available[0]
                    else:
                        probabilities = []
                        for machine in available:
                            tau = pheromone[job_idx, op_idx, machine]
                            eta = heuristic[job_idx, op_idx, machine]
                            probabilities.append((tau ** alpha) * (eta ** beta))

                        total = sum(probabilities)
                        if total > 0:
                            probabilities = [p / total for p in probabilities]
                        else:
                            probabilities = [1.0 / len(available)] * len(available)

                        if random.random() < q0:
                            machine = available[np.argmax(probabilities)]
                        else:
                            r = random.random()
                            cumulative = 0
                            machine = available[-1]
                            for idx, prob in enumerate(probabilities):
                                cumulative += prob
                                if r <= cumulative:
                                    machine = available[idx]
                                    break

                    job_schedule.append(machine)
                solution.append(job_schedule)
            solutions.append(solution)

        fitness_list = evaluator_fn(solutions, {"processing_times": p_times})

        for solution, fitness in zip(solutions, fitness_list):
            new_solution = SimpleSolution(solution, tuple(fitness))
            to_remove = []
            dominated = False

            for idx, arch_sol in enumerate(archive):
                if dominates(arch_sol.fitness, new_solution.fitness):
                    dominated = True
                    break
                elif dominates(new_solution.fitness, arch_sol.fitness):
                    to_remove.append(idx)

            for idx in sorted(to_remove, reverse=True):
                del archive[idx]

            if not dominated:
                archive.append(new_solution)

        pheromone *= (1.0 - rho)

        for solution in archive:
            sol = solution.candidate
            for job_idx in range(len(sol)):
                for op_idx in range(len(sol[job_idx])):
                    machine = sol[job_idx][op_idx]
                    if isinstance(machine, (int, np.integer)) and machine >= 0:
                        pheromone[job_idx, op_idx, machine] += 1.0

        q0 = max(0.1, q0 * 0.99)
        pbar.update(1)

    pbar.close()

    unique_archive = {}
    for s in archive:
        key = tuple(tuple(ops) for ops in s.candidate)
        unique_archive[key] = s

    archive = list(unique_archive.values())
    pareto_fitness = np.array([s.fitness for s in archive])

    return pareto_fitness, archive, time.time() - start_time
