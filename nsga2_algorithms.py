import time
import numpy as np
import random
from inspyred.ec.emo import NSGA2
from inspyred.ec import terminators

# ========== Global Parameters for Probability Adjustment ==========
DIVERSIFIED_INIT_PROB = 0.1  # Probability of heuristic generation
LOCAL_SEARCH_PROB = 0.05      # Probability of applying local search

# ========== Common Generator and Evaluator ==========
def standard_generator(random, args):
    """
    Standard NSGA-II generator: Randomly assign a machine to each operation of each job.
    """
    p_times = args["processing_times"]
    num_jobs, num_operations, num_machines = p_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_operations):
            available_machines = np.where(np.isfinite(p_times[job_idx, op_idx]))[0]
            if len(available_machines) > 0:
                machine_idx = random.choice(available_machines)
                job_schedule.append(machine_idx)
        solution.append(job_schedule)
    return solution


def standard_evaluator(candidates, args):
    """
    Standard NSGA-II evaluator: Calculate the objectives (Makespan and Load Balance) for each candidate solution.
    """
    p_times = args["processing_times"]
    fits = []
    for c in candidates:
        machine_times = np.zeros(p_times.shape[2])  # Total processing time for each machine
        job_end_times = np.zeros(p_times.shape[0])  # Completion time for each job

        for job_idx, job_schedule in enumerate(c):
            start_time = 0
            for op_idx, machine_idx in enumerate(job_schedule):
                op_time = p_times[job_idx, op_idx, machine_idx]
                if np.isfinite(op_time):
                    start_time = max(start_time, machine_times[machine_idx])
                    machine_times[machine_idx] = start_time + op_time
                    start_time += op_time
            job_end_times[job_idx] = start_time

        makespan = max(job_end_times)  # The longest job completion time
        load_balance = np.std(machine_times)  # Standard deviation of machine loads
        fits.append((makespan, load_balance))
    return fits

# ========== Standard NSGA-II ==========
def run_standard_nsga2(processing_times, pop_size, n_gen, seed):
    """
    Standard NSGA-II implementation
    """
    rnd = random.Random(seed)
    ea = NSGA2(rnd)
    ea.terminator = terminators.generation_termination

    start_time = time.time()
    final_pop = ea.evolve(
        generator=standard_generator,
        evaluator=standard_evaluator,
        pop_size=pop_size,
        maximize=False,
        max_generations=n_gen,
        processing_times=processing_times
    )
    runtime = time.time() - start_time
    return np.array([ind.fitness for ind in final_pop]), ea.archive, runtime

# ========== Improved NSGA-II ==========
def advanced_nsga2(processing_times, pop_size, n_gen, seed):
    """
    Improved NSGA-II: Diversified initialization (controlled by DIVERSIFIED_INIT_PROB) + local search (controlled by LOCAL_SEARCH_PROB)
    """
    rnd = random.Random(seed)
    ea = NSGA2(rnd)
    ea.terminator = terminators.generation_termination

    def diversified_generator(random, args):
        """
        Diversified initialization generator: Controlled by DIVERSIFIED_INIT_PROB
        """
        p_times = args["processing_times"]
        if random.random() < DIVERSIFIED_INIT_PROB:
            return heuristic_generator(p_times)  # Heuristic generation
        else:
            return standard_generator(random, args)  # Random generation

    def generator_with_local_search(random, args):
        """
        Generator with local search to further optimize the quality of solutions (controlled by LOCAL_SEARCH_PROB)
        """
        solution = diversified_generator(random, args)
        if random.random() < LOCAL_SEARCH_PROB:
            return local_search(solution, args["processing_times"])
        return solution

    start_time = time.time()
    final_pop = ea.evolve(
        generator=generator_with_local_search,
        evaluator=standard_evaluator,
        pop_size=pop_size,
        maximize=False,
        max_generations=n_gen,
        processing_times=processing_times
    )
    runtime = time.time() - start_time
    return np.array([ind.fitness for ind in final_pop]), ea.archive, runtime


def heuristic_generator(processing_times):
    """
    Heuristic generator: Assign the machine with the shortest processing time for each operation.
    """
    num_jobs, num_operations, num_machines = processing_times.shape
    solution = []
    for job_idx in range(num_jobs):
        job_schedule = []
        for op_idx in range(num_operations):
            available_machines = np.where(np.isfinite(processing_times[job_idx, op_idx]))[0]
            if len(available_machines) > 0:
                best_machine = available_machines[np.argmin(processing_times[job_idx, op_idx, available_machines])]
                job_schedule.append(best_machine)
        solution.append(job_schedule)
    return solution


def local_search(solution, processing_times):
    """
    Local search: Randomly select an operation and reassign a machine.
    """
    num_jobs = len(solution)
    num_machines = processing_times.shape[2]

    # Randomly select a job and an operation
    job_idx = np.random.randint(0, num_jobs)
    op_idx = np.random.randint(0, len(solution[job_idx]))

    current_machine = solution[job_idx][op_idx]
    available_machines = np.where(np.isfinite(processing_times[job_idx, op_idx]))[0]

    # If there are other available machines, randomly replace
    if len(available_machines) > 1:
        available_machines = available_machines[available_machines != current_machine]
        new_machine = np.random.choice(available_machines)
        solution[job_idx][op_idx] = new_machine

    return solution
