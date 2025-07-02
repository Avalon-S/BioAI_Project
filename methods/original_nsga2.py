import time, os, random
import numpy as np
from tqdm import tqdm

from methods.base_utils import (
    SimpleSolution,
    standard_generator,
    standard_evaluator,
    local_search,
    dominates,
    binary_tournament_selection,
    crossover_one_point,
    non_dominated_sorting,
    calculate_crowding_distance,
    MUTATION_PROB,
    CROSSOVER_PROB,
)

def standard_nsga2(p_times, pop_size, n_gen, seed,
                   generator_fn=standard_generator,
                   evaluator_fn=standard_evaluator):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd = random.Random(seed)
    args = {"processing_times": p_times}

    start_time = time.time()

    # 1. 初始化种群
    population = [generator_fn(rnd, args) for _ in range(pop_size)]
    fitness = evaluator_fn(population, args)

    pbar = tqdm(total=n_gen, desc="NSGA-II")

    for _ in range(n_gen):
        children = []

        # 2. 生成子代
        while len(children) < pop_size:
            p1 = binary_tournament_selection(population, fitness, rnd)
            p2 = binary_tournament_selection(population, fitness, rnd)
            
            if rnd.random() < CROSSOVER_PROB:
                child = crossover_one_point(p1, p2, rnd)
            else:
                child = [op.copy() for op in p1]

            if rnd.random() < MUTATION_PROB:
                child = local_search(child, p_times)

            children.append(child)

        # 3. 评估子代
        child_fitness = evaluator_fn(children, args)

        # 4. 合并种群
        combined_population = population + children
        combined_fitness = fitness + child_fitness
        fitness_np = np.array(combined_fitness)

        # 5. 非支配排序
        fronts, _ = non_dominated_sorting(combined_fitness)

        # 6. 环境选择（选择新一代）
        new_population = []
        new_fitness = []

        for front in fronts:
            if len(new_population) + len(front) > pop_size:
                distances = calculate_crowding_distance(front, fitness_np)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                remaining = pop_size - len(new_population)
                selected = sorted_front[:remaining]
                new_population.extend([combined_population[idx] for idx, _ in selected])
                new_fitness.extend([combined_fitness[idx] for idx, _ in selected])
                break
            else:
                new_population.extend([combined_population[idx] for idx in front])
                new_fitness.extend([combined_fitness[idx] for idx in front])

        # 7. 更新种群
        population = new_population
        fitness = new_fitness

        pbar.update(1)

    pbar.close()

    # 8. 最终返回最后一代的帕累托前沿，并进行去重
    final_fronts, _ = non_dominated_sorting(fitness)
    pareto_indices = final_fronts[0]

    raw_solutions = [SimpleSolution(population[idx], tuple(fitness[idx])) for idx in pareto_indices]

    # 去重（基于调度结构）
    unique_set = set()
    pareto_solutions = []
    for sol in raw_solutions:
        key = str(sol.candidate)  # 结构唯一标识
        if key not in unique_set:
            unique_set.add(key)
            pareto_solutions.append(sol)

    pareto_fitness = np.array([s.fitness for s in pareto_solutions])
    return pareto_fitness, pareto_solutions, time.time() - start_time
