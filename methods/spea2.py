import numpy as np
import time, os, random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from methods.base_utils import (
    SimpleSolution, 
    standard_generator, 
    standard_evaluator,
    local_search, 
    crossover_one_point, 
    dominates,
    MUTATION_PROB, 
    CROSSOVER_PROB, 
    MUTATION_DECAY
)

def calculate_strength(fitness):
    """计算每个个体的强度值"""
    size = len(fitness)
    S = np.zeros(size, dtype=int)
    
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if dominates(fitness[i], fitness[j]):
                S[i] += 1
    return S

def calculate_raw_fitness(S, fitness):
    """计算原始适应度"""
    size = len(fitness)
    R = np.zeros(size)
    
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if dominates(fitness[j], fitness[i]):
                R[i] += S[j]
    return R

def calculate_density_estimation(fitness, k=None):
    """计算密度估计"""
    n = len(fitness)
    if k is None:
        k = int(np.sqrt(n))  # k = √n
    
    # 计算欧氏距离矩阵
    dist_matrix = cdist(fitness, fitness)
    
    # 将对角线设为无穷大（排除自身）
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 对每行排序，找到第k个最近邻
    sorted_dist = np.sort(dist_matrix, axis=1)
    sigma_k = sorted_dist[:, k]
    
    # 密度估计 = 1/(σ_k + 2)
    D = 1 / (sigma_k + 2)
    return D

def truncate_archive(fitness, archive_size):
    """截断存档（保留多样性）"""
    n = len(fitness)
    indices = list(range(n))
    
    # 计算距离矩阵
    dist_matrix = cdist(fitness, fitness)
    np.fill_diagonal(dist_matrix, np.inf)
    
    while len(indices) > archive_size:
        # 找到最小距离最小的个体
        min_dist = np.inf
        remove_idx = -1
        
        for i in indices:
            min_d = np.min(dist_matrix[i, indices])
            if min_d < min_dist:
                min_dist = min_d
                remove_idx = i
        
        # 移除该个体
        indices.remove(remove_idx)
    
    return indices

def evolve_spea2(p_times, pop_size, n_gen, seed,
                   generator_fn=standard_generator,
                   evaluator_fn=standard_evaluator):
    """标准SPEA2算法实现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rnd = random.Random(seed)
    args = {"processing_times": p_times}
    mutation_prob = MUTATION_PROB

    start_time = time.time()
    
    # 1. 初始化种群和存档
    population = [generator_fn(rnd, args) for _ in range(pop_size)]
    fitness = evaluator_fn(population, args)
    archive = []  # 初始为空存档
    
    pbar = tqdm(total=n_gen, desc="SPEA2")
    
    for gen in range(n_gen):
        # 2. 合并种群和存档
        combined_pop = population + [s.candidate for s in archive]
        combined_fitness = fitness + [s.fitness for s in archive]
        combined_fitness_np = np.array(combined_fitness)
        n_total = len(combined_pop)
        
        # 3. 计算强度值
        S = calculate_strength(combined_fitness)
        
        # 4. 计算原始适应度
        R = calculate_raw_fitness(S, combined_fitness)
        
        # 5. 计算密度估计
        D = calculate_density_estimation(combined_fitness_np)
        
        # 6. 计算总适应度 F = R + D
        F = R + D
        
        # 7. 环境选择：识别非支配解
        non_dominated_indices = [i for i in range(n_total) if R[i] < 1.0]
        
        # 8. 构建新存档
        new_archive = []
        
        if len(non_dominated_indices) <= pop_size:
            # 非支配解不足，补充支配解
            new_archive_indices = non_dominated_indices
            remaining = pop_size - len(non_dominated_indices)
            
            # 选择剩余个体中适应度最好的
            dominated_indices = [i for i in range(n_total) if i not in non_dominated_indices]
            sorted_dominated = sorted(dominated_indices, key=lambda i: F[i])
            new_archive_indices.extend(sorted_dominated[:remaining])
        else:
            # 非支配解过多，需要截断
            non_dominated_fitness = combined_fitness_np[non_dominated_indices]
            selected_indices = truncate_archive(non_dominated_fitness, pop_size)
            new_archive_indices = [non_dominated_indices[i] for i in selected_indices]
        
        # 9. 更新存档
        new_archive = [
            SimpleSolution(combined_pop[i], tuple(combined_fitness[i])) 
            for i in new_archive_indices
        ]
        
        # 10. 繁殖新种群
        children = []
        while len(children) < pop_size:
            # 二元锦标赛选择
            candidates = rnd.sample(new_archive, 2)
            p1, p2 = candidates[0].candidate, candidates[1].candidate
            
            # 交叉
            if rnd.random() < CROSSOVER_PROB:
                child = crossover_one_point(p1, p2, rnd)
            else:
                child = [op.copy() for op in p1]
            
            # 变异
            if rnd.random() < mutation_prob:
                child = local_search(child, p_times)
            
            children.append(child)
        
        # 11. 更新种群
        population = children
        fitness = evaluator_fn(population, args)
        archive = new_archive
        
        # 12. 更新变异概率
        mutation_prob *= MUTATION_DECAY
        pbar.update(1)
    
    pbar.close()
    
    # 13. 最终存档处理
    unique_archive = {}
    for s in archive:
        key = tuple(tuple(ops) for ops in s.candidate)
        unique_archive[key] = s
    
    archive = list(unique_archive.values())
    pareto_fitness = np.array([s.fitness for s in archive])
    
    return pareto_fitness, archive, time.time() - start_time

def standard_spea2(p_times, pop_size, n_gen, seed):
     return evolve_spea2(
        p_times=p_times,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        generator_fn=standard_generator,
        evaluator_fn=standard_evaluator
    )