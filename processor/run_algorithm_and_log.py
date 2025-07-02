import os
import json
import numpy as np
import time
from visualization.visualize_schedule import visualize_schedule

def run_algorithm_and_log(
    algorithm_fn,
    algo_name,
    processing_times,
    num_machines,
    output_dir,
    log_file_path,
    schedule_prefix,
    pop_size,
    n_gen,
    seed
):
    print(f"  Running {algo_name}...")

    # === 为当前算法创建子目录（如 output_dir/Standard_NSAGA_II） ===
    algo_dir = os.path.join(output_dir, algo_name.replace(" ", "_"))
    os.makedirs(algo_dir, exist_ok=True)

    # === 开始计时并运行算法 ===
    population, archive, runtime = algorithm_fn(processing_times, pop_size, n_gen, seed)

    # === 保存为 JSON 格式 ===
    results = {
        "algorithm": algo_name,
        "runtime": runtime,
        "solutions": []
    }

    for i, solution in enumerate(archive):
        makespan, load_balance, idle_time = solution.fitness
        results["solutions"].append({
            "index": int(i + 1),
            "makespan": float(makespan),
            "load_balance": float(load_balance),
            "idle_time": float(idle_time),
            "schedule": [[int(x) for x in op] for op in solution.candidate]
        })

    json_save_path = os.path.join(algo_dir, f"{schedule_prefix}_results.json")
    with open(json_save_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # === 保存调度可视化图 ===
    print(f"  Saving {algo_name} scheduling visualizations...")
    for i, solution in enumerate(archive):
        makespan, load_balance, idle_time = solution.fitness
        schedule_path = os.path.join(algo_dir, f"{schedule_prefix}_{i + 1}.png")
        visualize_schedule(
            solution.candidate,
            processing_times,
            num_machines,
            makespan=makespan,
            load_balance=load_balance,
            idle_time=idle_time,
            save_path=schedule_path
        )

    return population, archive, runtime
