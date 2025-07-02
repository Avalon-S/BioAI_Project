from processor.run_algorithm_and_log import run_algorithm_and_log
from processor.metrics import hypervolume, diversity
from visualization.visualize_pareto import visualize_pareto_front_3d
from processor.data_processing import read_and_parse_fjsp_file
from methods.original_nsga2 import standard_nsga2
from methods.knn_nsga2 import advanced_nsga2
from methods.mo_aco import multiobjective_aco
from methods.nsga3 import standard_nsga3
from methods.spea2 import standard_spea2

from methods.ablation.wo_adap_crossover import advanced_nsga2_wo_adap_crossover
from methods.ablation.wo_archive import advanced_nsga2_wo_archive
from methods.ablation.wo_hybridinit import advanced_nsga2_wo_hybridinit
from methods.ablation.wo_knn import advanced_nsga2_wo_knn
from methods.ablation.wo_weight_schdule import advanced_nsga2_wo_weight_schdule


import os
import numpy as np
import json

SEED = 43
POP_SIZE = 50 # 50
N_GEN = 200 # 200

# 算法名称到函数和文件名前缀的映射
ALGO_REGISTRY = {
    "Standard NSGA-II": (standard_nsga2, "std_schedule"),
    "Advanced NSGA-II": (advanced_nsga2, "adv_schedule"),
    "Multiobjective ACO": (multiobjective_aco, "aco_schedule"),
    "NSGA-III": (standard_nsga3, "nsga3_schedule"),
    "SPEA-II": (standard_spea2, "spea2_schedule"),
    "without Adaptive Crossover": (advanced_nsga2_wo_adap_crossover, "wo_adap_crossover_schedule"),
    "without Archive": (advanced_nsga2_wo_archive, "wo_archive_schedule"),
    "without Hybrid Init": (advanced_nsga2_wo_hybridinit, "wo_hybridinit_schedule"),
    "without kNN": (advanced_nsga2_wo_knn, "wo_knn_schedule"),
    "without Weight Schedule": (advanced_nsga2_wo_weight_schdule, "wo_weight_schdule_schedule"),
    
}



def process_all_files(input_folder, output_folder, dataset_name, algorithms):
    files = [f for f in os.listdir(input_folder) if f.endswith(".fjs")]
    print(f"Processing {len(files)} files in dataset {dataset_name}\n")

    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(input_folder, filename)
        output_dir = os.path.join(output_folder, filename[:-4])
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{idx}/{len(files)}] Processing {filename}...")

        try:
            op_data, p_times, num_jobs, num_machines = read_and_parse_fjsp_file(file_path)
            avg_machines = np.isfinite(p_times).sum() / (num_jobs * p_times.shape[1])

            # 写入 dataset 信息
            dataset_info_path = os.path.join(output_dir, "dataset_info.json")
            with open(dataset_info_path, "w") as f:
                json.dump({
                    "dataset": dataset_name,
                    "jobs": num_jobs,
                    "machines": num_machines,
                    "avg_machines_per_op": round(avg_machines, 2)
                }, f, indent=4)

            results = {}
            population_dict = {}  # 用于多算法可视化

            # === 遍历指定算法 ===
            for algo_name in algorithms:
                if algo_name not in ALGO_REGISTRY:
                    print(f" Algorithm '{algo_name}' not recognized, skipping.")
                    continue

                algo_fn, prefix = ALGO_REGISTRY[algo_name]

                pop, archive, runtime = run_algorithm_and_log(
                    algo_fn, algo_name, p_times, num_machines,
                    output_dir, None, prefix, POP_SIZE, N_GEN, SEED
                )

                results[algo_name] = {
                    "runtime": runtime,
                    "hypervolume": None,  # 稍后统一计算
                    "diversity": diversity(pop),
                    "population": pop  # 暂存用于后续 HV/绘图
                }

                population_dict[algo_name] = pop

            if not population_dict:
                print(f"[{idx}] No valid algorithms ran. Skipping.")
                continue

            # === 计算参考点和更新 HV ===
            ref_point = np.max(np.vstack(list(population_dict.values())), axis=0)

            for algo_name in results:
                pop = results[algo_name]["population"]
                results[algo_name]["hypervolume"] = hypervolume(pop, ref_point)
                del results[algo_name]["population"]

            # === 写入 JSON
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(results, f, indent=4)

            # === 可视化 Pareto 前沿对比
            if len(population_dict) >= 2:
                visualize_pareto_front_3d(
                    population_dict=population_dict,
                    save_path=os.path.join(output_dir, f"comparison_multiple_algos.png"),
                    dataset_name=dataset_name
                )

            print(f"[{idx}] Done.\n")

        except Exception as e:
            print(f"[{idx}] Failed to process {filename}: {e}")
